#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Convert generated acoustic images (with `generate_acoustic_image_dataset.py`) into 1) individual JSON files that can
be used during training, and 2) annotated videos showing the acoustic image overlaid against the original recording.
"""

import json
from argparse import ArgumentParser
from pathlib import Path
from shutil import rmtree
from uuid import uuid4

import astropy.coordinates as coord
import astropy.units as u
import cv2
import matplotlib.pyplot as plt
import numpy as np
from h5py import File
from scipy.interpolate import griddata
from tqdm import tqdm

from starss_representations import utils

DEFAULT_DATASET_PATH = utils.get_project_root() / "outputs/apgd_dev"
DEFAULT_OUTPUT_PATH = utils.get_project_root() / "outputs/json_dev"

# Colormap: tab10 only has 10 colours, so we reflect it to have enough for every class
cm = (np.array(plt.get_cmap("tab10").colors) * 255).astype(int).tolist()
CMAP = [*cm, *cm, *cm]

# These are taken directly from the STARSS documentation
VIDEO_FPS = 29.97
METADATA_FPS = 10.0
# VIDEO_WIDTH, VIDEO_HEIGHT = 1920, 960
VIDEO_WIDTH, VIDEO_HEIGHT = 640, 320

# Radius of the circle drawn around annotated points
CIRCLE_RADIUS_DEG = 20.
CIRCLE_N_POINTS = 10

# POLYGON_MASK_THRESHOLD = 5e-6
POLYGON_MASK_THRESHOLD = 4e-5


def create_fibonacci_sphere(sh_order: int = 10, ):
    """Create approximately uniform points on a sphere using Fibonacci spiral."""
    if sh_order < 0:
        raise ValueError("Parameter[N] must be non-negative.")

    n_px = 4 * (sh_order + 1) ** 2
    n = np.arange(n_px)

    colat = np.arccos(1 - (2 * n + 1) / n_px)
    lon = (4 * np.pi * n) / (1 + np.sqrt(5))
    xyz = np.stack(pol2cart(1, colat, lon), axis=0)

    # these are the cartesian coordinates of the tesselation
    #  need to turn this into azimuth + elevation
    #  need to do the inverse of this: cart2pol
    #  sphere will have fewer points at the poles than expected
    #  to fill these, we need to do another interpolation
    return xyz.T


def pol2cart(r, colat, lon):
    lat = (np.pi / 2) - colat
    return eq2cart(r, lat, lon)


def eq2cart(r, lat, lon):
    r = np.array([r])  # if chk.is_scalar(r) else np.array(r, copy=False)
    if np.any(r < 0):
        raise ValueError("Parameter[r] must be non-negative.")

    return (
        coord.SphericalRepresentation(lon * u.rad, lat * u.rad, r)
        .to_cartesian()
        .xyz.to_value(u.dimensionless_unscaled)
    )


def spherical_to_cartesian(azimuth_deg, elevation_deg):
    """Convert spherical coordinates to Cartesian (x, y, z)."""
    az_rad = np.radians(azimuth_deg)
    el_rad = np.radians(elevation_deg)

    x = np.cos(el_rad) * np.cos(az_rad)
    y = np.cos(el_rad) * np.sin(az_rad)
    z = np.sin(el_rad)

    return x, y, z


def cartesian_to_spherical(x, y, z):
    """Convert Cartesian (x, y, z) to spherical (azimuth, elevation) in degrees."""
    azimuth = np.degrees(np.arctan2(y, x))
    elevation = np.degrees(np.arcsin(z))
    return azimuth, elevation


def metadata_frame_to_video_frame(metadata_frame_idx: int, ) -> int:
    """
    Convert metadata frame index to video frame index.

    Args:
        metadata_frame_idx: Frame index from metadata (100ms resolution = 10 fps)

    Returns:
        Video frame index (integer)
    """
    time_seconds = metadata_frame_idx / METADATA_FPS
    return int(time_seconds * VIDEO_FPS)


def video_frame_to_metadata_frame(video_frame_idx: int, ) -> int:
    """
    Convert video frame index to metadata frame index. Inverse of `metadata_frame_to_video_index`.

    Args:
        video_frame_idx: Frame index from video (29.97 fps)

    Returns:
        Metadata frame index (integer)
    """
    time_seconds = video_frame_idx / VIDEO_FPS
    return int(time_seconds * METADATA_FPS)


def spherical_to_equirectangular(azimuth_deg: int, elevation_deg: int, ) -> tuple[int, int]:
    """
    Convert spherical coordinates to equirectangular pixel coordinates.

    Args:
        azimuth_deg: Azimuth in degrees [-180, 180]
        elevation_deg: Elevation in degrees [-90, 90]

    Returns:
        (x, y) pixel coordinates
    """
    # Normalize azimuth from [-180, 180] to [0, img_width]
    # Azimuth 0° should be at center (x = img_width/2)
    # Azimuth -180° should be at left edge (x = 0)
    # Azimuth +180° should be at right edge (x = img_width)
    x = ((-azimuth_deg + 180) % 360) / 360.0 * VIDEO_WIDTH

    # Normalize elevation from [-90, 90] to [0, img_height]
    # Elevation +90° (up) should be at top (y = 0)
    # Elevation -90° (down) should be at bottom (y = img_height)
    y = (90 - elevation_deg) / 180.0 * VIDEO_HEIGHT

    return int(x), int(y)


def equirectangular_to_spherical(x: int, y: int) -> tuple[float, float]:
    """
    Convert equirectangular pixel coordinates back to spherical coordinates.

    Args:
        x: Pixel x-coordinate
        y: Pixel y-coordinate

    Returns:
        (azimuth_deg, elevation_deg)
    """
    azimuth_deg = 180.0 - (x / VIDEO_WIDTH) * 360.0
    elevation_deg = 90.0 - (y / VIDEO_HEIGHT) * 180.0
    return azimuth_deg, elevation_deg


def read_from_hdf(hdf: File, dataset: str) -> np.ndarray:
    retrieved = hdf[dataset]
    # Initialise an empty array and fill with the values from the dataset
    #  Note that this assumes the dataset is small enough to fit in memory ;)
    matrix_ = np.empty(retrieved.shape)
    retrieved.read_direct(matrix_)
    return matrix_


def interpolate_acoustic_image(acoustic_image: np.ndarray, tessellation: np.ndarray) -> np.ndarray:
    """
    Interpolate acoustic image from irregular tessellation to regular grid.

    Note that the median energy is selected per band, so that output array has shape (video_width, video_height,
    acoustic_image_frames).
    """
    n_px, n_bands, n_frames = acoustic_image.shape

    # Create regular target grid
    target_az = np.linspace(180, -180, VIDEO_WIDTH)
    target_el = np.linspace(90, -90, VIDEO_HEIGHT)

    target_az_grid, target_el_grid = np.meshgrid(target_az, target_el, indexing="xy")
    target_points = np.stack([target_az_grid.ravel(), target_el_grid.ravel()], axis=1)

    # Compute median over bands once (shape: 484, T)
    acoustic_image_medianed = np.median(acoustic_image, axis=1)

    # Interpolate all time steps at once
    # griddata expects values shape (n_points,) or (n_points, n_values)
    # We can pass (484, T) directly to interpolate all T frames simultaneously
    interpolated = griddata(
        tessellation,
        acoustic_image_medianed,
        target_points,
        method='linear',
        fill_value=0.
    )

    # Reshape from (W*H, T) to (H, W, T)
    output = interpolated.reshape(VIDEO_HEIGHT, VIDEO_WIDTH, n_frames)

    return output


def create_2d_gaussian(cx: int, cy: int) -> np.ndarray:
    # The circle should contain 2 SD of the vals (68-*95*-99.7% rule)
    sigma_deg = CIRCLE_RADIUS_DEG / 2.0

    deg_per_pixel_x = 360.0 / VIDEO_WIDTH
    deg_per_pixel_y = 180.0 / VIDEO_HEIGHT

    _, center_elevation_deg = equirectangular_to_spherical(cx, cy)

    x = np.arange(VIDEO_WIDTH)
    y = np.arange(VIDEO_HEIGHT)
    xx, yy = np.meshgrid(x, y, indexing="xy")  # (H, W)

    # Wrapped pixel deltas (preserve sign)
    dx = (xx - cx + VIDEO_WIDTH / 2) % VIDEO_WIDTH - VIDEO_WIDTH / 2
    dy = yy - cy

    # Convert to angular deltas
    delta_az_deg = -dx * deg_per_pixel_x  # azimuth increases leftward
    delta_el_deg = dy * deg_per_pixel_y

    cos_lat = np.cos(np.radians(center_elevation_deg))

    dist_sq_deg = (delta_el_deg ** 2) + (cos_lat * delta_az_deg) ** 2

    gaussian = np.exp(-dist_sq_deg / (2.0 * sigma_deg ** 2))

    return gaussian


def overlay_acoustic_mask(
        frame,
        acoustic_image,
        vmax: float,
        vmin: float,
        cmap=cv2.COLORMAP_JET,
        alpha_strength: float = 1.0,
):
    # 0 == fully transparent
    # alpha_strength = np.clip(alpha_strength, 0.0, 1.0)

    mask = ~np.isnan(acoustic_image)
    overlay_gray = np.zeros_like(acoustic_image, dtype=np.float32)

    if np.any(mask):
        if vmax > vmin:
            overlay_gray[mask] = np.clip((acoustic_image[mask] - vmin) / (vmax - vmin), 0, 1)

    overlay_gray = (overlay_gray * 255).astype(np.uint8)
    overlay_color = cv2.applyColorMap(overlay_gray, cmap)

    # Base alpha from intensity, scaled by user parameter
    alpha = (
            (overlay_gray.astype(np.float32) / 255.0)
            * alpha_strength
    )[..., None]

    blended = (
            frame.astype(np.float32) * (1.0 - alpha) +
            overlay_color.astype(np.float32) * alpha
    )

    return blended.astype(np.uint8)


def find_wrapped_contours(binary_mask: np.ndarray):
    """
    Find contours in an equirectangular mask with horizontal wrap-around, return contours in original coordinate space.
    """
    h, w = binary_mask.shape

    # Tile image horizontally (3x gives safety margin)
    tiled = np.hstack([binary_mask, binary_mask, binary_mask])

    contours, _ = cv2.findContours(
        tiled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    wrapped_contours = []

    for cnt in contours:
        # Shift contour back to center tile
        cnt = cnt.copy()
        cnt[:, 0, 0] -= w  # subtract one width

        # Keep contours that intersect original image range
        if np.any((cnt[:, 0, 0] >= 0) & (cnt[:, 0, 0] < w)):
            wrapped_contours.append(cnt)

    return wrapped_contours


def split_contour_on_seam(contour):
    """
    Split contour if it crosses the horizontal seam.
    Returns list of 1 or 2 contours.
    """
    xs = contour[:, 0, 0]

    if xs.max() - xs.min() < VIDEO_WIDTH / 2:
        # Normal contour, no wrap
        return [contour]

    # Split into left and right parts
    left = contour[xs < VIDEO_WIDTH * 0.5]
    right = contour[xs >= VIDEO_WIDTH * 0.5]

    if len(left) == 0 or len(right) == 0:
        return [contour]

    # Shift right part to negative space
    right_shifted = right.copy()
    right_shifted[:, 0, 0] -= VIDEO_WIDTH

    return [left, right_shifted]


def unwrap_contour_x(contour, width):
    """
    Unwrap contour x-coordinates so it is continuous across seam.
    """
    cnt = contour.astype(np.float32).copy()
    xs = cnt[:, 0, 0]

    # Detect large jumps (seam crossing)
    dx = np.diff(xs)
    jumps = np.where(np.abs(dx) > width / 2)[0]

    if len(jumps) == 0:
        return cnt

    # Unwrap by accumulating offsets
    offset = 0
    for i in range(len(xs)):
        if i > 0:
            if xs[i] - xs[i - 1] > width / 2:
                offset -= width
            elif xs[i - 1] - xs[i] > width / 2:
                offset += width
        cnt[i, 0, 0] = xs[i] + offset

    return cnt


def save_test_frame(test_idx: int, frame: np.ndarray) -> None:
    tc_folder = utils.get_project_root() / "test_cases" / f"tc{test_idx}"
    full_path = (tc_folder / str(uuid4())).with_suffix(".png")
    cv2.imwrite(full_path, frame)


def main(dataset_src: str, output_path: str):
    hdf_files = [p for p in Path(dataset_src).rglob("**/*.hdf")]
    # hdf_files = [Path("/home/huw-cheston/Documents/python_projects/starss_representations/outputs/apgd_dev/dev-train-tau/fold3_room13_mix009.hdf")]

    # Make output paths
    output_path = Path(output_path)
    if not output_path.exists():
        output_path.mkdir()

    # Create test case folders
    for i in range(1, 10):
        tc_folder = utils.get_project_root() / "test_cases" / "tc{i}".format(i=i)
        if tc_folder.exists():
            rmtree(tc_folder)
        tc_folder.mkdir(parents=True, exist_ok=True)

    for hdx_idx, hdf_file in enumerate(hdf_files):
        split = hdf_file.parent.name
        output_path_with_split = output_path / split
        if not output_path_with_split.exists():
            output_path_with_split.mkdir()

        dataset_res = {
            "videos": [],
            "annotations": []
        }

        # Update the video dict
        dataset_res["videos"].append({
            "id": hdx_idx,
            "file_name": hdf_file.name,
        })

        with File(hdf_file) as hdf:
            # Load up everything from HDF file
            # video_path_ = hdf.attrs["video_fpath"]
            video_path_ = str(hdf_file.with_suffix(".mp4")).replace("outputs/apgd_dev", "data/video_dev")
            metadata_ = read_from_hdf(hdf, "metadata")
            acoustic_image_matrix = read_from_hdf(hdf, "ai_apgd")

        # Load the acoustic image: shape (n_px, n_bands, n_frames)
        acoustic_image_trimmed = acoustic_image_matrix[:, :, :]

        # Get the tesselation coordinates and convert to spherical: shape (n_px, 2)
        tesselation = create_fibonacci_sphere(10)
        tesselation_eq = np.apply_along_axis(lambda x: cartesian_to_spherical(*x), 1, tesselation)

        # Create regular target grid
        target_az = np.linspace(180, -180, VIDEO_WIDTH)
        target_el = np.linspace(90, -90, VIDEO_HEIGHT)

        target_az_grid, target_el_grid = np.meshgrid(target_az, target_el, indexing="xy")
        target_points = np.stack([target_az_grid.ravel(), target_el_grid.ravel()], axis=1)

        # Compute median over bands once (shape: 484, T)
        acoustic_image_medianed = np.median(acoustic_image_trimmed, axis=1)

        # Compute loudest/quietest pixel statistics: used in the scaling
        valid_mask = ~np.isnan(acoustic_image_medianed)
        global_vmin = acoustic_image_medianed[valid_mask].min()
        global_vmax = acoustic_image_medianed[valid_mask].max()

        # Iterate over the frames we have ground truth annotations for
        frames_with_gt_annotations = np.unique(metadata_[:, 0])

        cap = cv2.VideoCapture(video_path_)
        if not cap.isOpened():
            raise IOError("Could not open video")

        # Get properties from video
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Sanity check observed properties vs what we know from STARSS
        assert video_fps == VIDEO_FPS

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(
            output_path_with_split / hdf_file.with_suffix(".mp4").name,
            fourcc,
            VIDEO_FPS,
            (VIDEO_WIDTH, VIDEO_HEIGHT)
        )

        # Iterate through every VIDEO frame we have
        for video_frame_idx in tqdm(range(total_frames)):

            # Grab next video frame
            ret, frame = cap.read()
            if not ret:
                break

            # Resize the video frame
            frame = cv2.resize(frame, (VIDEO_WIDTH, VIDEO_HEIGHT))

            # Convert to metadata index
            metadata_frame_idx = video_frame_to_metadata_frame(video_frame_idx)
            if metadata_frame_idx >= acoustic_image_medianed.shape[-1]:
                break

            # Interpolate the acoustic image for this frame
            acoustic_image_interpolated = griddata(
                tesselation_eq,
                acoustic_image_medianed[:, metadata_frame_idx],
                target_points,
                method='linear',
                fill_value=0.
            )
            # Need to reshape to get (height, width)
            acoustic_image_frame = acoustic_image_interpolated.reshape(VIDEO_HEIGHT, VIDEO_WIDTH)

            # If we've got metadata for this frame
            if metadata_frame_idx in frames_with_gt_annotations:

                n_polygons = 0

                # We can have multiple sound sources at each time so need to iterate again
                current_rows = metadata_[metadata_[:, 0] == metadata_frame_idx]
                for metadata_row in current_rows:

                    # Grab everything from the row of metadata
                    _, class_id, instance_id, gt_az, gt_el, gt_dist = metadata_row[:6]

                    # We'll need an annotations dictionary now
                    annotations_dict = {
                        "video_id": hdx_idx,
                        "video_frame_index": video_frame_idx,
                        "metadata_frame_index": metadata_frame_idx,
                        "instance_id": int(instance_id),
                        "category_id": int(class_id),
                        "segmentation": [],
                        "distance": gt_dist
                    }

                    # Grab the color for this class from our mapping
                    color = tuple(int(i) for i in CMAP[annotations_dict["category_id"]])

                    # Convert spherical azimuth/elevation to equirectangular
                    gt_az_eq, gt_el_eq = spherical_to_equirectangular(gt_az, gt_el)

                    # Compute the 2D Gaussian centered at (azimuth, elevation): shape (width, height)
                    gauss_gt = create_2d_gaussian(gt_az_eq, gt_el_eq)

                    # Multiply the acoustic image by the Gaussian to scale it
                    acoustic_image_gauss_scaled = acoustic_image_frame * gauss_gt

                    # Mask values in the scaled image that are below the threshold
                    acoustic_image_gauss_masked = acoustic_image_gauss_scaled.copy()
                    mask = np.where(acoustic_image_gauss_masked < POLYGON_MASK_THRESHOLD)
                    acoustic_image_gauss_masked[mask] = 0

                    # Compute the polygon boundaries: need to wrap around both sides of the image
                    binary_mask = (acoustic_image_gauss_masked > 0).astype(np.uint8) * 255
                    contours = find_wrapped_contours(binary_mask)

                    for cnt in contours:
                        split_contours = split_contour_on_seam(cnt)
                        for sc in split_contours:
                            sc = sc.copy()
                            sc[:, 0, 0] %= VIDEO_WIDTH

                            # Grab all pixels that are contained within the boundary
                            boundary = sc.squeeze()

                            # Skip over malformed boundaries
                            if boundary.ndim == 1:
                                continue

                            mask__ = np.zeros((VIDEO_HEIGHT, VIDEO_WIDTH), dtype=np.uint8)
                            cv2.fillPoly(mask__, [boundary.astype(np.int32)], 255)

                            # Stack to get (x, y, amplitude), with shape (N_coordinates, 3)
                            y_coords, x_coords = np.where(mask__ == 255)
                            amplitude_values = acoustic_image_gauss_masked[y_coords, x_coords]
                            pixels_data = np.column_stack([x_coords, y_coords, amplitude_values])
                            pixels_list = [[int(x), int(y), amp] for (x, y, amp) in pixels_data.tolist()]

                            annotations_dict["segmentation"].append(pixels_list)

                            # Draw the contour on the frame
                            frame = cv2.drawContours(frame, [sc], -1, color, 1)
                            n_polygons += 1

                    # Overlay Gaussian scaled acoustic image: use a different colormap
                    # TODO: reduce transparency
                    frame = overlay_acoustic_mask(
                        frame,
                        acoustic_image_gauss_scaled,
                        vmax=global_vmax,
                        vmin=global_vmin,
                        cmap=cv2.COLORMAP_WINTER,
                        alpha_strength=1.0
                    )

                    # Add the results to the overall dict
                    dataset_res["annotations"].append(annotations_dict)

                    # Test case 1: high polygon (elevation / height > 0.95)
                    if gt_el_eq / VIDEO_HEIGHT > 0.95 and len(contours) >= 1:
                        save_test_frame(1, frame)

                    # Test case 2: low polygon (elevation / height < 0.05)
                    if gt_el_eq / VIDEO_HEIGHT < 0.05 and len(contours) >= 1:
                        save_test_frame(2, frame)

                    # Test case 3: polygon wrapping at left edge of screen (azimuth / width < 0.05)
                    if gt_az_eq / VIDEO_WIDTH < 0.05 and len(contours) >= 1:
                        save_test_frame(3, frame)

                    # Test case 4: polygon wrapping at right edge of screen (azimuth / width > 0.95)
                    if gt_az_eq / VIDEO_WIDTH > 0.95 and len(contours) >= 1:
                        save_test_frame(4, frame)

                    # Test case 5: two polygons for one sound source
                    if gt_az_eq / VIDEO_WIDTH > 0.95 and len(contours) >= 1:
                        save_test_frame(5, frame)

                    # Test case 6: ground truth label with no polygons
                    if len(contours) == 0:
                        save_test_frame(6, frame)

                # Test case 7: three+ polygons for all sources on one frame
                if n_polygons >= 3:
                    save_test_frame(7, frame)

                # Test case 8: more annotations than polygons
                if len(current_rows) > n_polygons:
                    save_test_frame(8, frame)

                # Test case 9: more polygons than annotations
                if len(current_rows) < n_polygons:
                    save_test_frame(9, frame)

            # Write the frame
            writer.write(frame)

        # Dump the JSON file
        with open(output_path_with_split / hdf_file.with_suffix(".json").name, "w") as f:
            json.dump(dataset_res, f, indent=4, ensure_ascii=False)

        cap.release()
        writer.release()


if __name__ == "__main__":
    # Set up argument parser
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-src",
        type=str,
        help="Path to the processed data directory, should contain HDF files",
        default=DEFAULT_DATASET_PATH
    )
    parser.add_argument(
        "--output-path",
        type=str,
        help="Path to the store the output JSON files",
        default=DEFAULT_OUTPUT_PATH
    )
    # Parse arguments
    args = vars(parser.parse_args())

    main(**args)
