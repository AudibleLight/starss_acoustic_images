#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Convert generated acoustic images (with `generate_acoustic_image_dataset.py`) into 1) individual JSON files that can
be used during training, and 2) annotated videos showing the acoustic image overlaid against the original recording.
"""

import json
from argparse import ArgumentParser
from pathlib import Path

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

CMAP = (np.array(plt.get_cmap("tab10").colors) * 255).astype(int).tolist()

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


def get_circle_mask(tessellation_coords, azimuth_deg, elevation_deg):
    """
    Get a boolean mask for tessellation points within a circle around the DOA.

    Args:
        tessellation_coords: Array of shape (n_points, 3) with unit vectors
        azimuth_deg: Center azimuth in degrees
        elevation_deg: Center elevation in degrees

    Returns:
        Boolean mask of shape (n_points,) where True means inside circle
    """
    # Convert target direction to Cartesian
    target_x, target_y, target_z = spherical_to_cartesian(azimuth_deg, elevation_deg)
    target = np.array([target_x, target_y, target_z])

    # Calculate angular distances using dot product
    # For unit vectors: cos(angle) = dot product
    dot_products = tessellation_coords @ target

    # Clip to handle numerical errors
    dot_products = np.clip(dot_products, -1.0, 1.0)

    # Convert to angles in degrees
    angles_deg = np.degrees(np.arccos(dot_products))

    # Return mask for points within radius
    return angles_deg <= CIRCLE_RADIUS_DEG


def process_doa_matrix(matrix, metadata, ):
    """
    Process matrix according to DOA-based masking algorithm.

    Args:
        matrix: Array of shape (tessellation, bands, frames)
        metadata: Array of shape (n_items, 7) with columns [frame_idx, class_idx, source_idx, azimuth, elevation, distance, unique_idx]

    Returns:
        Processed z-scored and clipped values of shape (tessellation, frames)
    """
    tessellation, bands, frames = matrix.shape

    # Remove metadata rows outside the number of frames in the recording
    metadata = metadata[metadata[:, 0] < frames]

    # Create tessellation coordinates
    tessellation_coords = create_fibonacci_sphere(10)

    # Create a masked copy of the matrix (use np.nan for masked values)
    masked_matrix = matrix.astype(float).copy()

    # Get unique items
    unique_items = np.unique(metadata[:, -1])

    for item_idx in unique_items:
        # Get all metadata rows for this item
        item_metadata = metadata[metadata[:, -1] == item_idx]

        # Get frames where this item appears
        item_frames = item_metadata[:, 0].astype(int)

        # Get DOA for each frame this item appears in
        for i, frame_idx in enumerate(item_frames):
            azimuth = item_metadata[i, 3]
            elevation = item_metadata[i, 4]

            # Get circle mask for this DOA
            circle_mask = get_circle_mask(tessellation_coords, azimuth, elevation)

            # Set all tessellation points OUTSIDE the circle to NaN
            # circle_mask is True for inside, so ~circle_mask is True for outside
            masked_matrix[~circle_mask, :, frame_idx] = np.nan

    # Step 2: Compute median across bands (ignoring NaN)
    # Shape: (tessellation, frames)
    median_energy = np.nanmedian(masked_matrix, axis=1)

    # Step 3: Global z-scoring across all values
    # Flatten to compute global statistics
    all_values = median_energy.flatten()
    valid_values = all_values[~np.isnan(all_values)]

    if len(valid_values) == 0:
        raise ValueError("No valid values after masking")

    global_mean = np.nanmean(valid_values)
    global_std = np.nanstd(valid_values)

    if global_std == 0:
        raise ValueError("Standard deviation is zero, cannot z-score")

    z_scored = (median_energy - global_mean) / global_std

    # Step 4: Add 0.5 to z-score distribution
    z_scored = z_scored + 0.5

    # Step 5: Clip between 0 and 1
    clipped = np.clip(z_scored, 0, 1)

    return clipped


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


def draw_circle_on_equirectangular(
        img: np.ndarray, azimuth_deg: int, elevation_deg: int, color: np.ndarray = CMAP[0], thickness: int = 3
):
    """
    Draw a circle on an equirectangular image.

    Args:
        img: Image array (H, W, 3)
        azimuth_deg: Center azimuth in degrees
        elevation_deg: Center elevation in degrees
        color: BGR color tuple
        thickness: Line thickness

    Returns:
        Image with circle drawn
    """

    # Draw circle boundary
    circle_points_pixel = []
    for angle in np.linspace(0, 2 * np.pi, CIRCLE_N_POINTS):
        lat = np.radians(elevation_deg)
        lon = np.radians(azimuth_deg)
        d = np.radians(CIRCLE_RADIUS_DEG)

        bearing = angle
        lat2 = np.arcsin(np.sin(lat) * np.cos(d) +
                         np.cos(lat) * np.sin(d) * np.cos(bearing))
        lon2 = lon + np.arctan2(np.sin(bearing) * np.sin(d) * np.cos(lat),
                                np.cos(d) - np.sin(lat) * np.sin(lat2))

        # Convert to degrees
        az2 = np.degrees(lon2)
        el2 = np.degrees(lat2)

        # Convert to pixel coordinates
        x, y = spherical_to_equirectangular(az2, el2)
        circle_points_pixel.append([x, y])

    # Draw the circle
    circle_points_pixel = np.array(circle_points_pixel, dtype=np.int32)
    cv2.polylines(img, [circle_points_pixel], isClosed=True, color=color, thickness=thickness)

    # Draw center point
    center_x, center_y = spherical_to_equirectangular(azimuth_deg, elevation_deg)
    cv2.drawMarker(img, (center_x, center_y), color, markerType=cv2.MARKER_STAR, markerSize=thickness * 10,
                   thickness=thickness)

    # also return the points of the circle
    return img


def create_energy_heatmap_equirectangular(
        tessellation_coords: np.ndarray, energy_values: np.ndarray, colormap=cv2.COLORMAP_JET
) -> np.ndarray:
    """
    Create an equirectangular heatmap from tessellation energy values.

    Args:
        tessellation_coords: Array of shape (n_points, 3) with unit vectors
        energy_values: Array of shape (n_points,) with energy values (can contain NaN)
        colormap: OpenCV colormap to use

    Returns:
        RGBA heatmap overlay (H, W, 4) where alpha channel indicates valid data
    """
    # Convert tessellation coords to spherical
    x, y, z = tessellation_coords.T
    tess_azimuth, tess_elevation = cartesian_to_spherical(x, y, z)

    # Create empty heatmap
    heatmap = np.zeros((VIDEO_HEIGHT, VIDEO_WIDTH), dtype=np.float32)
    counts = np.zeros((VIDEO_HEIGHT, VIDEO_WIDTH), dtype=np.int32)

    # Map each tessellation point to pixel coordinates
    valid_mask = ~np.isnan(energy_values)

    pixels = []

    for i in range(len(tessellation_coords)):
        if not valid_mask[i]:
            continue

        az = tess_azimuth[i]
        el = tess_elevation[i]

        # Convert to pixel coordinates
        px, py = spherical_to_equirectangular(az, el, )

        pixels.append((px, py))

        # Accumulate energy values (with some spatial spreading for smoother visualization)
        radius = 15  # pixels to spread around each point
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                dist_sq = dx * dx + dy * dy
                if dist_sq > radius * radius:
                    continue

                new_y = py + dy
                new_x = (px + dx) % VIDEO_WIDTH  # Wrap around horizontally

                if 0 <= new_y < VIDEO_HEIGHT:
                    weight = np.exp(-dist_sq / (2 * (radius / 2) ** 2))  # Gaussian weight
                    heatmap[new_y, new_x] += energy_values[i] * weight
                    counts[new_y, new_x] += 1

    # Average where multiple points contributed
    mask = counts > 0
    heatmap[mask] /= counts[mask]

    # Normalize to 0-255 range
    if np.any(mask):
        valid_energies = heatmap[mask]
        vmin, vmax = valid_energies.min(), valid_energies.max()
        if vmax > vmin:
            heatmap[mask] = 255 * (heatmap[mask] - vmin) / (vmax - vmin)

    # Apply colormap
    heatmap_uint8 = heatmap.astype(np.uint8)
    colored_heatmap = cv2.applyColorMap(heatmap_uint8, colormap)

    # Create alpha channel (opaque where we have data)
    alpha = np.zeros((VIDEO_HEIGHT, VIDEO_WIDTH), dtype=np.uint8)
    alpha[mask] = 180  # Semi-transparent

    # Combine into RGBA
    heatmap_rgba = np.dstack([colored_heatmap, alpha])

    return heatmap_rgba, np.array(pixels)


def overlay_doa_on_frame(frame, metadata, metadata_frame_idx, matrix=None, show_energy=True):
    """
    Overlay DOA annotations and energy heatmap on a given video frame.
    """
    frame_metadata = metadata[metadata[:, 0] == metadata_frame_idx]

    tessellation_coords = create_fibonacci_sphere(10)

    # Overlay energy heatmap
    if show_energy and matrix is not None:
        if metadata_frame_idx < matrix.shape[2]:
            masked_matrix = matrix[:, :, metadata_frame_idx].astype(float).copy()

            # if len(frame_metadata) > 0:
            #     # Initialize a combined mask with all False
            #     combined_mask = np.zeros(masked_matrix.shape[0], dtype=bool)
            #
            #     for row in frame_metadata:
            #         azimuth, elevation = row[3], row[4]
            #         circle_mask = get_circle_mask(tessellation_coords, azimuth, elevation)
            #         # Combine masks using logical OR
            #         combined_mask |= circle_mask
            #
            #     # Set all tessellation points OUTSIDE any circle to NaN
            #     masked_matrix[~combined_mask, :] = np.nan

            energy_after = np.nanmean(masked_matrix, axis=1)

            heatmap_rgba, pixels = create_energy_heatmap_equirectangular(tessellation_coords, energy_after)

            alpha = heatmap_rgba[:, :, 3:4] / 255.0
            frame = (frame * (1 - alpha) + heatmap_rgba[:, :, :3] * alpha).astype(np.uint8)

    # Draw DOA circles
    for row in frame_metadata:
        azimuth, elevation = row[3], row[4]
        # opencv is annoying, needs a tuple of ints
        color = tuple(int(i) for i in CMAP[int(row[1])])

        # add the center marker on
        center_x, center_y = spherical_to_equirectangular(azimuth, elevation)
        frame = cv2.drawMarker(
            frame, (center_x, center_y), color, markerType=cv2.MARKER_STAR, markerSize=3 * 10, thickness=3
        )

    return frame


def generate_circle_boundary_points(azimuth_deg, elevation_deg, radius_deg=CIRCLE_RADIUS_DEG, n_points=50):
    """
    Generate points along a circle boundary on a sphere.

    Args:
        azimuth_deg: Center azimuth in degrees
        elevation_deg: Center elevation in degrees
        radius_deg: Radius of circle in degrees
        n_points: Number of points to generate along the boundary

    Returns:
        Array of (azimuth, elevation) coordinates along the circle boundary
    """
    # Convert center to radians
    lat = np.radians(elevation_deg)
    lon = np.radians(azimuth_deg)
    d = np.radians(radius_deg)

    # Generate points along the circle using spherical geometry
    boundary_points = []
    for angle in np.linspace(0, 2 * np.pi, n_points, endpoint=False):
        bearing = angle

        # Use spherical geometry formulas to find points at distance d
        lat2 = np.arcsin(np.sin(lat) * np.cos(d) +
                         np.cos(lat) * np.sin(d) * np.cos(bearing))
        lon2 = lon + np.arctan2(np.sin(bearing) * np.sin(d) * np.cos(lat),
                                np.cos(d) - np.sin(lat) * np.sin(lat2))

        # Convert back to degrees
        az2 = np.degrees(lon2)
        el2 = np.degrees(lat2)

        boundary_points.append((az2, el2))

    return np.array(boundary_points)


def process_frame_metadata(metadata: np.ndarray, matrix: np.ndarray, metadata_frame_idx: int, video_frame_idx: int) -> \
list[dict]:
    """
    Processes metadata for the frame at `metadata_frame_idx`.
    """
    frame_metadata = metadata[metadata[:, 0] == metadata_frame_idx]

    # Convert tessellation coords to spherical
    tessellation_coords = create_fibonacci_sphere(10)
    x, y, z = tessellation_coords.T
    tess_azimuth, tess_elevation = cartesian_to_spherical(x, y, z)

    # store all results inside here
    res = []

    # Get matrix results for this frame: shape (n_px, n_bands)
    masked_matrix = matrix[:, :, metadata_frame_idx].astype(float).copy()

    # iterate over every annotation in this frame
    for row in frame_metadata:
        # Unpack metadata row
        _, class_id, source_id, az_center, el_center, dist = row[:6]

        # get a mask where False == coordinate > 20deg from center, True == within
        circle_mask = get_circle_mask(tessellation_coords, az_center, el_center)

        # set tesselation values outside of the circle to nan
        current_poly = masked_matrix.copy()
        current_poly[~circle_mask, :] = np.nan

        # compute median energy per pixel across all bands
        energy_values = np.nanmedian(current_poly, axis=1)

        # find intensity of pixels inside circle
        valid_mask = ~np.isnan(energy_values)
        intensity_values = energy_values[valid_mask]

        # Generate circle boundary points in spherical coordinates
        boundary_spherical = generate_circle_boundary_points(az_center, el_center)

        # Convert boundary points to pixel coordinates
        polygon_coords = []
        for az_boundary, el_boundary in boundary_spherical:
            px, py = spherical_to_equirectangular(az_boundary, el_boundary)
            polygon_coords.append((px, py))

        polygon_coords = np.array(polygon_coords)

        # Get center in equirectangular pixel coordinates
        px_center, py_center = spherical_to_equirectangular(az_center, el_center)

        # construct results
        res_dict = {
            "video_id": 1,
            "video_frame_idx": int(video_frame_idx),
            "metadata_frame_idx": int(metadata_frame_idx),
            "category_id": int(class_id),
            "instance_id": int(source_id),
            "segmentation": polygon_coords.astype(np.int32),
            "center": (int(px_center), int(py_center)),
            "attributes": {
                "intensity": intensity_values.tolist(),
                "distance": int(dist)
            }
        }
        res.append(res_dict)

    return res


def create_acoustic_image_json_file(video_path, metadata, matrix, frame_cap: int = 500):
    """
    Create the target JSON file used during training for this recording
    """

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError("Could not open video")

    # Get properties from video
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Sanity check observed properties vs what we know from STARSS
    assert video_fps == VIDEO_FPS
    assert width == VIDEO_WIDTH
    assert height == VIDEO_HEIGHT

    # fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    # writer = cv2.VideoWriter(
    #     output_video_path,
    #     fourcc,
    #     VIDEO_FPS,
    #     (VIDEO_WIDTH, VIDEO_HEIGHT)
    # )

    # get frames which have annotations
    annotated_frames = np.unique(metadata[:, 0])

    print(f"Rendering {total_frames} video frames at {VIDEO_FPS:.2f} FPS")

    all_frames_metadata = []

    frames_to_iter = min([frame_cap, total_frames]) if frame_cap is not None else total_frames
    for video_frame_idx in range(frames_to_iter):

        # Grab next video frame
        # ret, frame = cap.read()
        # if not ret:
        #     break

        # Convert video frame to metadata frame
        metadata_frame_idx = video_frame_to_metadata_frame(video_frame_idx, )

        # Grab metadata for this frame and extend the list
        if metadata_frame_idx in annotated_frames:
            frame_metadatas = process_frame_metadata(metadata, matrix, metadata_frame_idx, video_frame_idx)

            # Draw the polygon and marker onto the video
            # for frame_metadata in frame_metadatas:
            #     color = CMAP[frame_metadata["category_id"]]
            #     frame = cv2.drawMarker(
            #         frame, frame_metadata["center"], color, markerType=cv2.MARKER_STAR, markerSize=30, thickness=3
            #     )
            #     frame = cv2.polylines(frame, [cv2.convexHull(frame_metadata["segmentation"])], isClosed=True,
            #                           color=color, thickness=3)

            # Also store the metadata
            all_frames_metadata.extend(frame_metadatas)

        # cv2.putText(
        #     frame,
        #     f"Video frame {video_frame_idx}/{total_frames} | Metadata frame {metadata_frame_idx}",
        #     (10, 30),
        #     cv2.FONT_HERSHEY_SIMPLEX,
        #     0.7,
        #     (255, 255, 255),
        #     2
        # )
        #
        # writer.write(frame)
        #
        # if video_frame_idx % 10 == 0:
        #     print(f"Rendered video frame {video_frame_idx}/{total_frames} | metadata frame {metadata_frame_idx}")

    # Normalise all the intensity scores
    all_intensities = np.array([x for xs in all_frames_metadata for x in xs["attributes"]["intensity"]])
    mean_intensity = np.mean(all_intensities)
    std_intensity = np.std(all_intensities)
    all_annotations_normalized = []
    for frame_metadata in all_frames_metadata:
        intensities = frame_metadata["attributes"]["intensity"]

        # z-score the intensity values
        intensities_norm = (intensities - mean_intensity) / std_intensity

        # shift the mean of the distribution onto 0.5
        intensities_norm += 0.5

        # clip the normalised values between 0 and 1
        intensities_clip = np.clip(intensities_norm, 0, 1)

        # update the frame data and append to the list
        frame_metadata["attributes"]["intensity"] = intensities_clip.tolist()
        frame_metadata["segmentation"] = frame_metadata["segmentation"].tolist()
        all_annotations_normalized.append(frame_metadata)

    finalised_annotations = {
        "videos": [
            {"id": 1, "file_name": video_path}
        ],
        "annotations": all_annotations_normalized
    }

    # dump the json
    with open("test_annotations_dcase.json", "w") as js_out:
        json.dump(finalised_annotations, js_out, indent=4, ensure_ascii=False)

    # finalise the video
    # cap.release()
    # writer.release()

    return finalised_annotations


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


def main(dataset_src: str, output_path: str):
    hdf_files = [p for p in Path(dataset_src).rglob("**/*.hdf")]
    # hdf_files = [Path("/home/huw-cheston/Documents/python_projects/starss_representations/outputs/apgd_dev/dev-train-tau/fold3_room13_mix009.hdf")]

    output_path = Path(output_path)
    if not output_path.exists():
        output_path.mkdir()

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

                # We can have multiple sound sources at each time so need to iterate again
                for metadata_row in metadata_[metadata_[:, 0] == metadata_frame_idx]:

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
