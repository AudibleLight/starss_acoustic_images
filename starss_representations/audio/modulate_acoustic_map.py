#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Overlay acoustic map vs ground truth annotations
"""

from argparse import ArgumentParser
from pathlib import Path
from collections.abc import Sized, Iterable

import astropy.coordinates as coord
import astropy.units as u
import cv2
import matplotlib.pyplot as plt
import matplotlib.colors as cm
import matplotlib.patches as mpatches
import mpl_toolkits.basemap as basemap
import matplotlib.tri as tri
import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
import librosa
from h5py import File
from matplotlib.animation import FuncAnimation
from scipy.io import wavfile

from starss_representations import utils
from starss_representations.audio.generate_acoustic_image_dataset import get_field

DEFAULT_IN_FILE = utils.get_project_root() / "outputs/apgd_dev/dev-test-tau/fold4_room8_mix004.hdf"

# Mask anything from acoustic map that is not within this radius from a ground truth annotation
MASK_RADIUS = 30

# maximum number of frames to process: set to -1 to use all frames
DEFAULT_FRAME_CAP = 300
DEFAULT_DPI = 400


def read_from_hdf(hdf_file: File, dataset_name: str) -> np.ndarray:
    """
    Read a dataset from HDF file as a numpy array
    """

    retrieved = hdf_file[dataset_name]
    # Initialise an empty array and fill with the values from the dataset
    #  Note that this assumes the dataset is small enough to fit in memory ;)
    filled = np.empty(retrieved.shape)
    retrieved.read_direct(filled)
    return filled


def cart2pol(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> tuple:
    """
    Cartesian coordinates to Polar coordinates.
    """
    cart = coord.CartesianRepresentation(x, y, z)
    sph = coord.SphericalRepresentation.from_cartesian(cart)

    r = sph.distance.to_value(u.dimensionless_unscaled)
    colat = u.Quantity(90 * u.deg - sph.lat).to_value(u.rad)
    lon = u.Quantity(sph.lon).to_value(u.rad)

    return r, colat, lon


def cart2eq(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> tuple:
    """
    Cartesian coordinates to Equatorial coordinates.
    """
    r, colat, lon = cart2pol(x, y, z)
    lat = (np.pi / 2) - colat
    return r, lat, lon


# noinspection PyUnresolvedReferences
def wrapped_rad2deg(lat_r: np.ndarray, lon_r: np.ndarray) -> tuple:
    """
    Equatorial coordinate [rad] -> [deg] unit conversion.
    Output longitude guaranteed to lie in [-180, 180) [deg].
    """
    lat_d = coord.Angle(lat_r * u.rad).to_value(u.deg)
    lon_d = coord.Angle(lon_r * u.rad).wrap_at(180 * u.deg).to_value(u.deg)
    return lat_d, lon_d


def to_rgb(i: np.ndarray) -> np.ndarray:
    """
    Convert real-valued intensity array (per frequency) with shape (N_band, N_px) to color-band array (3, N_px)

    If N_band > 9, set N_band == 9
    """
    n_px = i.shape[1]
    # Create a copy of the input array if it's going to be modified
    i_copy = i.copy()
    if i.shape[0] != 9:
        i_copy = i_copy[1:10, :]  # grab 9 frequency bands
    # Reshape and sum to get (3, N_px)
    return i_copy.reshape((3, 3, n_px)).sum(axis=1)


def acoustic_map_to_rgb(apgd_arr: np.ndarray, normalize: bool = True, scaling: float = None) -> np.ndarray:
    """
    Convert intensity map with shape (n_bands, n_frames, n_px) to shape (3, n_frames)

    If normalise, scale so that maximum intensity across all frames == 1
    If scaling, multiply all values by a constant then clip to within original range
    """
    ip = np.zeros((apgd_arr.shape[1], 3, apgd_arr.shape[-1]))
    for fi in range(apgd_arr.shape[1]):
        vs__ = apgd_arr[:, fi, :]
        map_vs__ = to_rgb(vs__)
        ip[fi, :, :] = map_vs__

    # Normalize or not
    if normalize:
        ip /= ip.max()

    # Scale by a floating point value then clip to avoid overflow
    if scaling:
        orig_min, orig_max = ip.min(), ip.max()
        ip *= scaling
        ip = np.clip(ip, orig_min, orig_max)

    return ip


def cmap_from_list(name: list, colors: list, n: int = 256, gamma: float = 1.0):
    """
    Create segmented colormap from list of colors.
    """

    if not isinstance(colors, Iterable):
        raise ValueError('colors must be iterable')

    # List of value, color pairs
    if (
            isinstance(colors[0], Sized) and
            (len(colors[0]) == 2) and
            (not isinstance(colors[0], str))
    ):
        vals, colors = zip(*colors)
    else:
        vals = np.linspace(0, 1, len(colors))

    cdict = dict(red=[], green=[], blue=[], alpha=[])
    for val, color in zip(vals, colors):
        r, g, b, a = cm.to_rgba(color)
        cdict['red'].append((val, r, r))
        cdict['green'].append((val, g, g))
        cdict['blue'].append((val, b, b))
        cdict['alpha'].append((val, a, a))

    return cm.LinearSegmentedColormap(name, cdict, n, gamma)


def draw_ellipse(position, covariance, ax=None, **kwargs):
    """
    Draw an ellipse with a given position and covariance
    """
    ax = ax or plt.gca()
    # Convert covariance to principal axes
    if covariance.shape == (2, 2):
        u_, s, vt = np.linalg.svd(covariance)
        angle = np.degrees(np.arctan2(u_[1, 0], u_[0, 0]))
        width, height = 2 * np.sqrt(s)
        print(width, height)
    else:
        angle = 0
        width, height = 2 * np.sqrt(covariance)

    # Draw the Ellipse
    for nsig in range(1, 4):
        ell = mpatches.Ellipse(
            position,
            nsig * width,
            nsig * height,
            angle,
            **kwargs
        )
        print("Width", ell.width)
        ax.add_patch(ell)


def plot_gmm(
        gmm: GaussianMixture,
        x: np.ndarray,
        label: bool = True,
        ax: plt.Axes = None
) -> None:
    """
    Plot gaussian mixture model
    """
    ax = ax or plt.gca()
    labels = gmm.fit(x).predict(x)  # noqa
    if label:
        ax.scatter(x[:, 0], x[:, 1], c=labels, s=40, cmap='viridis', zorder=2)
    else:
        ax.scatter(x[:, 0], x[:, 1], s=40, zorder=2)
    ax.axis('equal')

    w_factor = 0.2 / gmm.weights_.max()
    for pos, covar, w in zip(gmm.means_, gmm.covariances_, gmm.weights_):
        draw_ellipse(pos, covar, alpha=w * w_factor)


def draw_map(
        i: np.ndarray,
        r: np.ndarray,
        lon_ticks: np.ndarray,
        show_labels: bool = False,
        show_axis: bool = False,
        fig: plt.Figure = None,
        ax: plt.Axes = None,
        kmeans: bool = False,
        gaussian_mixture: bool = False,
        az0: list[float] | float = None,
        el0: list[float] | float = None,
        mask_radius: float = 30,
        alpha: float = 0.9
) -> tuple:
    """
    Draw acoustic map with optional angular masking.
    """

    # Convert coordinate grid to elevation/azimuth (already spherical)
    _, r_el, r_az = cart2eq(*r)
    r_el, r_az = wrapped_rad2deg(r_el, r_az)
    r_el_min, r_el_max = np.around([np.min(r_el), np.max(r_el)])
    r_az_min, r_az_max = np.around([np.min(r_az), np.max(r_az)])

    # Save original layout
    orig_pos = ax.get_position()
    orig_aspect = ax.get_aspect()

    # Basemap init
    bm = basemap.Basemap(
        projection='mill',
        llcrnrlat=r_el_min,
        urcrnrlat=r_el_max,
        llcrnrlon=r_az_min,
        urcrnrlon=r_az_max,
        resolution='c',
        ax=ax
    )

    if show_axis:
        bm_labels = [1, 0, 0, 1]
        bm.drawparallels(
            np.linspace(r_el_min, r_el_max, 5),
            color='w',
            dashes=[1, 0],
            labels=bm_labels,
            labelstyle='+/-',
            textcolor='#565656',
            zorder=0,
            linewidth=2
        )
        bm.drawmeridians(
            lon_ticks,
            color='w',
            dashes=[1, 0],
            labels=bm_labels,
            labelstyle='+/-',
            textcolor='#565656',
            zorder=0,
            linewidth=2
        )

    if show_labels:
        ax.set_xlabel('Azimuth (degrees)', labelpad=20)
        ax.set_ylabel('Elevation (degrees)', labelpad=40)

    # Project spherical to 2D basemap coordinates
    r_x, r_y = bm(r_az, r_el)

    # optional masking
    if az0 is not None and el0 is not None:
        if isinstance(az0, float):
            az0 = np.ndarray([az0])
        if isinstance(el0, float):
            el0 = np.ndarray([el0])

        # Convert to radians
        az1 = np.deg2rad(r_az)
        el1 = np.deg2rad(r_el)
        az0r = np.deg2rad(az0)
        el0r = np.deg2rad(el0)

        # Spherical angular distance
        ang = np.arccos(
            np.sin(el0r[:, None]) * np.sin(el1[None, :]) +
            np.cos(el0r[:, None]) * np.cos(el1[None, :]) * np.cos(az1[None, :] - az0r[:, None])
        )
        ang_deg = np.rad2deg(ang)

        # compute mask, use logical OR across rows
        mask = (ang_deg <= mask_radius).any(axis=0)

        # apply mask
        i[:, ~mask] = 0

    # triangulation + plotting
    triangulation = tri.Triangulation(r_x, r_y)

    n_px = i.shape[1]
    mycmap = cmap_from_list('mycmap', i.T, n=n_px)
    colors_cmap = np.arange(n_px)

    ax.tripcolor(
        triangulation,
        colors_cmap,
        cmap=mycmap,
        shading='gouraud',  # removes interpolation patterns
        edgecolors='none',
        linewidth=0,
        alpha=alpha,
        zorder=100,
    )

    # annotate the target point (always)
    if az0 is not None and el0 is not None:
        for az0_, el0_ in zip(az0, el0):
            px, py = bm(az0_, el0_)
            ax.plot(px, py, 'ro', markersize=10, zorder=1000)
            ax.text(px, py, f" ({az0_}°, {el0_}°)", color="red", fontsize=9, zorder=1000)

    # clustering with kmeans/GMM if required
    cluster_center = None
    if kmeans:
        npts = 18  # find N maximum points
        i_s = np.square(i).sum(axis=0)
        max_idx = i_s.argsort()[-npts:][::-1]
        x_y = np.column_stack((r_x[max_idx], r_y[max_idx]))  # stack N max energy points
        km_res = KMeans(n_clusters=3).fit(x_y)  # apply k-means to max points
        # get center of the cluster of N points
        clusters = km_res.cluster_centers_  # noqa
        ax.scatter(r_x[max_idx], r_y[max_idx], c='b', s=5)  # plot all N points
        ax.scatter(clusters[:, 0], clusters[:, 1], s=500, alpha=0.3)  # plot the center as a large point
        cluster_center = bm(clusters[:, 0][0], clusters[:, 1][0], inverse=True)

    elif gaussian_mixture:
        npts = 18
        i_s = np.square(i).sum(axis=0)
        max_idx = i_s.argsort()[-npts:][::-1]
        x_y = np.column_stack((r_x[max_idx], r_y[max_idx]))  # stack N max energy points
        gmm = GaussianMixture(n_components=1, random_state=42)
        plot_gmm(gmm, x_y)

    # cosmetic cleanup
    ax.tick_params(left=False, right=False, top=False, bottom=False)
    ax.set_xticks([])
    ax.set_yticks([])
    for txt in ax.texts:
        txt.set_visible(False)

    # restore axis positions
    ax.set_position(orig_pos)
    ax.set(xticklabels=[], yticklabels=[])
    ax.set_aspect(orig_aspect)

    return fig, ax, cluster_center


def process_video(in_file: str | Path, out_file: str | Path, frame_cap) -> None:
    """
    Process a video file
    """

    def update(frame_idx: int) -> plt.Axes:
        if frame_idx % 10 == 0:
            print(f"Processing frame {frame_idx} / {frame_count if frame_cap == -1 else frame_cap}")

        ax.clear()

        # Get the current video frame and convert to RGB
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()

        if not ret:
            return ax

        # Need to flip the frame so that it matches the acoustic image
        frame = cv2.flip(frame, 1)

        # Show the frame
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        ax.imshow(
            cv2.flip(rgb, 1),
            extent=(0, 1, 0, 1),
            transform=ax.transAxes,
            zorder=0,
            aspect='auto'
        )

        # If we don't have any acoustic map at this frame, just return the axis
        if frame_idx > apgd_map.shape[0]:
            return ax

        # Get acoustic map for current frame: shape (3, n_px)
        vs = apgd_map[frame_idx, :, :]

        # Get any annotations at this frame: azimuth and elevation
        annot_at_frame = np.where(annotations[:, 0] == frame_idx)[0]

        # Little hack, use this to still draw a map but mask everything out
        #  used when we don't have any annotations at this frame
        if len(annot_at_frame) == 0:
            vs[:, :] = 0
            annot_az, annot_el = None, None
        else:
            annot_az = annotations[annot_at_frame, 3]
            annot_el = annotations[annot_at_frame, 4]

        # Draw acoustic map with annotations
        draw_map(
            r=r,
            i=vs,
            lon_ticks=lon_ticks,
            fig=fig,
            ax=ax,
            show_labels=False,
            show_axis=False,
            az0=annot_az,
            el0=annot_el,
            alpha=0.4,
            mask_radius=MASK_RADIUS,
        )

        # Flip the x-axis so the acoustic image looks correct
        ax.invert_xaxis()

        return ax

    with File(in_file, "r") as hdf:
        # (n_bands, n_frames, n_px)
        apgd = read_from_hdf(hdf, "ai_apgd").transpose((1, 2, 0))
        # (video frame idx, class idx, source number idx, azimuth, elevation, distance, unique source idx)
        annotations = read_from_hdf(hdf, "metadata")
        # (3, n_px)
        r = get_field()
        # Get samplerate directly from the HDF file
        sr = hdf.attrs.get("audio_sr")

    # Read eigenmike signal and convert to mono
    #  this is just for the reference audio played against the video
    eigen_path = utils.EIGEN_PATH / in_file.parent.name / in_file.name.replace(".hdf", ".wav")
    em32_out, _ = librosa.load(str(eigen_path), sr=sr, mono=True)

    # Truncate audio approximately according to frame cap
    if frame_cap != -1:
        em32_out = em32_out[:round(sr * (frame_cap * utils.VIDEO_FPS))]

    # Save audio to a temporary directory. Needed to mux later with FFmpeg
    wavfile.write("tmp.wav", sr, em32_out)

    # Convert the acoustic map to RGB and normalize so that max == 1
    #  this flattens to (n_frame, 3, n_px)
    apgd_map = acoustic_map_to_rgb(apgd, normalize=True, scaling=2)

    # Get the video too
    video_path = utils.VIDEO_PATH / in_file.parent.name / in_file.name.replace(".hdf", ".mp4")
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Create figure and axis: using high DPI removes edges in the triangle mesh
    fig, ax = plt.subplots(1, 1, figsize=(utils.DEFAULT_FIG_WIDTH, utils.DEFAULT_FIG_HEIGHT), dpi=DEFAULT_DPI)

    # Create evenly spaced longitudinal ticks
    lon_ticks = np.linspace(-180, 180, 5)

    # Create the animation and save to a temporary file
    anim = FuncAnimation(
        fig,
        update,
        frames=frame_cap if frame_cap > 0 else frame_count,
        interval=utils.VIDEO_FRAME_TIME * 1000,
        repeat=False
    )
    anim.save("tmp.mp4", dpi=DEFAULT_DPI)

    # Finally, combine both the animation and the wavfile together and remove the temporary files
    utils.combine_audio_and_video(
        "tmp.mp4", "tmp.wav", str(out_file), cleanup_video=True, cleanup_audio=True
    )


def main(input_file: str | Path, out_file: str | Path, frame_cap: int) -> None:
    process_video(Path(input_file), Path(out_file), frame_cap)


if __name__ == "__main__":
    # Set up argument parser
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-file",
        type=str,
        help="Path to a HDF file for a single recording",
        default=str(DEFAULT_IN_FILE)
    )
    parser.add_argument(
        "--out-file",
        type=str,
        help="Path to save output",
        default=str(DEFAULT_IN_FILE.parent / str(DEFAULT_IN_FILE.name).replace(".hdf", "_modulated.mp4"))
    )
    parser.add_argument(
        "--frame-cap",
        type=int,
        help=f"Maximum number of frames to process: set to -1 to process all frames",
        default=DEFAULT_FRAME_CAP
    )
    args = vars(parser.parse_args())

    main(**args)
