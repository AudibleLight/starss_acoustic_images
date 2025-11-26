#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Overlay acoustic map vs ground truth annotations
"""

from argparse import ArgumentParser
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from h5py import File
from matplotlib.animation import FuncAnimation

from starss_representations import utils
from starss_representations.audio.annotate_with_apgd import acoustic_map_to_rgb, draw_map


DEFAULT_IN_FILE = utils.get_project_root() / "outputs/combined/dev-train-tau/fold3_room14_mix003.hdf"

# Mask anything from acoustic map that is not within this radius from a ground truth annotation
MASK_RADIUS = 30

# maximum number of frames to process: set to -1 to use all frames
DEFAULT_FRAME_CAP = 30
DEFAULT_DPI = 200


def read_from_hdf(hdf_file: File, dataset_name: str) -> np.ndarray:
    """
    Read a dataset from HDF file as a numpy array
    """

    retrieved = hdf_file[dataset_name]
    # Initialise an empty array and fill with the values from the dataset
    #  Note that this assumes the dataset is small enough to fit in memory ;)
    filled = np.zeros(retrieved.shape)
    retrieved.read_direct(filled)
    return filled


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

        # Show the frame
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        ax.imshow(
            rgb,
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
            annot_az = annotations[annot_at_frame, 4]
            annot_el = annotations[annot_at_frame, 5]

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

        return ax

    with File(in_file, "r") as hdf:
        # (n_bands, n_frames, n_px)
        apgd = read_from_hdf(hdf, "apgd")
        # (video frame idx, metadata frame idx, class idx, source number idx, azimuth, elevation, distance, distance scaled)
        annotations = read_from_hdf(hdf, "annotations")
        # (3, n_px)
        r = read_from_hdf(hdf, "field")

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

    # Create the animation and save if required
    anim = FuncAnimation(
        fig,
        update,
        frames=frame_cap if frame_cap > 0 else frame_count,
        interval=utils.VIDEO_FRAME_TIME * 1000,
        repeat=False
    )
    anim.save(out_file, dpi=DEFAULT_DPI)


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
