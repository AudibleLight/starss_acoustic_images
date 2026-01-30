#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Create the following (synchronised) videos:

1) YOLOS bounding box annotations
2) APGD acoustic map
3) Ground truth metadata annotations
"""

import os
import subprocess
from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile
from h5py import File

from starss_representations.video.annotate_with_yolos import extract_bounding_boxes, animate_bounding_boxes
from starss_representations.video.annotate_with_ground_truth_metadata import process_video as ground_truth_annotate, create_annotations_map
from starss_representations.audio.generate_acoustic_image_dataset import get_visibility_matrix
from starss_representations import utils

DEFAULT_OUTPATH = utils.get_project_root() / "outputs/combined"

# window length for processing audio
DEFAULT_AUDIO_TS = utils.VIDEO_FRAME_TIME / 10
DEFAULT_AUDIO_NBANDS = 9
DEFAULT_AUDIO_SCALE = "linear"

# maximum number of frames to process: set to -1 to use all frames
DEFAULT_FRAME_CAP = -1


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


def generate_acoustic_map_video(
        apgd_arr: np.ndarray,
        r: np.ndarray,
        ts: float,
        f_out: str | Path = None,
        fig: plt.Figure = None,
        ax: plt.Figure = None
) -> FuncAnimation:
    """
    Generate acoustic map as video
    """

    def update(frame_idx: int) -> plt.Axes:
        # Clear the current canvas
        ax.clear()

        # Get the current frame from the normalised RGB array
        vs = ip_norm[frame_idx, :, :]

        # Redraw frame
        draw_map(
            r=r,
            i=vs,
            lon_ticks=np.linspace(-180, 180, 5),
            fig=fig,
            ax=ax,
            show_labels=False,
            show_axis=True
        )

        # Set plot aesthetics
        ax.set(xticks=[], yticks=[], title="Acoustic Map", xticklabels=[], yticklabels=[])
        ax.invert_xaxis()
        fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95)

        return ax

    # Convert intensity map to RGB
    ip_norm = acoustic_map_to_rgb(apgd_arr, normalize=True)

    # Create figure and axis if not already existing
    if fig is None or ax is None:
        fig, ax = plt.subplots(1, 1)

    # Need to plot the first frame
    _ = update(0)

    # Create the animation and save if required
    anim = FuncAnimation(
        fig,
        update,
        frames=apgd_arr.shape[1],
        interval=ts,
        repeat=False
    )
    if f_out is not None:
        anim.save(f_out)

    return anim


def main(
    outpath: str | Path,
    audio_ts: float,
    audio_nbands: int,
    audio_scale: str,
    frame_cap: int
) -> None:
    """
    Compute all annotations for all videos.

    The following videos are created and combined into a single, syncronised file:
        1) YOLOS bounding box annotations
        2) APGD acoustic map
        3) Ground truth metadata annotations
    """
    # Sanitise output directory
    outdir = Path(outpath)
    utils.create_output_dir_with_subdirs(outdir)

    # Set frame cap to None if below 0
    if frame_cap < 0:
        frame_cap = None

    for f in utils.DESIRED_FILES:
        eigen_file = utils.EIGEN_PATH / (f + "_eigen.wav")
        meta_file = utils.METADATA_PATH / (f + ".csv")
        video_file = utils.VIDEO_PATH / (f + ".mp4")

        # Sanitise data files
        for fi in [eigen_file, meta_file, video_file]:
            assert fi.exists(), f"File {fi} does not exist!"

        # Annotate video with YOLOS
        fig_yolos, ax_yolos = plt.subplots(nrows=1, ncols=1, figsize=(utils.DEFAULT_FIG_WIDTH, utils.DEFAULT_FIG_HEIGHT))
        yolos_orig, yolos_sanit = extract_bounding_boxes(
            video_file,
            frame_cap=frame_cap
        )
        yolos_anim = animate_bounding_boxes(
            video_file,
            yolos_sanit,
            fig=fig_yolos,
            ax=ax_yolos,
            add_frame=True,
            frame_cap=frame_cap
        )

        # Annotate audio with APGD
        sr, eigen_sig = wavfile.read(eigen_file)
        _, apgd, mapper = get_visibility_matrix(
            eigen_sig,
            sr,
            apgd=True,
            t_sti=audio_ts,
            scale=audio_scale,
            nbands=audio_nbands,
            frame_cap=frame_cap
        )
        fig_apgd, ax_apgd = plt.subplots(nrows=1, ncols=1, figsize=(utils.DEFAULT_FIG_WIDTH, utils.DEFAULT_FIG_HEIGHT))
        apgd_anim = generate_acoustic_map_video(
            apgd_arr=apgd,
            r=mapper,
            ts=audio_ts * 10000,
            fig=fig_apgd,
            ax=ax_apgd
        )

        # Annotate with ground truth
        fig_gt, ax_gt = plt.subplots(nrows=1, ncols=1, figsize=(utils.DEFAULT_FIG_WIDTH, utils.DEFAULT_FIG_HEIGHT))
        annotations_map = create_annotations_map(meta_file)
        gt_anim = ground_truth_annotate(
            input_file=video_file,
            annotations_map=annotations_map,
            fig=fig_gt,
            ax=ax_gt,
            add_frame=True,
            frame_cap=frame_cap
        )

        yolos_anim.save("yolos.mp4")
        apgd_anim.save("apgd.mp4")
        gt_anim.save("gt.mp4")

        output_fpath = outpath / video_file.parent.name / video_file.name
        cmd = (
            f'ffmpeg -i gt.mp4 -i yolos.mp4 -i apgd.mp4 -i {str(eigen_file)} '
            f'-c:v libx264 -crf 23 -preset veryfast -c:a aac '
            f'-filter_complex "vstack=inputs=3[vout];[3:a]pan=stereo|c0=c0|c1=c1[aout]" '
            f'-map "[vout]" -map "[aout]" -y '
            f'{str(output_fpath)}'
        )
        subprocess.call(cmd, shell=True,)

        # Remove all temporary videos
        os.remove("yolos.mp4")
        os.remove("apgd.mp4")
        os.remove("gt.mp4")

        # Convert annotations map into a numpy array from dict
        annotations_map_np = []
        for idx, vals in annotations_map.items():
            for v in vals:
                # video frame idx, metadata frame idx, class idx, source number idx, azimuth, elevation, distance, distance scaled
                annotations_map_np.append((idx, *list(v.values())))
        annotations_map_np = np.array(annotations_map_np)

        # Stack everything into a single HDF file for this recording
        with File(output_fpath.with_suffix(".hdf"), "w") as hdf:
            # Create all individual datasets
            hdf.create_dataset("em32", data=eigen_sig)
            hdf.create_dataset("apgd", data=apgd)
            hdf.create_dataset("field", data=mapper)
            hdf.create_dataset("annotations", data=annotations_map_np)
            hdf.create_dataset("yolos_orig", data=yolos_orig)
            hdf.create_dataset("yolos_sanitised", data=yolos_sanit)

            # Create attributes
            hdf.attrs["sr"] = sr
            hdf.attrs["audio_nbands"] = apgd.shape[0]
            hdf.attrs["audio_ts"] = audio_ts
            hdf.attrs["audio_nframes"] = apgd.shape[1]
            hdf.attrs["audio_npx"] = apgd.shape[2]
            hdf.attrs["audio_scale"] = audio_scale


if __name__ == "__main__":
    # Set up argument parser
    parser = ArgumentParser(description="Generate HDF5 dataset for LAM training.")
    parser.add_argument(
        "--outpath",
        type=str,
        help="Path to the output HDF5 dataset",
        default=DEFAULT_OUTPATH
    )
    parser.add_argument(
        "--audio-ts",
        type=float,
        help=f"Audio window length, defaults to {round(DEFAULT_AUDIO_TS)}",
        default=DEFAULT_AUDIO_TS
    )
    parser.add_argument(
        "--audio-nbands",
        type=int,
        help=f"Number of bands to use when processing audio, defaults to {DEFAULT_AUDIO_NBANDS}",
        default=DEFAULT_AUDIO_NBANDS
    )
    parser.add_argument(
        "--audio-scale",
        type=str,
        help=f"Frequency scale to use when processing audio, defaults to {DEFAULT_AUDIO_SCALE}",
        default=DEFAULT_AUDIO_SCALE
    )
    parser.add_argument(
        "--frame-cap",
        type=int,
        help=f"Maximum number of frames to process: set to -1 to process all frames",
        default=DEFAULT_FRAME_CAP
    )

    # Parse arguments
    args = vars(parser.parse_args())

    main(**args)

