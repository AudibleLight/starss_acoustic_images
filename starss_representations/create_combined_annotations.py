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
from starss_representations.audio.annotate_with_apgd import generate_acoustic_map_video, get_visibility_matrix
from starss_representations import utils

DEFAULT_FIG_WIDTH, DEFAULT_FIG_HEIGHT = 10, 5
DEFAULT_OUTPATH = utils.get_project_root() / "outputs/combined"

# window length for processing audio
DEFAULT_AUDIO_TS = utils.VIDEO_FRAME_TIME / 10
DEFAULT_AUDIO_NBANDS = 9
DEFAULT_AUDIO_SCALE = "linear"

# maximum number of frames to process: set to -1 to use all frames
DEFAULT_FRAME_CAP = -1


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
        fig_yolos, ax_yolos = plt.subplots(nrows=1, ncols=1, figsize=(DEFAULT_FIG_WIDTH, DEFAULT_FIG_HEIGHT))
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
        fig_apgd, ax_apgd = plt.subplots(nrows=1, ncols=1, figsize=(DEFAULT_FIG_WIDTH, DEFAULT_FIG_HEIGHT))
        apgd_anim = generate_acoustic_map_video(
            apgd_arr=apgd,
            r=mapper,
            ts=audio_ts * 10000,
            fig=fig_apgd,
            ax=ax_apgd
        )

        # Annotate with ground truth
        fig_gt, ax_gt = plt.subplots(nrows=1, ncols=1, figsize=(DEFAULT_FIG_WIDTH, DEFAULT_FIG_HEIGHT))
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
        default=-1
    )

    # Parse arguments
    args = vars(parser.parse_args())

    main(**args)

