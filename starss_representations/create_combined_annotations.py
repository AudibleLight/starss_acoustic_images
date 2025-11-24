#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Create the following (synchronised) videos:

1) YOLOS bounding box annotations
2) APGD acoustic map
3) Ground truth metadata annotations
4) Everything combined
"""

import os
import subprocess
from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
from scipy.io import wavfile

from starss_representations.video.annotate_with_yolos import process_video as yolos_annotate
from starss_representations.video.annotate_with_ground_truth_metadata import process_video as ground_truth_annotate, create_annotations_map
from starss_representations.audio.annotate_with_apgd import generate_acoustic_map_video, get_visibility_matrix
from starss_representations import utils

DEFAULT_FIG_WIDTH, DEFAULT_FIG_HEIGHT = 10, 5
DEFAULT_OUTPATH = utils.get_project_root() / "outputs/combined"

AUDIO_TS = utils.VIDEO_FRAME_TIME / 10


def main(outpath: str | Path) -> None:
    # Sanitise output directory
    outdir = Path(outpath)
    utils.create_output_dir_with_subdirs(outdir)

    for f in utils.DESIRED_FILES:
        eigen_file = utils.EIGEN_PATH / (f + "_eigen.wav")
        meta_file = utils.METADATA_PATH / (f + ".csv")
        video_file = utils.VIDEO_PATH / (f + ".mp4")

        # Sanitise data files
        for fi in [eigen_file, meta_file, video_file]:
            assert fi.exists(), f"File {fi} does not exist!"

        # Annotate video with YOLOS
        fig_yolos, ax_yolos = plt.subplots(nrows=1, ncols=1, figsize=(DEFAULT_FIG_WIDTH, DEFAULT_FIG_HEIGHT))
        yolos_anim = yolos_annotate(
            video_file,
            fig=fig_yolos,
            ax=ax_yolos,
            add_frame=True,
            # frame_cap=5
        )

        # Annotate audio with APGD
        sr, eigen_sig = wavfile.read(eigen_file)
        _, apgd, mapper = get_visibility_matrix(
            eigen_sig,
            sr,
            apgd=True,
            t_sti=AUDIO_TS,
            scale="linear",
            nbands=9,
            # frame_cap=5
        )
        fig_apgd, ax_apgd = plt.subplots(nrows=1, ncols=1, figsize=(DEFAULT_FIG_WIDTH, DEFAULT_FIG_HEIGHT))
        apgd_anim = generate_acoustic_map_video(
            apgd_arr=apgd,
            r=mapper,
            ts=AUDIO_TS * 10000,
            fig=fig_apgd,
            ax=ax_apgd
        )

        # Annotate with ground truth
        fig_gt, ax_gt = plt.subplots(nrows=1, ncols=1, figsize=(DEFAULT_FIG_WIDTH, DEFAULT_FIG_HEIGHT))
        gt_anim = ground_truth_annotate(
            input_file=video_file,
            annotations_map=create_annotations_map(
                meta_file
            ),
            fig=fig_gt,
            ax=ax_gt,
            add_frame=True,
            # frame_cap=5
        )

        yolos_anim.save("yolos.mp4")
        apgd_anim.save("apgd.mp4")
        gt_anim.save("gt.mp4")

        output_fpath = str(outpath / video_file.parent.name / video_file.name)
        cmd = (
            f'ffmpeg -i gt.mp4 -i yolos.mp4 -i apgd.mp4 -i {str(eigen_file)} '
            f'-c:v libx264 -crf 23 -preset veryfast -c:a aac '
            f'-filter_complex "vstack=inputs=3[vout];[3:a]pan=stereo|c0=c0|c1=c1[aout]" '
            f'-map "[vout]" -map "[aout]" -y '
            f'{output_fpath}'
        )
        subprocess.call(cmd, shell=True,)

        os.remove("yolos.mp4")
        os.remove("apgd.mp4")
        os.remove("gt.mp4")


if __name__ == "__main__":
    # Set up argument parser
    parser = ArgumentParser(description="Generate HDF5 dataset for LAM training.")
    parser.add_argument(
        "--outpath",
        type=str,
        help="Path to the output HDF5 dataset",
        default=DEFAULT_OUTPATH
    )

    # Parse arguments
    args = vars(parser.parse_args())

    main(**args)

