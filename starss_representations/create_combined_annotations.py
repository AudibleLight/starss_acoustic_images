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
import numpy as np
from scipy.io import wavfile
from h5py import File

from starss_representations.video.annotate_with_yolos import extract_bounding_boxes, animate_bounding_boxes
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
        yolos_orig, yolos_sanit = extract_bounding_boxes(video_file, frame_cap=5)
        yolos_anim = animate_bounding_boxes(
            video_file,
            yolos_sanit,
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
        annotations_map = create_annotations_map(meta_file)
        gt_anim = ground_truth_annotate(
            input_file=video_file,
            annotations_map=annotations_map,
            fig=fig_gt,
            ax=ax_gt,
            add_frame=True,
            # frame_cap=5
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
            hdf.attrs["nbands"] = apgd.shape[0]
            hdf.attrs["nframes_audio"] = apgd.shape[1]
            hdf.attrs["npx"] = apgd.shape[2]


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

