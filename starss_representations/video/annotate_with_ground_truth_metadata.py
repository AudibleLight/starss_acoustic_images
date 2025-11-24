#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Combine video files with annotated ground truth metadata
"""

from argparse import ArgumentParser
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
from matplotlib.animation import FuncAnimation
from tqdm import tqdm

from starss_representations import utils

VIDEO_OUTPUT_DIR = utils.get_project_root() / "outputs/video_dev_annotated_ground_truth"
TEMP_OUTPUT_NAME = "tmp_out.mp4"


def format_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies formatting to all columns of the dataframe
    """
    # Set columns
    df.columns = utils.LABEL_COLUMNS

    # Express distance in terms of min-max
    max_distance = df["distance"].max()
    min_distance = df["distance"].min()
    df["distance_multiplier"] = ((df["distance"] - min_distance) / (max_distance - min_distance))

    return df


def create_annotations_map(metadata) -> dict[int, dict]:
    """
    Prepare annotations for fast lookup by actual video frame index

    This process builds a dictionary mapping each video frame index to a list of annotations that apply to it.
    """
    if isinstance(metadata, (str, Path)):
        metadata = pd.read_csv(metadata)

    metadata_fmt = format_df(metadata)

    frame_to_annotations_map = {}
    for idx, row in metadata_fmt.iterrows():
        # Calculate the start and end time of the label in seconds (from the metadata)
        label_frame_start_time = row["frame_number"] * utils.LABEL_RES
        label_frame_end_time = (row["frame_number"] + 1) * utils.LABEL_RES

        # Convert these label times to actual video frame indices
        video_start_idx = round(label_frame_start_time / utils.VIDEO_FRAME_TIME)
        video_end_idx = round(label_frame_end_time / utils.VIDEO_FRAME_TIME)

        # Store the annotation data for each video frame it applies to
        for video_idx in range(video_start_idx, video_end_idx):
            if video_idx not in frame_to_annotations_map:
                frame_to_annotations_map[video_idx] = []
            frame_to_annotations_map[video_idx].append(row.to_dict())

    return frame_to_annotations_map


# noinspection PyUnresolvedReferences
def process_video(
    input_file: str | Path,
    annotations_map: dict[int, dict],
    output_file: str | Path = None,
    fig: plt.Figure = None,
    ax: plt.Axes = None,
    add_frame: bool = True,
    frame_cap: int = None
) -> FuncAnimation:
    """
    Process a video frame-by-frame with all annotations
    """
    def update_video(frame_idx: int):
        if frame_idx % 100 == 0:
            print(f"Frame {frame_idx} / {n_frames}...")

        # Clear the axis for the current frame
        ax.clear()

        # Get the current frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()

        if not ret:
            return ax

        if add_frame:
            ax.imshow(frame)

        if frame_idx not in annotations_map.keys():
            return ax

        width, height = frame.shape[1], frame.shape[0]
        annotations = annotations_map[frame_idx]

        for annotation_row in annotations:
            # Extract pixel coordinates, distance multiplier, and active class from the annotation
            x, y = utils.spherical_to_pixel(
                annotation_row["azimuth"],
                annotation_row["elevation"],
                width,
                height
            )
            active_class = utils.LABEL_MAPPING_INV[annotation_row["active_class_idx"]]

            # Draw a circle at the annotated location
            ellip = mpatches.Ellipse(
                (x, y),
                width=10,
                height=10,
                edgecolor="red",
                linewidth=2,
                facecolor="none",
                zorder=10000
            )
            ax.add_patch(ellip)

            # Add text label for the active class
            ax.text(
                x + 10,
                y + 10,
                active_class,
                bbox=dict(facecolor='red', zorder=10000)
            )

        # Close everything after reaching the last frame
        if frame_idx == n_frames - 1:
            cap.release()

        return ax


    # Create the figure and axis if not provided
    if fig is None or ax is None:
        fig, ax = plt.subplots(1, 1)

    # Open the input video file for reading frames
    cap = cv2.VideoCapture(str(input_file))
    if not cap.isOpened():
        raise ValueError(f"Could not open video file {input_file}")

    # Compute frame count
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Create and save the animation
    fa = FuncAnimation(
        fig,
        update_video,
        # frames=200,
        frames=n_frames if not frame_cap else frame_cap,
        repeat=False,
        # interval needs to be provided in milliseconds
        interval=utils.VIDEO_FRAME_TIME * 1000
    )
    if output_file is not None:
        fa.save(output_file)

    return fa


def main(input_files: list[str], output_dir: str) -> None:
    # Create folder + subdirs if not existing
    output_dir = Path(output_dir)
    utils.create_output_dir_with_subdirs(output_dir, utils.DATA_SPLITS)

    # Run the pipeline over every file in the input
    for fi in tqdm(input_files, desc="Running pipeline..."):
        # Skip over outputs that already exist
        output_file = output_dir / f"{fi}.mp4"
        # if output_file.exists():
        #     continue

        # Load metadata and format it
        metadata = pd.read_csv(utils.METADATA_PATH / f"{fi}.csv")

        # Prepare annotations for fast lookup by actual video frame index
        # This process builds a dictionary mapping each video frame index to a list of annotations that apply to it.
        frame_to_annotations_map = create_annotations_map(metadata)

        # Process the video
        inpt = utils.VIDEO_PATH / f"{fi}.mp4"
        process_video(input_file=inpt, output_file=output_file, annotations_map=frame_to_annotations_map)

        # # Use ffmpeg to combine the temporary annotated video with its original audio track
        # utils.combine_audio_and_video(
        #     video_path=TEMP_OUTPUT_NAME,
        #     audio_path=utils.AUDIO_PATH / f"{fi}.wav",
        #     output_path=output_file,
        #     cleanup=True
        # )


if __name__ == "__main__":
    # Use module docstring for the help text
    parser = ArgumentParser(description=__doc__)

    # Here come the user parameters
    parser.add_argument(
        "--input-files",
        type=str,
        nargs="+",
        help="The name of the input files to use without extensions, e.g. 'dev-test-tau/fold4_room23_mix002'",
        default=utils.DESIRED_FILES,
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=VIDEO_OUTPUT_DIR,
    )
    vars__ = vars(parser.parse_args())

    main(**vars__)
