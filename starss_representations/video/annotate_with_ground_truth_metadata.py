#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Combine video files with annotated ground truth metadata
"""

from argparse import ArgumentParser
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
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


def create_annotations_map(metadata_fmt: pd.DataFrame) -> dict[int, dict]:
    """
    Prepare annotations for fast lookup by actual video frame index

    This process builds a dictionary mapping each video frame index to a list of annotations that apply to it.
    """
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


def annotate_frame(frame: np.ndarray, annotations: list[dict]) -> np.ndarray:
    """
    Applies annotation to a given frame at an index
    """
    width, height = frame.shape[1], frame.shape[0]

    for annotation_row in annotations:
        # Extract pixel coordinates, distance multiplier, and active class from the annotation
        x, y = utils.spherical_to_pixel(
            annotation_row["azimuth"],
            annotation_row["elevation"],
            width,
            height
        )
        distance_multiplier = annotation_row["distance_multiplier"]
        active_class = utils.LABEL_MAPPING_INV[annotation_row["active_class_idx"]]

        # Calculate annotation color based on distance (closer = darker color)
        color_val = int(utils.MAX_COLOR * distance_multiplier) + 10
        color_val = utils.MAX_COLOR - color_val

        # Draw a circle at the annotated location (OpenCV uses BGR color format)
        frame = cv2.circle(
            frame,
            (x, y),
            radius=5,
            color=(color_val, 50, 50),
            thickness=2
        )

        # Add text label for the active class
        frame = cv2.putText(
            frame,
            active_class,
            (x + 10, y + 10),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.5,
            color=(color_val, 50, 50),
            thickness=2
        )

    return frame


# noinspection PyUnresolvedReferences
def process_video(filename: str, annotations_map: dict[int, dict]) -> None:
    """
    Process a video frame-by-frame with all annotations
    """

    # Open the input video file for reading frames
    input_video_path = utils.VIDEO_PATH / f"{filename}.mp4"
    cap = cv2.VideoCapture(str(input_video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video file {input_video_path}")

    # Open an output video writer to save the annotated frames to a temporary file
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Define the video codec
    # The output video will be color (isColor=True) to allow for color annotations
    out_writer = cv2.VideoWriter(TEMP_OUTPUT_NAME, fourcc, utils.VIDEO_FPS, utils.VIDEO_RES, isColor=True)

    if not out_writer.isOpened():
        cap.release()  # Release the input video capture if output writer fails
        raise ValueError("Could not open video writer.")

    current_video_frame_idx = 0
    while True:
        # Read a single frame from the input video
        ret, frame = cap.read()

        # Break the loop if no more frames are returned (end of video)
        if not ret:
            break

        # Check if there are any annotations that apply to the current video frame
        if current_video_frame_idx in annotations_map.keys():
            frame = annotate_frame(frame, annotations_map[current_video_frame_idx])

        # Write the (potentially) annotated frame to the temporary output video
        out_writer.write(frame)
        current_video_frame_idx += 1

    # Release the video capture and writer objects after processing each video
    cap.release()
    out_writer.release()


def main(input_files: list[str], output_dir: str) -> None:
    # Create folder + subdirs if not existing
    output_dir = Path(output_dir)
    utils.create_output_dir_with_subdirs(output_dir, utils.DATA_SPLITS)

    # Run the pipeline over every file in the input
    for fi in tqdm(input_files, desc="Running pipeline..."):
        # Skip over outputs that already exist
        output_file = output_dir / f"{fi}.mp4"
        if output_file.exists():
            continue

        # Load metadata and format it
        metadata = pd.read_csv(utils.METADATA_PATH / f"{fi}.csv")
        metadata_fmt = format_df(metadata)

        # Prepare annotations for fast lookup by actual video frame index
        # This process builds a dictionary mapping each video frame index to a list of annotations that apply to it.
        frame_to_annotations_map = create_annotations_map(metadata_fmt)

        # Process the video
        process_video(fi, frame_to_annotations_map)

        # Use ffmpeg to combine the temporary annotated video with its original audio track
        utils.combine_audio_and_video(
            video_path=TEMP_OUTPUT_NAME,
            audio_path=utils.AUDIO_PATH / f"{fi}.wav",
            output_path=output_file,
            cleanup=True
        )


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
