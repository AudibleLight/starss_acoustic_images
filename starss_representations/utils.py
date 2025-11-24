import os
import subprocess
from pathlib import Path
from typing import Optional
from importlib import resources

import numpy as np
import cv2


# Parameters for video files: taken from paper
VIDEO_FPS = 29.97
VIDEO_FRAME_TIME = 1 / VIDEO_FPS
VIDEO_RES = (1920, 960)
VIDEO_WIDTH, VIDEO_HEIGHT = VIDEO_RES
VIDEO_RES_RESIZED = (960, 480)

# Color measured in integers from 0 - 255
MIN_COLOR, MAX_COLOR = 0, 255

# Parameters for audio files: taken from paper
AUDIO_SR = 24000

# Parameters for metadata: taken from paper
LABEL_RES = 0.1   # 100 msec per frame
LABEL_COLUMNS = [
    "frame_number",
    "active_class_idx",
    "source_number_idx",
    "azimuth",
    "elevation",
    "distance"
]
# AudioSet-like label mappings
LABEL_MAPPING = {
    "femaleSpeech": 0,
    "maleSpeech": 1,
    "clapping": 2,
    "telephone": 3,
    "laughter": 4,
    "domesticSounds": 5,
    "footsteps": 6,
    "doorCupboard": 7,
    "music": 8,
    "musicInstrument": 9,
    "waterTap": 10,
    "bell": 11,
    "knock": 12,
}
LABEL_MAPPING_INV = {v: k for k, v in LABEL_MAPPING.items()}

DESIRED_FILES = [
    "dev-test-tau/fold4_room8_mix004",
    "dev-train-tau/fold3_room6_mix001",
    "dev-train-tau/fold3_room14_mix003"
]


# Only using first two files from every room for now
# DESIRED_FILES = [
#     'dev-test-sony/fold4_room23_mix002',
#     'dev-test-sony/fold4_room23_mix001',
#     'dev-test-sony/fold4_room24_mix002',
#     'dev-test-sony/fold4_room24_mix001',
#     'dev-train-sony/fold3_room21_mix013',
#     'dev-train-sony/fold3_room21_mix014',
#     'dev-train-sony/fold3_room22_mix002',
#     'dev-train-sony/fold3_room22_mix001',
#     'dev-test-tau/fold4_room15_mix001',
#     'dev-test-tau/fold4_room15_mix002',
#     'dev-test-tau/fold4_room16_mix001',
#     'dev-test-tau/fold4_room16_mix002',
#     'dev-test-tau/fold4_room10_mix001',
#     'dev-test-tau/fold4_room10_mix002',
#     'dev-test-tau/fold4_room2_mix001',
#     'dev-test-tau/fold4_room2_mix002',
#     'dev-test-tau/fold4_room8_mix001',
#     'dev-test-tau/fold4_room8_mix002',
#     'dev-train-tau/fold3_room12_mix001',
#     'dev-train-tau/fold3_room12_mix002',
#     'dev-train-tau/fold3_room13_mix001',
#     'dev-train-tau/fold3_room13_mix002',
#     'dev-train-tau/fold3_room14_mix001',
#     'dev-train-tau/fold3_room14_mix002',
#     'dev-train-tau/fold3_room4_mix001',
#     'dev-train-tau/fold3_room4_mix004',
#     'dev-train-tau/fold3_room6_mix001',
#     'dev-train-tau/fold3_room6_mix002',
#     'dev-train-tau/fold3_room7_mix001',
#     'dev-train-tau/fold3_room7_mix002',
#     'dev-train-tau/fold3_room9_mix001',
#     'dev-train-tau/fold3_room9_mix002',
# ]
DATA_SPLITS = [
    "dev-train-tau",
    "dev-test-tau",
    "dev-train-sony",
    "dev-test-sony",
]


# noinspection PyUnresolvedReferences
def get_project_root() -> Path:  # pragma: no cover
    """Returns the root directory of the project."""
    return resources.files("starss_representations").parent


STARSS_ROOT = get_project_root() / "data"
VIDEO_PATH = STARSS_ROOT / "video_dev"
AUDIO_PATH = STARSS_ROOT / "foa_dev"
METADATA_PATH = STARSS_ROOT / "metadata_dev"
EIGEN_PATH = STARSS_ROOT / "eigen_dev"


def load_video(
        video_path: str | Path,
        grayscale: Optional[bool] = False,
        resize: Optional[bool] = False
) -> np.ndarray:
    """
    Loads a full video as a numpy array with OpenCV

    Resizing or converting to grayscale are optional.
    """
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Get correct resizing parameters
    if resize:
        width, height = VIDEO_RES_RESIZED
    else:
        width, height = VIDEO_RES

    # Create buffer to store frames
    #  Grayscale buffer has no channel dimension
    if grayscale:
        buf = np.empty((frame_count, height, width,), np.dtype('uint8'))
    else:
        buf = np.empty((frame_count, height, width, 3), np.dtype('uint8'))

    fc = 0
    ret = True

    # Keep reading frames until we run out
    while fc < frame_count and ret:
        ret, img = cap.read()

        # Apply transforms as required
        if grayscale:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if resize:
            img = cv2.resize(img, (width, height))

        # Update buffer and counter
        buf[fc] = img
        fc += 1

    cap.release()
    return buf


# noinspection PyUnresolvedReferences
def write_video(annotated_video: np.ndarray, outpath: str | Path, fps: float = VIDEO_FPS) -> None:
    """
    Writes annotated video to outpath.
    """
    # Get parameters from video file
    width = annotated_video.shape[2]
    height = annotated_video.shape[1]
    # Color video has four dimensions
    is_color = annotated_video.ndim == 4

    # Create output writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_writer = cv2.VideoWriter(outpath, fourcc, fps, (width, height), isColor=is_color)

    if not out_writer.isOpened():
        raise ValueError("Could not open video writer")

    # Write all frames sequentially
    for i in range(annotated_video.shape[0]):
        out_writer.write(annotated_video[i])
    out_writer.release()


def spherical_to_pixel(azimuth: float, elevation: float, width: float, height: float) -> np.ndarray:
    """
    Converts spherical coordinates (azimuth, elevation) to pixel coordinates (x, y) for an equirectangular video frame.
    """
    # Azimuth to X conversion
    x = int(width / 2 - (azimuth * width / 360))

    # Ensure x stays within bounds (0 to width-1)
    x = max(0, min(width - 1, x))

    # Elevation to Y conversion
    y = int((90 - elevation) * height / 180)

    # Ensure y stays within bounds (0 to height-1)
    y = max(0, min(height - 1, y))

    return x, y


def interp2d(array: np.ndarray, n_out: int) -> np.ndarray:
    # Build input and output time axes
    n_in = len(array)
    t_in = np.arange(n_in) * 0.1
    duration = t_in[-1]
    t_out = np.linspace(0, duration, n_out)

    # Split into individual 1D arrays, interpolate all of them
    ins = [np.interp(t_out, t_in, array[:, i]) for i in range(array.shape[1])]

    # Stack back to 2D
    return np.column_stack(ins)


def combine_audio_and_video(video_path: str, audio_path: str, output_path: str, cleanup: bool = True) -> None:
    """
    Use ffmpeg to combine the temporary annotated video with its original audio track
    """
    # Define ffmpeg command and run
    ffmpeg_command = f"ffmpeg -i {video_path} -i {audio_path} -c:v copy -c:a aac {output_path} -y"
    out = subprocess.run(
        ffmpeg_command,
        shell=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )

    # Raise on non-zero error code
    if out.returncode != 0:
        raise ValueError(f"Returned non-zero exit code from FFmpeg with command {ffmpeg_command}")

    # If cleaning up, remove original (temporary) video path
    if cleanup:
        os.remove(video_path)


def create_output_dir_with_subdirs(output_dir: str | Path, subdirs: list[str] = None) -> None:
    """
    Sanitise output directory and create if not existing
    """
    if subdirs is None:
        subdirs = DATA_SPLITS

    output_dir = Path(output_dir)
    if not output_dir.exists():
        output_dir.mkdir()
    for split in subdirs:
        if not (output_dir / split).exists():
            (output_dir / split).mkdir()