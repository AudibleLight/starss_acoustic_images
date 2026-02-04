import os
import subprocess
from pathlib import Path
from typing import Optional
from importlib import resources

import numpy as np
import cv2
from joblib import Parallel, delayed
from joblib.externals.loky.process_executor import TerminatedWorkerError


# Parameters for video files: taken from paper
VIDEO_FPS = 29.97
VIDEO_FRAME_TIME = 1 / VIDEO_FPS
VIDEO_RES = (1920, 960)
VIDEO_WIDTH, VIDEO_HEIGHT = VIDEO_RES
VIDEO_RES_RESIZED = (960, 480)
N_JOBS = -1

# Parameters for matplotlib figures
DEFAULT_FIG_WIDTH, DEFAULT_FIG_HEIGHT = 10, 5

# Color measured in integers from 0 to 255
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


def dynamic_parallel_run(func, args_list: list[tuple] = None, kwargs_list: list[dict] = None, n_jobs: int = N_JOBS):
    """
    Run func over a list of argument tuples in parallel, dynamically reducing workers
    if a TerminatedWorkerError occurs.

    Parameters:
        func : callable
            The function to run.
        args_list : list of tuples
            Each tuple contains the positional arguments for a single call to func.
        kwargs_list : list of dicts, optional
            Each dict contains keyword arguments for the corresponding call.
        n_jobs : int
            Number of parallel jobs; -1 means use all CPU cores.

    Returns:
        List of results.
    """
    if args_list is None:
        args_list = []

    if kwargs_list is None:
        kwargs_list = [{} for _ in args_list]

    if n_jobs == -1:
        n_jobs = os.cpu_count() or 1

    current_jobs = n_jobs

    while current_jobs > 1:
        try:
            print(f"Trying with n_jobs={current_jobs}...")
            results = Parallel(n_jobs=current_jobs)(
                delayed(func)(*args_, **kwargs_) for args_, kwargs_ in zip(args_list, kwargs_list)
            )
            return results
        except TerminatedWorkerError:
            print(f"Workers terminated at n_jobs={current_jobs}. Reducing workers...")
            if current_jobs == 1:
                print("Already at 1 job. Running serially...")
            current_jobs = max(1, current_jobs // 2)

    # Fallback: serial execution if all else fails
    print("Falling back to serial execution...")
    return [func(*args_, **kwargs_) for args_, kwargs_ in zip(args_list, kwargs_list)]
