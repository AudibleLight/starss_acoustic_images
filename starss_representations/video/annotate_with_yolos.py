#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Detected bounding boxes with YOLOS
"""

from argparse import ArgumentParser
from copy import deepcopy
from pathlib import Path

import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from transformers import YolosImageProcessor, YolosForObjectDetection
from tqdm import tqdm
from matplotlib.animation import FuncAnimation

from starss_representations import utils

# Four non-overlapping views covering the entire sphere
FOV = 90
OUT_WIDTH, OUT_HEIGHT = 512, 512
VIEWS = [
    (-135, 0),
    (-45, 0),
    (45, 0),
    (135, 0)
]

VIDEO_OUTPUT_DIR = utils.get_project_root() / "outputs/video_dev_annotated_yolos"

MODEL_NAME = "hustvl/yolos-base"
feature_extractor = YolosImageProcessor.from_pretrained(MODEL_NAME)
model = YolosForObjectDetection.from_pretrained(MODEL_NAME)

# Set devices correctly
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

# Keep only bounding boxes that exceed this threshold
thresh = 0.7
desired_labels = [
    "person",
    "cell phone",
    "sink",
]
desired_ids = [model.config.label2id[i] for i in desired_labels]


def equirectangular_to_perspective(
        fov: float,
        theta: float,
        phi: float,
        height_in: float = utils.VIDEO_HEIGHT,
        width_in: float = utils.VIDEO_WIDTH,
        height_out: float = OUT_HEIGHT,
        width_out: float = OUT_WIDTH,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute translation to map equirectangular input to perspective output.

    theta is left/right angle, phi is up/down angle, both given in degrees
    """
    equ_cx = (width_in - 1) / 2.0
    equ_cy = (height_in - 1) / 2.0

    w_fov = fov
    h_fov = float(height_out) / width_out * w_fov

    w_len = np.tan(np.radians(w_fov / 2.0))
    h_len = np.tan(np.radians(h_fov / 2.0))

    x_map = np.ones([height_out, width_out], np.float32)
    y_map = np.tile(np.linspace(-w_len, w_len, width_out), [height_out, 1])
    z_map = -np.tile(np.linspace(-h_len, h_len, height_out), [width_out, 1]).T

    d = np.sqrt(x_map ** 2 + y_map ** 2 + z_map ** 2)
    xyz = np.stack((x_map, y_map, z_map), axis=2) / np.repeat(d[:, :, np.newaxis], 3, axis=2)

    y_axis = np.array([0.0, 1.0, 0.0], np.float32)
    z_axis = np.array([0.0, 0.0, 1.0], np.float32)
    r1, _ = cv2.Rodrigues(z_axis * np.radians(theta))
    r2, _ = cv2.Rodrigues(np.dot(r1, y_axis) * np.radians(-phi))

    xyz = xyz.reshape([height_out * width_out, 3]).T
    xyz = np.dot(r1, xyz)
    xyz = np.dot(r2, xyz).T
    lat = np.arcsin(xyz[:, 2])
    lon = np.arctan2(xyz[:, 1], xyz[:, 0])

    lon = lon.reshape([height_out, width_out]) / np.pi * 180
    lat = -lat.reshape([height_out, width_out]) / np.pi * 180

    lon = lon / 180 * equ_cx + equ_cx
    lat = lat / 90 * equ_cy + equ_cy

    return lon.astype(np.float32), lat.astype(np.float32)


def perspective_bbox_to_equirectangular(bbox: np.ndarray, trans_x: np.ndarray, trans_y: np.ndarray) -> np.ndarray:
    h, w = trans_x.shape
    xmin, ymin, xmax, ymax, cls = bbox

    # Clamp coordinates to valid pixel indices
    x0, y0 = int(np.clip(xmin, 0, w - 1)), int(np.clip(ymin, 0, h - 1))
    x1, y1 = int(np.clip(xmax, 0, w - 1)), int(np.clip(ymax, 0, h - 1))

    # Get the equirectangular coordinates of the bbox corners
    eq_x_min = trans_x[y0, x0]
    eq_y_min = trans_y[y0, x0]
    eq_x_max = trans_x[y1, x1]
    eq_y_max = trans_y[y1, x1]

    # Build the new bbox
    return np.array([eq_x_min, eq_y_min, eq_x_max, eq_y_max, cls])


def compute_bounding_boxes(perspective_images: np.ndarray) -> np.ndarray:
    """
    Computes bounding boxes for a batch of perspective images (square) using YOLOS.
    """

    # add batch dimension if not present initially
    if perspective_images.ndim == 3:
        perspective_images = perspective_images[None, ...]

    b, h, w, c = perspective_images.shape

    # Preprocess: batched pixel_values [B, C, H, W]
    pixel_values = feature_extractor(perspective_images, return_tensors="pt").pixel_values

    # Forward pass: shape (batch, patches, classes)
    with torch.no_grad():
        outs = model(pixel_values.to(device), output_attentions=True)

    # Compute predicted probabilities along class dimension
    probas = outs.logits.softmax(-1)

    # Keep mask per image: (batch, patches)
    keep = probas.max(-1).values > thresh

    # Target sizes for scaling (H,W for each image)
    target_sizes = torch.tensor([[h, w]]).repeat(b, 1).to(device)

    # Postprocess YOLOS outputs: list of dicts
    postprocessed = feature_extractor.post_process(outs, target_sizes)

    results = []

    # need to iterate over all items in the "batch"
    for i in range(b):
        # (num_patches, 4), where 4 == (xmin, ymin, xmax, ymax)
        boxes_scaled = postprocessed[i]["boxes"].cpu()
        # (num_patches,)
        pred_classes = probas[i].argmax(dim=1).cpu()

        # Apply probability threshold filtering
        k = keep[i].cpu()
        boxes_k = boxes_scaled[k]
        classes_k = pred_classes[k]

        # Filter by desired class IDs
        desired = torch.isin(classes_k, torch.tensor(desired_ids))
        boxes_final = boxes_k[desired].cpu().numpy()
        classes_final = classes_k[desired, None].cpu().numpy()

        # Concatenate (xmin, ymin, xmax, ymax, class_id)
        results.append(np.column_stack([boxes_final, classes_final]))

    # return list with length B
    return results


def merge_bboxes(bboxes: np.ndarray, delta_x: float = 0.1, delta_y: float = 0.1) -> np.ndarray:
    """
    Merge multiple bounding boxes into one. Delta values are margins in width/height to merge.
    """

    def is_in_bbox(point: np.ndarray, bbox: np.ndarray) -> bool:
        """
        Returns True if point is inside box, False otherwise
        """
        return bbox[0] <= point[0] <= bbox[2] and bbox[1] <= point[1] <= bbox[3]

    def intersect(bbox: np.ndarray, bbox_: np.ndarray) -> bool:
        """
        Returns True if boxes intersect, false otherwise
        """
        for i_ in range(int(len(bbox) / 2)):
            for j_ in range(int(len(bbox) / 2)):
                # Check if one of the corner of bbox inside bbox_
                if is_in_bbox([bbox[2 * i_], bbox[2 * j_ + 1]], bbox_):
                    return True
        return False


    # Sort bboxes by ymin
    bboxes = sorted(bboxes, key=lambda x: x[1])

    tmp_bbox = None
    while True:
        nb_merge = 0
        used = []
        new_bboxes = []

        # Loop over bboxes
        for i, b in enumerate(bboxes):
            for j, b_ in enumerate(bboxes):

                # If the bbox has already been used just continue
                if i in used or j <= i:
                    continue

                # Compute the bboxes with a margin
                bmargin = [
                    b[0] - (b[2] - b[0]) * delta_x, b[1] - (b[3] - b[1]) * delta_y,
                    b[2] + (b[2] - b[0]) * delta_x, b[3] + (b[3] - b[1]) * delta_y,
                ]
                b_margin = [
                    b_[0] - (b_[2] - b_[0]) * delta_x, b_[1] - (b[3] - b[1]) * delta_y,
                    b_[2] + (b_[2] - b_[0]) * delta_x, b_[3] + (b_[3] - b_[1]) * delta_y,
                ]

                # Merge bboxes if bboxes with margin have an intersection
                #  Check if one of the corner is in the other bbox
                #  We must verify the other side away in case one bounding box is inside the other
                if intersect(bmargin, b_margin) or intersect(b_margin, bmargin):

                    # Also only keep bboxes with the same class label
                    if b[4] == b_[4]:

                        tmp_bbox = [min(b[0], b_[0]), min(b[1], b_[1]), max(b_[2], b[2]), max(b[3], b_[3]), b[4]]
                        used.append(j)
                        nb_merge += 1

                if tmp_bbox:
                    b = tmp_bbox

            if tmp_bbox:
                new_bboxes.append(tmp_bbox)
            elif i not in used:
                new_bboxes.append(b)

            used.append(i)
            tmp_bbox = None

        # If no merge were done, that means all bboxes were already merged
        if nb_merge == 0:
            break

        # Make a copy to avoid modifying original object
        bboxes = deepcopy(new_bboxes)

    return np.array(new_bboxes)


def sanitise_bboxes(bboxes: np.ndarray) -> np.ndarray:
    """
    Sanitise bounding boxes, removing those with invalid values
    """
    bboxes = np.array(bboxes)
    mask_x = (bboxes[:, 3] - bboxes[:, 1]) > 0
    mask_y = (bboxes[:, 4] - bboxes[:, 2]) > 0
    mask_cls = np.isin(bboxes[:, -1].astype(int), np.array(desired_ids))
    mask = np.logical_and.reduce((mask_x, mask_y, mask_cls))
    return bboxes[mask]


def extract_bounding_boxes(
    input_file: str | Path,
    frame_cap: int = None
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract bounding boxes from a video and return raw and sanitized arrays.
    """
    global_bboxes, global_bboxes_sanitised = [], []

    cap = cv2.VideoCapture(str(input_file))
    if not cap.isOpened():
        raise ValueError(f"Could not open video file {input_file}")

    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    trans = [equirectangular_to_perspective(FOV, x, y) for (x, y) in VIEWS]

    for frame_idx in range(n_frames if not frame_cap else frame_cap):
        if frame_idx % 10 == 0:
            print(f"Processing YOLOS, frame {frame_idx} / {n_frames if not frame_cap else frame_cap}...")

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break

        frame_bboxes = []

        all_persp = np.zeros((len(trans), OUT_WIDTH, OUT_HEIGHT, 3))
        for idx, (trans_x, trans_y) in enumerate(trans):
            persp = cv2.remap(frame, trans_x, trans_y, cv2.INTER_CUBIC, borderMode=cv2.BORDER_WRAP)
            all_persp[idx] = persp

        bboxes_persp = compute_bounding_boxes(all_persp)

        for bb, (trans_x, trans_y) in zip(bboxes_persp, trans):
            frame_bboxes.extend([(frame_idx, *perspective_bbox_to_equirectangular(b, trans_x, trans_y)) for b in bb])

        # frame_idx, xmin, ymin, xmax, ymax, class
        #  dimensions are in EQUIRECTANGULAR form now
        valid_bboxes = sanitise_bboxes(frame_bboxes)

        global_bboxes.append(np.array(frame_bboxes))
        global_bboxes_sanitised.append(valid_bboxes)

    cap.release()
    return np.vstack(global_bboxes), np.vstack(global_bboxes_sanitised)


def animate_bounding_boxes(
    input_file: str | Path,
    bboxes_sanitised: np.ndarray,
    output_file: str | Path = None,
    fig: plt.Figure = None,
    ax: plt.Axes = None,
    add_frame: bool = True,
    frame_cap: int = None
) -> FuncAnimation:
    """
    Annotate a video with bounding boxes and create/save an animation.
    """
    cap = cv2.VideoCapture(str(input_file))
    if not cap.isOpened():
        raise ValueError(f"Could not open video file {input_file}")

    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames_to_use = n_frames if frame_cap is None else frame_cap

    if fig is None or ax is None:
        fig, ax = plt.subplots(1, 1)

    def update_video(frame_idx: int):
        ax.clear()
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            return ax

        ax.set(xticks=[], yticks=[], title="YOLOS Bounding Boxes")
        fig.tight_layout()

        if add_frame:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            ax.imshow(rgb)

        # get bounding boxes for current frame idx
        bboxes_at_frame = bboxes_sanitised[np.argwhere(bboxes_sanitised[:, 0] == frame_idx).flatten(), :]

        for (frm, xmin, ymin, xmax, ymax, cls) in bboxes_at_frame:
            assert frm == frame_idx

            rect = mpatches.Rectangle(
                (xmin, ymin), xmax - xmin, ymax - ymin,
                facecolor="none", edgecolor="red", linewidth=2, zorder=10000
            )
            cls_label = model.config.id2label[cls]
            ax.text(xmax + 10, ymax + 10, cls_label, bbox=dict(facecolor='red', zorder=10000))
            ax.add_patch(rect)

        if frame_idx == frames_to_use - 1:
            cap.release()

        return ax

    fa = FuncAnimation(
        fig,
        update_video,
        frames=frames_to_use,
        repeat=False,
        interval=utils.VIDEO_FRAME_TIME * 1000
    )

    if output_file is not None:
        fa.save(output_file)

    return fa


def main(input_files: list[str], output_dir: str):
    # Create folder + subdirs if not existing
    output_dir = Path(output_dir)
    utils.create_output_dir_with_subdirs(output_dir, utils.DATA_SPLITS)

    # Run the pipeline over every file in the input
    for fi in tqdm(input_files, desc="Running pipeline..."):
        # Skip over outputs that already exist
        output_file = output_dir / f"{fi}.mp4"
        # if output_file.exists():
        #     continue

        input_file = utils.VIDEO_PATH / f"{fi}.mp4"
        bbox_orig, bbox_sanit = extract_bounding_boxes(input_file)
        animate_bounding_boxes(input_file, bbox_sanit, output_file)


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
