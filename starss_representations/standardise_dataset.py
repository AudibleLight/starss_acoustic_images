#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Standardise acoustic maps saved as JSON files.

The standardisation process is as follows:
- We start by grabbing the mean and standard deviation values of maximum pixel amplitudes in the STARSS23 training set
    - Note that these values are HARDCODED as they have been found from empirical testing with this data
- Each dictionary in the input consists of 1/2 polygons for a single object in a single frame
    - Usually, we'd expect to just have one polygon if the object is in the middle of the frame, but if it
        crosses to the edge of the frame, it will wrap around, so we'll end up with two polygons.
- For each polygon, take the amplitude values, subtract STARSS_MU, and divide by STARSS_SD (i.e., z-scoring)
    - Next, apply the sigmoid function to the amplitude values, to scale them within [0., 1.]

This file should be run after `generate_acoustic_image_dataset.py`
"""

import json
from argparse import ArgumentParser
from decimal import Decimal
from pathlib import Path

import numpy as np
import ijson

from starss_representations import utils


# These values are hardcoded and should not be changed
#  The represent the mean/std of the max pixel amplitude
#  for every polygon mask in the STARSS training data
STARSS_MU, STARSS_SIGMA = 0.0006106861602095730349659024345, 0.0004970147144300498711058668102
DEFAULT_INPUT_DIRECTORY = utils.get_project_root() / "outputs/apgd_dev"
DEFAULT_OUTPUT_DIRECTORY = utils.get_project_root() / "outputs/apgd_dev_std"


def sigmoid(x: np.ndarray) -> np.ndarray:
    """
    Applies sigmoid function to an array or scalar input.

    Output is mapped within the range [0, 1.]
    """
    return np.exp(-np.logaddexp(0, -x))


def convert_decimals(obj):
    """
    Recursively convert Decimal to float in nested structures.
    """
    if isinstance(obj, Decimal):
        return float(obj)

    elif isinstance(obj, list):
        return [convert_decimals(x) for x in obj]

    elif isinstance(obj, dict):
        return {k: convert_decimals(v) for k, v in obj.items()}

    else:
        return obj


def do_standard(js, output_dir: Path):
    # create output path
    new_js_path = output_dir / (str(js.stem) + "_std.json")
    new_js_path.parent.mkdir(parents=True, exist_ok=True)

    if new_js_path.exists():
        print(f"Skipping file {new_js_path}, exists!")
        return

    with open(js, "rb") as fin, open(new_js_path, "w", encoding="utf-8") as fout:

        # Start JSON
        fout.write('{\n')
        fout.write('  "annotations": [\n')

        parser = ijson.items(fin, "annotations.item")

        first = True

        for ann in parser:

            new_segs = []

            for seg in ann["segmentation"]:
                seg_arr = np.array(seg, dtype=float)
                seg_amp = seg_arr[:, -1]

                seg_amp_z = (seg_amp - STARSS_MU) / STARSS_SIGMA
                seg_amp_sig = sigmoid(seg_amp_z)

                assert np.logical_and(
                    seg_amp_sig >= 0.0,
                    seg_amp_sig <= 1.0
                ).all(), "val mismatch"

                new_seg = seg_arr.copy()
                new_seg[:, -1] = seg_amp_sig

                new_segs.append(new_seg.tolist())

            ann["segmentation"] = new_segs

            # Write comma between items
            if not first:
                fout.write(",\n")
            first = False

            # Dump single annotation
            ann_clean = convert_decimals(ann)
            json.dump(ann_clean, fout, ensure_ascii=False)

        # Close JSON
        fout.write("\n  ]\n")
        fout.write("}\n")


def main(input_dir: str, output_dir: str):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    utils.create_output_dir_with_subdirs(output_dir, subdirs=utils.DATA_SPLITS)

    if not input_dir.exists():
        input_dir.mkdir(parents=True)
    if not output_dir.exists():
        output_dir.mkdir(parents=True)

    js_files = Path(input_dir).rglob("**/*.json")

    # Dynamic parallel run iteratively reduces workers on crash
    args_list = [(inp, output_dir) for inp in js_files]
    utils.dynamic_parallel_run(do_standard, args_list,)


if __name__ == "__main__":
    # Set up argument parser
    parse_ = ArgumentParser(description=__doc__)
    parse_.add_argument(
        "--input-dir",
        type=str,
        help="Path to the source data directory",
        default=DEFAULT_INPUT_DIRECTORY
    )
    parse_.add_argument(
        "--output-dir",
        type=str,
        help="Path to the output (standardised) data directory",
        default=DEFAULT_OUTPUT_DIRECTORY
    )
    # Parse arguments
    args = vars(parse_.parse_args())

    main(**args)
