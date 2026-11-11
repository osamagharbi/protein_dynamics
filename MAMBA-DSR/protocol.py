"""Shared helpers for explicit frame lists and DSR-compatible time scaling.

The paper-comparison manifest lives in ``paper_protocol.py``. The legacy Aβ
helpers remain for the original exploratory held-out-split experiment.
"""

import os


DSR_TRAIN_INTERVAL = 5
DSR_RESOLUTION = 128
PROTOCOL_NAME = "dsr_abeta_every5_train"


def dsr_train_indices(num_frames, interval=DSR_TRAIN_INTERVAL):
    """Return the frame indices used by the existing DSR Aβ baseline."""
    return list(range(0, num_frames, interval))


def dsr_eval_indices(num_frames, train_indices=None):
    """Return all valid non-training frame indices in trajectory order."""
    if train_indices is None:
        train_indices = dsr_train_indices(num_frames)
    train = set(train_indices)
    return [i for i in range(num_frames) if i not in train]


def parse_indices(value):
    """Parse a comma-separated frame-index list."""
    if value is None or value == "":
        return None
    indices = [int(item.strip()) for item in value.split(",") if item.strip()]
    if len(indices) != len(set(indices)):
        raise ValueError("frame indices must be unique")
    return sorted(indices)


def validate_indices(indices, num_frames, name="indices"):
    if not indices:
        raise ValueError(f"{name} must not be empty")
    bad = [i for i in indices if i < 0 or i >= num_frames]
    if bad:
        raise ValueError(
            f"{name} contains out-of-range values {bad}; valid range is [0, {num_frames - 1}]"
        )


def frame_time_from_index(frame_indices, all_times):
    """Select physical times for a list of original trajectory indices."""
    return all_times[frame_indices]


def protocol_times(num_frames, train_indices):
    """Return DSR-compatible normalized times for all original frame indices.

    DSR normalizes time from the first to the last *training* frame, not from
    the last available ground-truth frame.  For Aβ this is frame 0..145, so
    frames 146..148 intentionally receive values slightly above 1.0, exactly
    as in DSR's reconstruction script.
    """
    t_min = min(train_indices)
    t_max = max(train_indices)
    if t_max == t_min:
        raise ValueError("train_indices must span more than one frame for time normalization")
    return [2.0 * (i - t_min) / (t_max - t_min) - 1.0 for i in range(num_frames)]


def ground_truth_path(dsr_dir, gt_case, frame_index):
    return os.path.join(dsr_dir, "datasets", gt_case, f"traj_{frame_index}.obj")


def seed_everything(seed):
    """Seed Python, NumPy, and PyTorch without importing them at module load."""
    import random

    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
