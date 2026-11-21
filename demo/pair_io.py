"""Small reusable helpers for reading one aligned RGB/IR pair."""

from pathlib import Path

import cv2
import numpy as np
import torch


CLASS_NAMES = {0: "fire", 1: "person"}


def pair_paths(data_root, split, file_name):
    root = Path(data_root).expanduser().resolve()
    file_name = Path(file_name).name
    return (
        root / "RGB" / split / file_name,
        root / "IR" / split / file_name,
        root / "labels_fire_person" / split / Path(file_name).with_suffix(".txt").name,
    )


def load_pair(data_root, split, file_name):
    rgb_path, ir_path, label_path = pair_paths(data_root, split, file_name)
    rgb = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
    ir = cv2.imread(str(ir_path), cv2.IMREAD_COLOR)
    if rgb is None:
        raise FileNotFoundError(f"RGB image not found or unreadable: {rgb_path}")
    if ir is None:
        raise FileNotFoundError(f"IR image not found or unreadable: {ir_path}")
    if rgb.shape[:2] != ir.shape[:2]:
        raise ValueError(f"RGB/IR image sizes differ: {rgb.shape[:2]} vs {ir.shape[:2]}")
    return rgb, ir, label_path


def image_tensor(image):
    """Convert an OpenCV BGR image to a normalized [1, 3, H, W] tensor."""
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    array = np.ascontiguousarray(rgb.transpose(2, 0, 1))
    return torch.from_numpy(array).float().div(255.0).unsqueeze(0)


def pair_tensor(rgb, ir, ir_first=True):
    """Convert an aligned pair to a normalized [1, 6, H, W] tensor."""
    tensors = [image_tensor(ir), image_tensor(rgb)] if ir_first else [image_tensor(rgb), image_tensor(ir)]
    return torch.cat(tensors, dim=1)
