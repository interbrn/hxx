"""Run one complete 1:1 dual-stream forward pass."""

import argparse
import os
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ["YOLO_CONFIG_DIR"] = str(PROJECT_ROOT / ".yolo_config")
sys.path.insert(0, str(PROJECT_ROOT))

from ultralytics import YOLO

from pair_io import load_pair, pair_tensor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--file-name", required=True)
    parser.add_argument("--model", type=Path, default=PROJECT_ROOT / "configs" / "yolov8n_dual_concat.yaml")
    args = parser.parse_args()

    rgb, ir, _ = load_pair(args.data_root, args.split, args.file_name)
    model_path = args.model if args.model.is_absolute() else PROJECT_ROOT / args.model
    model = YOLO(str(model_path.resolve())).model.eval()
    inputs = pair_tensor(rgb, ir)
    with torch.inference_mode():
        output = model(inputs)
    prediction = output[0] if isinstance(output, (tuple, list)) else output
    print("Input:", tuple(inputs.shape))
    print("Prediction:", tuple(prediction.shape) if hasattr(prediction, "shape") else type(prediction).__name__)


if __name__ == "__main__":
    main()
