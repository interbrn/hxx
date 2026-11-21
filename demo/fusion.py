"""Inspect 1:1 RGB/IR feature sizes and direct fusion outputs."""

import argparse
import os
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ["YOLO_CONFIG_DIR"] = str(PROJECT_ROOT / ".yolo_config")
sys.path.insert(0, str(PROJECT_ROOT))

from ultralytics import YOLO

from pair_io import load_pair, image_tensor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--file-name", required=True)
    parser.add_argument("--model", type=Path, default=PROJECT_ROOT / "configs" / "yolov8n_dual_concat.yaml")
    args = parser.parse_args()

    rgb, ir, _ = load_pair(args.data_root, args.split, args.file_name)
    model = YOLO(str(args.model.resolve() if args.model.is_absolute() else (PROJECT_ROOT / args.model).resolve())).model
    model.eval()
    with torch.inference_mode():
        rgb_features, ir_features = model._forward_both_backbones(image_tensor(rgb), image_tensor(ir))
        for stage in ("p2", "p3", "p4", "p5"):
            fused = model.fusion_convs[stage](rgb_features[stage], ir_features[stage])
            print(stage, "RGB:", tuple(rgb_features[stage].shape),
                  "IR:", tuple(ir_features[stage].shape),
                  "fused:", tuple(fused.shape))


if __name__ == "__main__":
    main()
