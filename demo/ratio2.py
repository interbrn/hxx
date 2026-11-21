"""Inspect 2:1 IR resizing and feature alignment before fusion."""

import argparse
import os
import sys
from pathlib import Path

import cv2
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ["YOLO_CONFIG_DIR"] = str(PROJECT_ROOT / ".yolo_config")
sys.path.insert(0, str(PROJECT_ROOT))

from ultralytics import YOLO

from pair_io import image_tensor, load_pair


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--file-name", required=True)
    parser.add_argument("--model", type=Path, default=PROJECT_ROOT / "configs" / "yolov8n_dual_concat_ratio2.yaml")
    args = parser.parse_args()

    rgb, ir, _ = load_pair(args.data_root, args.split, args.file_name)
    ir_small = cv2.resize(ir, (rgb.shape[1] // 2, rgb.shape[0] // 2), interpolation=cv2.INTER_AREA)
    model_path = args.model if args.model.is_absolute() else PROJECT_ROOT / args.model
    model = YOLO(str(model_path.resolve())).model
    model.eval()
    with torch.inference_mode():
        rgb_features, ir_features = model._forward_both_backbones(image_tensor(rgb), image_tensor(ir_small))
        for stage in ("p2", "p3", "p4", "p5"):
            rgb_feature, ir_feature = rgb_features[stage], ir_features[stage]
            ir_aligned = F.interpolate(ir_feature, size=rgb_feature.shape[-2:], mode="bilinear", align_corners=False)
            fused = model.fusion_convs[stage](rgb_feature, ir_aligned)
            print(stage, "RGB:", tuple(rgb_feature.shape), "IR:", tuple(ir_feature.shape),
                  "aligned IR:", tuple(ir_aligned.shape), "fused:", tuple(fused.shape))


if __name__ == "__main__":
    main()
