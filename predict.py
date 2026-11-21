"""Run detection on one aligned RGB/IR image pair."""

# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""Draw HieraFire predictions on one aligned RGB/thermal pair."""
import argparse
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
os.environ["YOLO_CONFIG_DIR"] = str((PROJECT_ROOT / ".yolo_config").resolve())
os.environ["MPLCONFIGDIR"] = str((PROJECT_ROOT / ".matplotlib").resolve())

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.utils.ops import non_max_suppression


def paired_tensor(rgb, ir, shape, resolution_ratio=1):
    """Build the six-channel tensor used by the dual-stream detector."""
    height, width = shape
    rgb_canvas = cv2.resize(rgb, (width, height), interpolation=cv2.INTER_LINEAR)
    ir_height = max(1, height // resolution_ratio)
    ir_width = max(1, width // resolution_ratio)
    ir_low = cv2.resize(ir, (ir_width, ir_height), interpolation=cv2.INTER_AREA)
    ir_canvas = cv2.resize(ir_low, (width, height), interpolation=cv2.INTER_LINEAR)
    pair = np.concatenate([rgb_canvas, ir_canvas], axis=2)
    # The dataset stores VIS_BGR|IR_BGR and reverses all channels to IR_RGB|VIS_RGB.
    return torch.from_numpy(np.ascontiguousarray(pair[..., ::-1].transpose(2, 0, 1))).float().unsqueeze(0) / 255


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True, help="训练得到的权重文件")
    parser.add_argument("--rgb", required=True, help="RGB 图像路径")
    parser.add_argument("--ir", required=True, help="IR 图像路径")
    parser.add_argument("--imgsz", nargs=2, type=int, default=[480, 640], metavar=("H", "W"))
    parser.add_argument("--ratio", type=int, choices=(1, 2), default=1, help="RGB/IR 分辨率比例")
    parser.add_argument("--device", default="cpu", help="cpu、0 或 cuda:0")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "runs" / "predict")
    args = parser.parse_args()
    rgb, ir = cv2.imread(args.rgb), cv2.imread(args.ir)
    if rgb is None or ir is None:
        raise FileNotFoundError("RGB 和 IR 图像必须能够读取")
    weights_path = Path(args.weights).expanduser()
    if not weights_path.is_absolute():
        weights_path = PROJECT_ROOT / weights_path
    model = YOLO(str(weights_path.resolve())).model
    model.resolution_ratio = args.ratio
    model.to(args.device).float().eval()
    with torch.inference_mode():
        tensor = paired_tensor(rgb, ir, args.imgsz, args.ratio).to(args.device)
        prediction = model(tensor)
        boxes = non_max_suppression(prediction, args.conf, args.iou, nc=len(model.names))[0].cpu()
    output_dir = args.output.expanduser()
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rgb_canvas = cv2.resize(rgb, (args.imgsz[1], args.imgsz[0]), interpolation=cv2.INTER_LINEAR)
    ir_canvas = cv2.resize(ir, (args.imgsz[1], args.imgsz[0]), interpolation=cv2.INTER_LINEAR)
    for modality, canvas in [("rgb", rgb_canvas), ("ir", ir_canvas)]:
        for x1, y1, x2, y2, confidence, cls in boxes.tolist():
            p1 = (round(x1 * canvas.shape[1] / args.imgsz[1]), round(y1 * canvas.shape[0] / args.imgsz[0]))
            p2 = (round(x2 * canvas.shape[1] / args.imgsz[1]), round(y2 * canvas.shape[0] / args.imgsz[0]))
            cv2.rectangle(canvas, p1, p2, (0, 200, 255), 2)
            cv2.putText(canvas, f"{model.names[int(cls)]} {confidence:.2f}", p1,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)
        cv2.imwrite(str(output_dir / f"{modality}.jpg"), canvas)


if __name__ == "__main__":
    main()
