# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""
模型复杂度统计入口。

本文件统计本毕设模型的参数量和 FLOPs，并支持比较 1:1 与 2:1 红外分辨率。
它不训练模型，也不改变模型权重；用途是生成论文中的效率对比数据。
"""

from __future__ import annotations

import argparse
import os
import time
from collections import Counter
from copy import deepcopy
from pathlib import Path

import torch

# Keep Ultralytics settings, persistent caches and Matplotlib state inside the
# D: project workspace, matching train.py/val.py and the project's storage
# policy.  This must be set before importing YOLO.
PROJECT_ROOT = Path(__file__).resolve().parent
os.environ.setdefault("YOLO_CONFIG_DIR", str((PROJECT_ROOT / ".yolo_config").resolve()))
os.environ.setdefault("MPLCONFIGDIR", str((PROJECT_ROOT / ".matplotlib").resolve()))

from ultralytics import YOLO
from ultralytics.nn.tasks import DualStreamDetectionModel


def load_profile_model(source: Path) -> torch.nn.Module:
    """Load YAML or checkpoint and rebuild serialized dual-stream models for compatibility."""
    loaded = YOLO(str(source)).model
    yaml = getattr(loaded, "yaml", {})
    if source.suffix.lower() == ".pt" and yaml.get("dual_stream"):
        model = DualStreamDetectionModel(deepcopy(yaml), ch=6, nc=yaml.get("nc"), verbose=False)
        model.load_state_dict(loaded.state_dict(), strict=True)
        return model
    return loaded


def profile_flops(
    model: torch.nn.Module,
    input_shape: tuple[int, int, int, int],
    device: torch.device,
) -> tuple[int, Counter]:
    """Return profiler-supported FLOPs and an operator-level breakdown."""
    activities = [torch.profiler.ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    model = model.to(device=device, dtype=torch.float32).eval()
    sample = torch.zeros(input_shape, device=device, dtype=torch.float32)
    with torch.inference_mode(), torch.profiler.profile(activities=activities, with_flops=True) as profiler:
        model(sample)
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    breakdown = Counter()
    for event in profiler.key_averages():
        flops = int(event.flops or 0)
        if flops:
            breakdown[event.key] += flops
    return sum(breakdown.values()), breakdown


def profile_latency(
    model: torch.nn.Module,
    input_shape: tuple[int, int, int, int],
    device: torch.device,
    warmup: int = 2,
    iterations: int = 10,
) -> tuple[float, float, float]:
    """Measure network-only forward latency in milliseconds.

    The measurement excludes image decoding, resizing, label handling and NMS.
    Those pipeline costs must be measured separately in the final server
    experiment; keeping this definition explicit prevents mixing unlike costs.
    """
    if warmup < 0 or iterations < 1:
        raise ValueError("warmup must be >= 0 and iterations must be >= 1")
    model = model.to(device=device, dtype=torch.float32).eval()
    sample = torch.zeros(input_shape, device=device, dtype=torch.float32)
    with torch.inference_mode():
        for _ in range(warmup):
            model(sample)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = []
        for _ in range(iterations):
            start = time.perf_counter()
            model(sample)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elapsed.append((time.perf_counter() - start) * 1000.0)
    values = torch.tensor(elapsed, dtype=torch.float64)
    return float(values.mean()), float(values.median()), float(torch.quantile(values, 0.95))


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("configs/hierafire.yaml"))
    parser.add_argument("--imgsz", type=int, nargs=2, default=(480, 640), metavar=("H", "W"))
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--device", default="cpu", help="cpu, cuda, or cuda:0")
    parser.add_argument("--resolution-ratio", type=int, choices=(1, 2), default=None)
    parser.add_argument("--topk", type=int, default=6, help="number of operator types to print")
    parser.add_argument("--latency-warmup", type=int, default=2)
    parser.add_argument("--latency-iters", type=int, default=10)
    parser.set_defaults(fuse=True)
    fuse_group = parser.add_mutually_exclusive_group()
    fuse_group.add_argument(
        "--fuse",
        dest="fuse",
        action="store_true",
        help="在统计前融合 Conv-BN；默认开启，适合部署/验证口径",
    )
    fuse_group.add_argument(
        "--no-fuse",
        dest="fuse",
        action="store_false",
        help="保留未融合模型，用于与训练初始化结构对照",
    )
    return parser.parse_args()


def main() -> None:
    """Load a model, run the profiler, and print a reproducible complexity report."""
    args = parse_args()
    if not args.model.exists():
        raise FileNotFoundError(f"Model configuration or checkpoint not found: {args.model}")
    if args.batch < 1:
        raise ValueError(f"Batch size must be positive, got {args.batch}")

    device = torch.device(args.device)
    model = load_profile_model(args.model)
    if args.fuse and hasattr(model, "fuse"):
        model = model.fuse()
    if args.resolution_ratio is not None and hasattr(model, "resolution_ratio"):
        model.resolution_ratio = args.resolution_ratio
        model.yaml["resolution_ratio"] = args.resolution_ratio
    channels = int(getattr(model, "ch", 3))
    if args.model.suffix.lower() in {".yaml", ".yml"} and not isinstance(model, DualStreamDetectionModel):
        raise RuntimeError("A dual-stream YAML must construct DualStreamDetectionModel, but it did not.")

    input_shape = (args.batch, channels, args.imgsz[0], args.imgsz[1])
    total_flops, breakdown = profile_flops(model, input_shape, device)
    parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)

    print(f"Model: {args.model}")
    print(f"Model class: {type(model).__name__}")
    print(f"Input: {input_shape} (FP32)")
    print(f"Parameters: {parameters:,} ({parameters / 1e6:.4f} M)")
    print(f"Trainable parameters: {trainable:,} ({trainable / 1e6:.4f} M)")
    print(f"FLOPs: {total_flops:,} ({total_flops / 1e9:.4f} GFLOPs)")
    print(f"Fusion state: {'fused (deployment/validation)' if args.fuse else 'unfused (training structure)'}")
    mean_ms, median_ms, p95_ms = profile_latency(
        model,
        input_shape,
        device,
        warmup=args.latency_warmup,
        iterations=args.latency_iters,
    )
    print(
        f"Network forward latency: mean={mean_ms:.3f} ms, "
        f"median={median_ms:.3f} ms, p95={p95_ms:.3f} ms "
        f"(warmup={args.latency_warmup}, iterations={args.latency_iters})"
    )
    print("Convention: one multiplication and one addition count as two FLOPs.")
    print("Operator coverage follows PyTorch profiler's with_flops support.")
    if args.topk > 0:
        print("Top operators:")
        for operator, flops in breakdown.most_common(args.topk):
            print(f"  {operator}: {flops / 1e9:.6f} GFLOPs")


if __name__ == "__main__":
    main()
