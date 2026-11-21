"""快速检查三种 2∶1 模型能否构建并完成一次前向传播。"""

import os
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ["YOLO_CONFIG_DIR"] = str((PROJECT_ROOT / ".yolo_config").resolve())
os.environ["MPLCONFIGDIR"] = str((PROJECT_ROOT / ".matplotlib").resolve())

from ultralytics.nn.tasks import DualStreamDetectionModel
from ultralytics.utils import yaml_load


ROOT = PROJECT_ROOT
CONFIGS = [
    "yolov8n_dual_concat.yaml",
    "yolov8n_dual_concat_ratio2.yaml",
    "yolov8n_dual_simple_ratio2.yaml",
    "yolov8n_dual_adapter_ratio2.yaml",
    "yolov8n_dual_gated_ratio2.yaml",
]


def main():
    # 只用较小的假输入检查结构，不读取数据集，也不进行训练。
    dummy = torch.zeros(1, 6, 64, 96)
    for config_name in CONFIGS:
        config_path = ROOT / "configs" / config_name
        config = yaml_load(str(config_path))
        model = DualStreamDetectionModel(config, ch=6, nc=2, verbose=False)
        model.eval()
        with torch.no_grad():
            output = model(dummy)
        if isinstance(output, (tuple, list)):
            first = output[0]
        else:
            first = output
        shape = tuple(first.shape) if hasattr(first, "shape") else type(first).__name__
        print(f"{config_name}: 输出 {shape}; ratio={model.resolution_ratio}; fusion={model.fusion_type}")


if __name__ == "__main__":
    main()
