# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""
验证入口：使用固定验证集评价训练好的火情检测模型。

本文件不负责重新训练，只负责加载权重、按配置读取 RGB/红外配对数据，
计算 AP50、AP50:95、Recall 和分类别指标，并保存验证结果。
"""
import argparse
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
os.environ["YOLO_CONFIG_DIR"] = str((PROJECT_ROOT / ".yolo_config").resolve())
os.environ["MPLCONFIGDIR"] = str((PROJECT_ROOT / ".matplotlib").resolve())

from ultralytics import YOLO
from ultralytics.utils import yaml_load


def resolve_project_path(value):
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def resolve_split_paths(data, data_root):
    for key in ("train", "val", "test"):
        value = data.get(key)
        if not value:
            continue
        path = Path(str(value)).expanduser()
        if path.is_absolute():
            data[key] = str(path.resolve())
            continue
        project_path = (PROJECT_ROOT / path).resolve()
        dataset_path = (data_root / path).resolve()
        data[key] = str(project_path if project_path.exists() or not dataset_path.exists() else dataset_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--data", default="configs/rgbt3m.yaml")
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--imgsz", type=int, nargs=2, default=[480, 640])
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="0")
    parser.add_argument("--name", default="hierafire_val")
    args = parser.parse_args()
    data_root = resolve_project_path(args.data_root)
    data = yaml_load(str(resolve_project_path(args.data)))
    data["path"] = str(data_root)
    resolve_split_paths(data, data_root)
    data["cache_dir"] = str((PROJECT_ROOT / ".dataset_cache").resolve())
    data["img_size"] = args.imgsz
    data["rgb_size"] = list(args.imgsz)
    ratio = int(data.get("resolution_ratio", 1))
    data["ir_size"] = [max(1, args.imgsz[0] // ratio), max(1, args.imgsz[1] // ratio)]
    model = YOLO(str(resolve_project_path(args.weights)))
    if hasattr(model.model, "resolution_ratio"):
        model.model.resolution_ratio = ratio
        model.model.yaml["resolution_ratio"] = ratio
    model.val(data=data, imgsz=args.imgsz, batch=args.batch,
                          device=args.device, half=False, plots=True,
                          project=str((PROJECT_ROOT / "runs" / "detect").resolve()), name=args.name)


if __name__ == "__main__":
    main()
