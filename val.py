# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""Evaluate a trained detector on a configured validation split."""
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
    parser.add_argument("--weights", required=True, help="训练得到的权重文件")
    parser.add_argument(
        "--data",
        default="configs/rgbt3m_ratio2_grouped.yaml",
        help="数据配置文件；默认使用完整 2∶1 分组配置",
    )
    parser.add_argument("--data-root", required=True, type=Path, help="数据集根目录")
    parser.add_argument("--imgsz", type=int, nargs=2, default=[480, 640], metavar=("H", "W"))
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--device", default="cpu", help="cpu、0 或 cuda:0")
    parser.add_argument("--name", default="validation")
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
    from ultralytics.data import build as data_build

    data_build.PIN_MEMORY = str(args.device).lower() not in {"cpu", "-1"}
    model = YOLO(str(resolve_project_path(args.weights)))
    if hasattr(model.model, "resolution_ratio"):
        model.model.resolution_ratio = ratio
        model.model.yaml["resolution_ratio"] = ratio
    model.val(
        data=data,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        half=False,
        plots=True,
        project=str((PROJECT_ROOT / "runs" / "detect").resolve()),
        name=args.name,
    )


if __name__ == "__main__":
    main()
