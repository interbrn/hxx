"""用已训练好的 best.pt 在独立 test 集上评估，不重新训练。"""

import argparse
import json
import os
import sys
from pathlib import Path

# 无论从哪个目录启动，都优先使用本科项目自己的 ultralytics 修改版。
# 否则从原始师兄代码目录启动时，Python 可能误加载另一套旧代码。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ["YOLO_CONFIG_DIR"] = str((PROJECT_ROOT / ".yolo_config").resolve())
os.environ["MPLCONFIGDIR"] = str((PROJECT_ROOT / ".matplotlib").resolve())
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

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
    parser.add_argument("--weights", required=True, help="训练得到的 best.pt")
    parser.add_argument("--data", required=True, help="包含 test 字段的数据 YAML")
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--imgsz", type=int, nargs=2, default=[480, 640], metavar=("H", "W"))
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--project", default="runs/detect")
    parser.add_argument("--name", default="checkpoint_test")
    args = parser.parse_args()

    data_root = resolve_project_path(args.data_root)
    data = yaml_load(str(resolve_project_path(args.data)))
    data["path"] = str(data_root)
    resolve_split_paths(data, data_root)
    data["img_size"] = list(args.imgsz)
    data["rgb_size"] = list(args.imgsz)
    ratio = int(data.get("resolution_ratio", 1))
    data["ir_size"] = [max(1, args.imgsz[0] // ratio), max(1, args.imgsz[1] // ratio)]
    # 标签扫描缓存统一放在本科项目根目录，不写入原始数据集，
    # 也避免 train 和 test 各生成一份重复缓存。
    data["cache_dir"] = str((PROJECT_ROOT / ".dataset_cache").resolve())

    test_split = data.get("test")
    if not test_split:
        raise ValueError("数据 YAML 中缺少 test 字段")
    test_path = Path(test_split)
    if not test_path.is_absolute():
        test_path = resolve_project_path(test_path)
    if not test_path.exists():
        raise FileNotFoundError(f"测试集划分文件不存在: {test_path}")

    model = YOLO(str(resolve_project_path(args.weights)))
    if hasattr(model.model, "resolution_ratio"):
        model.model.resolution_ratio = ratio
        model.model.yaml["resolution_ratio"] = ratio

    project_dir = resolve_project_path(args.project)
    run_dir = project_dir / args.name
    metrics = model.val(
        data=data,
        split="test",
        imgsz=list(args.imgsz),
        batch=args.batch,
        device=args.device,
        workers=0,
        half=False,
        plots=True,
        project=str(project_dir),
        name=args.name,
    )
    results = getattr(metrics, "results_dict", {})
    serializable = {key: float(value) for key, value in results.items()}
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "test_metrics.json").write_text(
        json.dumps(serializable, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"独立 test 完成：{run_dir / 'test_metrics.json'}")
    print(json.dumps(serializable, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
