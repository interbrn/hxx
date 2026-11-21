"""Validate dataset, configuration and experiment prerequisites without training."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.experiment_registry import (  # noqa: E402
    ALL_EXPERIMENTS,
    CORE_EXPERIMENTS,
    EXPERIMENT_REGISTRY,
)


def resolve_path(value: str | Path) -> Path:
    """Resolve a project-relative path or preserve an absolute path."""
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def resolve_split(value: str, data_root: Path) -> Path:
    """Resolve a split file from the project or dataset root."""
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    project_path = (PROJECT_ROOT / path).resolve()
    dataset_path = (data_root / path).resolve()
    return project_path if project_path.exists() else dataset_path


def sample_image_paths(data_root: Path, split_path: Path, image_ext: str, limit: int) -> list[Path]:
    """Return representative RGB, IR and label paths for a split."""
    rows = [
        line.strip().replace("\\", "/")
        for line in split_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    samples = []
    for row in rows[:limit]:
        row_path = Path(row)
        if row_path.suffix.lower() != image_ext.lower():
            row_path = row_path.with_suffix(image_ext)
        parts = row_path.as_posix().split("/", 1)
        relative = (
            parts[1]
            if parts[0].lower() in {"rgb", "ir", "labels_fire_person", "labels"}
            else row_path.as_posix()
        )
        samples.extend(
            [
                data_root / "RGB" / relative,
                data_root / "IR" / relative,
                data_root / "labels_fire_person" / Path(relative).with_suffix(".txt"),
            ]
        )
    return samples


def selected_experiments(values: list[str]) -> list[str]:
    """Expand core/all aliases and validate experiment names."""
    if values == ["core"]:
        return list(CORE_EXPERIMENTS)
    if values == ["all"]:
        return list(ALL_EXPERIMENTS)
    unknown = sorted(set(values) - set(EXPERIMENT_REGISTRY))
    if unknown:
        raise ValueError(f"未知实验模式: {', '.join(unknown)}")
    return values


def main() -> int:
    """Validate prerequisites and return a process status code."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path, help="RGBT-3M 数据集根目录")
    parser.add_argument("--profile", choices=("local_cpu", "server_full"), default="local_cpu")
    parser.add_argument("--experiments", nargs="+", default=["core"])
    parser.add_argument("--sample-checks", type=int, default=3, help="每个实验检查的样本数量")
    args = parser.parse_args()

    data_root = resolve_path(args.data_root)
    selected = selected_experiments(args.experiments)
    errors: list[str] = []

    if not data_root.is_dir():
        errors.append(f"数据集根目录不存在：{data_root}")
    for required_dir in ("RGB", "IR", "labels_fire_person"):
        path = data_root / required_dir
        if not path.is_dir():
            errors.append(f"数据集目录不存在：{path}")

    for experiment in selected:
        spec = EXPERIMENT_REGISTRY[experiment]
        data_key = "local_data" if args.profile == "local_cpu" else "full_data"
        data_path = resolve_path(spec[data_key])
        model_path = resolve_path(spec["model"])
        if not data_path.is_file():
            errors.append(f"数据配置不存在：{data_path}")
            continue
        if not model_path.is_file():
            errors.append(f"模型配置不存在：{model_path}")
        data = yaml.safe_load(data_path.read_text(encoding="utf-8")) or {}
        names = data.get("names", {})
        names_text = {str(value) for value in names.values()} if isinstance(names, dict) else set(names)
        if names_text != {"fire", "person"}:
            errors.append(f"{experiment} 类别配置不符合要求：{names_text}")
        if tuple(data.get("rgb_size", [])) != (480, 640):
            errors.append(f"{experiment} RGB 尺寸不是 [480, 640]：{data.get('rgb_size')}")
        ratio = int(data.get("resolution_ratio", 1))
        expected_ir = (240, 320) if ratio == 2 else (480, 640)
        if tuple(data.get("ir_size", [])) != expected_ir:
            errors.append(f"{experiment} IR 尺寸与 ratio={ratio} 不一致：{data.get('ir_size')}")
        for split_name in ("train", "val", "test"):
            split_value = data.get(split_name)
            if not split_value:
                errors.append(f"{experiment} 缺少 {split_name} 划分")
                continue
            split_path = resolve_split(str(split_value), data_root)
            if not split_path.is_file():
                errors.append(f"{experiment} 的 {split_name} 划分不存在：{split_path}")
            elif args.sample_checks > 0:
                missing = [
                    path
                    for path in sample_image_paths(
                        data_root, split_path, data.get("image_ext", ".jpg"), args.sample_checks
                    )
                    if not path.exists()
                ]
                if missing:
                    errors.append(f"{experiment} 的 {split_name} 样本文件缺失：{missing[0]}")

    if errors:
        print("项目检查失败：")
        for item in errors:
            print(f"[ERROR] {item}")
        return 1

    print("项目检查通过：")
    print(f"[OK] 数据集根目录：{data_root}")
    print(f"[OK] profile：{args.profile}")
    print(f"[OK] 实验模式：{', '.join(selected)}")
    if args.profile == "local_cpu":
        print("[INFO] local_cpu 使用 micro 划分，仅用于流程验证，不用于论文最终结论。")
    else:
        print("[INFO] server_full 使用完整视频分组划分，正式训练固定 100 epoch 和统一 test 流程。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
