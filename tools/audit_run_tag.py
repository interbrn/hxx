"""Audit one reproducible experiment family without starting training.

The audit is intentionally read-only with respect to weights and datasets.
It verifies the common split contract, video-group disjointness, required
artifacts, experiment manifests and independent-test metrics for a run tag.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import date
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = PROJECT_ROOT / "runs" / "detect"
SPLIT_ROOT = PROJECT_ROOT / "splits"
ARTIFACTS = (
    "weights/last.pt",
    "weights/best.pt",
    "results.csv",
    "results.png",
    "test_metrics.json",
    "experiment_manifest.json",
)
METRIC_KEYS = (
    "metrics/precision(B)",
    "metrics/recall(B)",
    "metrics/mAP50(B)",
    "metrics/mAP50-95(B)",
)


def read_lines(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def video_group(item: str) -> str:
    return item.split("/", 1)[-1].split("_frame_", 1)[0]


def metric_values(path: Path) -> dict[str, float] | None:
    if not path.exists():
        return None
    values = json.loads(path.read_text(encoding="utf-8"))
    if not all(key in values for key in METRIC_KEYS):
        return None
    return {key: float(values[key]) for key in METRIC_KEYS}


def format_metric(metrics: dict[str, float] | None, key: str) -> str:
    return "—" if metrics is None else f"{metrics[key]:.5f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-tag", required=True, help="末尾运行标签，例如 rebuild_micro_20261121")
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "docs" / "本机micro-smoke审计.md",
        help="审计报告输出路径",
    )
    args = parser.parse_args()

    run_dirs = sorted(RUN_ROOT.glob(f"*_{args.run_tag}"))
    if not run_dirs:
        raise FileNotFoundError(f"未找到运行标签对应的目录: {args.run_tag}")

    split_names = ("train_grouped_micro.txt", "val_grouped_micro.txt", "test_grouped_micro.txt")
    split_rows = {name: read_lines(SPLIT_ROOT / name) for name in split_names}
    split_groups = {name: {video_group(row) for row in rows} for name, rows in split_rows.items()}
    overlaps = {}
    for index, left in enumerate(split_names):
        for right in split_names[index + 1 :]:
            overlap = split_groups[left] & split_groups[right]
            if overlap:
                overlaps[f"{left} ↔ {right}"] = sorted(overlap)

    records = []
    for run_dir in run_dirs:
        manifest_path = run_dir / "experiment_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else None
        metrics = metric_values(run_dir / "test_metrics.json")
        missing = [item for item in ARTIFACTS if not (run_dir / item).exists()]
        records.append(
            {
                "name": run_dir.name.rsplit(f"_{args.run_tag}", 1)[0],
                "dir": run_dir.name,
                "manifest": manifest,
                "metrics": metrics,
                "missing": missing,
            }
        )

    manifests = [record["manifest"] for record in records if record["manifest"]]
    resolved = [manifest.get("resolved_data", {}) for manifest in manifests]
    training_configs = [manifest.get("training", {}) for manifest in manifests]
    epoch_values = sorted({config.get("epochs") for config in training_configs})
    device_values = sorted({config.get("device") for config in training_configs})
    common_contract = bool(resolved) and len({
        (
            item.get("train"),
            item.get("val"),
            item.get("test"),
            item.get("path"),
        )
        for item in resolved
    }) == 1

    lines = [
        f"# 本机 micro smoke 实验审计（{date.today().isoformat()}）",
        "",
        f"运行标签：`{args.run_tag}`。本报告只审计已有输出，不启动训练、不覆盖权重。micro 数据仅用于 CPU 流程验证，不能作为论文最终精度结论。",
        "",
        "## 审计结论",
        "",
        f"- 发现运行目录：{len(records)} 个；manifest 完整：{len(manifests)} 个。",
        f"- 所有 manifest 的数据根目录和 train/val/test 路径一致：`{'是' if common_contract else '否'}`；分辨率比例按实验设计分别记录。",
        f"- 训练 epoch 配置：`{', '.join(map(str, epoch_values)) or '未记录'}`；设备配置：`{', '.join(map(str, device_values)) or '未记录'}`。",
        f"- train/val/test 视频组无交集：`{'是' if not overlaps else '否'}`。",
        "- 所有实验应使用同一组 `train_grouped_micro.txt`、`val_grouped_micro.txt` 和 `test_grouped_micro.txt`；独立 test 指标来自各自目录的 `test_metrics.json`。",
        "",
        "## 数据划分",
        "",
        "| 文件 | 样本数 | 视频组 | SHA-256 |",
        "|---|---:|---|---|",
    ]
    for name in split_names:
        lines.append(
            f"| `{name}` | {len(split_rows[name])} | {', '.join(sorted(split_groups[name]))} | `{file_sha256(SPLIT_ROOT / name)}` |"
        )
    if overlaps:
        lines.extend(["", "视频组交集：", ""])
        lines.extend(f"- `{key}`：{', '.join(groups)}" for key, groups in overlaps.items())

    lines.extend(
        [
            "",
            "## 运行产物与独立 test 指标",
            "",
            "| 实验 | 必需产物 | Precision | Recall | mAP50 | mAP50-95 | profile | 分辨率比例 |",
            "|---|---|---:|---:|---:|---:|---|---:|",
        ]
    )
    for record in records:
        manifest = record["manifest"] or {}
        data = manifest.get("resolved_data", {})
        training = manifest.get("training", {})
        artifact_state = "完整" if not record["missing"] else "缺少：" + ", ".join(record["missing"])
        metrics = record["metrics"]
        lines.append(
            f"| `{record['name']}` | {artifact_state} | "
            f"{format_metric(metrics, METRIC_KEYS[0])} | {format_metric(metrics, METRIC_KEYS[1])} | "
            f"{format_metric(metrics, METRIC_KEYS[2])} | {format_metric(metrics, METRIC_KEYS[3])} | "
            f"`{manifest.get('profile', '—')}` | `{data.get('resolution_ratio', '—')}` |"
        )
        lines.append(f"  - 参数：`epochs={training.get('epochs')}`、`device={training.get('device')}`、`batch={training.get('batch')}`。")

    lines.extend(
        [
            "",
            "## 解释边界",
            "",
            f"1. 本轮使用 `local_cpu`、epoch 配置为 `{', '.join(map(str, epoch_values)) or '未记录'}`、batch=1 和 micro 划分，目的为验证 RGB/IR 配对、2∶1 尺寸变换、模型前向、训练、验证、独立 test 和结果归档。",
            "2. micro 划分固定为 train 128、val 64、test 64，不能代表完整 RGBT-3M 的样本分布，也不能用于论文最终结论。",
            "3. 服务器正式实验仍使用 `server_full`、完整视频分组、100 epoch 和统一的正式测试流程。",
            "",
        ]
    )
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
