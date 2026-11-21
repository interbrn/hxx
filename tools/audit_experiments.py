"""Audit smoke experiment artifacts without starting training.

The audit deliberately treats existing output directories as read-only.  It
checks the common split/configuration contract, records the presence of the
expected artifacts, and reads already-produced test metrics.  Historical runs
may predate ``experiment_manifest.json``; those runs are reported as
partially traceable instead of being silently treated as fully reproducible.
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


# ``metric_dir`` is separate for legacy runs where test_checkpoint.py wrote
# the independent-test result into a sibling directory.  No training is
# inferred from, or started for, these records.
EXPERIMENTS = {
    "rgb": {
        "train_dir": "rgb_grouped_smoke_epoch2",
        "metric_dir": "rgb_grouped_smoke_epoch2",
        "data_config": "configs/rgbt3m_grouped_rgb_smoke.yaml",
    },
    "ir": {
        "train_dir": "ir_grouped_smoke_epoch2",
        "metric_dir": "ir_grouped_smoke_epoch2",
        "data_config": "configs/rgbt3m_grouped_ir_smoke.yaml",
    },
    "ratio1_concat": {
        "train_dir": "ratio1_grouped_smoke_epoch5",
        "metric_dir": "ratio1_grouped_smoke_epoch5",
        "fair_metric_dir": "ratio1_grouped_smoke_epoch2_fair_test",
        "data_config": "configs/rgbt3m_grouped_ratio1_smoke.yaml",
    },
    "ratio2_concat": {
        "train_dir": "ratio2_grouped_smoke_epoch2",
        "metric_dir": "ratio2_grouped_smoke_epoch2",
        "data_config": "configs/rgbt3m_ratio2_grouped_smoke.yaml",
    },
    "ratio2_simple": {
        "train_dir": "ratio2_grouped_smoke_simple_epoch2",
        "metric_dir": "ratio2_grouped_smoke_simple_epoch2_test",
        "data_config": "configs/rgbt3m_ratio2_grouped_smoke.yaml",
    },
    "ratio2_adapter": {
        "train_dir": "ratio2_grouped_smoke_adapter_epoch2",
        "metric_dir": "ratio2_grouped_smoke_adapter_epoch2",
        "data_config": "configs/rgbt3m_ratio2_grouped_smoke.yaml",
    },
    "ratio2_gated": {
        "train_dir": "ratio2_grouped_smoke_gated_epoch5_protocol",
        "metric_dir": "ratio2_grouped_smoke_gated_epoch5_protocol",
        "fair_metric_dir": "ratio2_grouped_smoke_gated_epoch2_test_v6",
        "data_config": "configs/rgbt3m_ratio2_grouped_smoke.yaml",
    },
}

METRIC_KEYS = (
    "metrics/precision(B)",
    "metrics/recall(B)",
    "metrics/mAP50(B)",
    "metrics/mAP50-95(B)",
)
ARTIFACTS = (
    "weights/last.pt",
    "weights/best.pt",
    "results.csv",
    "results.png",
    "test_metrics.json",
    "experiment_manifest.json",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def nonempty_lines(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def video_groups(path: Path) -> set[str]:
    groups = set()
    for line in nonempty_lines(path):
        stem = line.split("/", 1)[-1]
        groups.add(stem.split("_frame_", 1)[0])
    return groups


def metric_summary(directory: Path) -> dict[str, float] | None:
    path = directory / "test_metrics.json"
    if not path.exists():
        return None
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not all(key in raw for key in METRIC_KEYS):
        return None
    return {key: float(raw[key]) for key in METRIC_KEYS}


def audit_experiment(name: str, spec: dict) -> dict:
    train_dir = RUN_ROOT / spec["train_dir"]
    metric_dir = RUN_ROOT / spec["metric_dir"]
    return {
        "name": name,
        "train_dir": spec["train_dir"],
        "metric_dir": spec["metric_dir"],
        "data_config": spec["data_config"],
        "artifacts": {item: (train_dir / item).exists() for item in ARTIFACTS},
        "metrics": metric_summary(metric_dir),
        "fair_metrics": metric_summary(RUN_ROOT / spec["fair_metric_dir"])
        if spec.get("fair_metric_dir")
        else None,
    }


def build_report() -> str:
    split_names = ("train_grouped_smoke.txt", "val_grouped_smoke.txt", "test_grouped_smoke.txt")
    split_info = []
    for name in split_names:
        path = SPLIT_ROOT / name
        split_info.append(
            {
                "name": name,
                "count": len(nonempty_lines(path)),
                "sha256": sha256(path),
            }
        )
    split_groups = {item["name"]: video_groups(SPLIT_ROOT / item["name"]) for item in split_info}
    group_disjoint = not (
        split_groups[split_names[0]] & split_groups[split_names[1]]
        or split_groups[split_names[0]] & split_groups[split_names[2]]
        or split_groups[split_names[1]] & split_groups[split_names[2]]
    )

    rows = [audit_experiment(name, spec) for name, spec in EXPERIMENTS.items()]
    all_configs = {row["data_config"] for row in rows}
    config_text = "\n".join(
        (PROJECT_ROOT / config).read_text(encoding="utf-8") for config in sorted(all_configs)
    )
    common_split_contract = all(
        token in config_text
        for token in (
            "train: splits/train_grouped_smoke.txt",
            "val: splits/val_grouped_smoke.txt",
            "test: splits/test_grouped_smoke.txt",
        )
    )

    lines = [
        f"# 当前 smoke 实验审计（{date.today().isoformat()}）",
        "",
        "本报告由 `tools/audit_experiments.py` 生成，不启动训练，不修改权重。所有数值均为 smoke/短 epoch 流程验证结果，不能作为论文最终结论。",
        "",
        "## 结论",
        "",
        f"- 已审计核心实验：{len(rows)} 个；共同 smoke 划分配置一致：`{'是' if common_split_contract else '否'}`；视频组无交集：`{'是' if group_disjoint else '否'}`。",
        "- 统一数据契约为按视频分组的 `train_grouped_smoke`、`val_grouped_smoke`、`test_grouped_smoke`；独立 test 指标从已有 `test_metrics.json` 读取。",
        "- 历史输出目录均未发现 `experiment_manifest.json`，因此只能依据现有配置、目录命名和指标文件核对，不能把历史运行声明为完全可追溯。后续训练由当前 `train.py` 生成 manifest。",
        "- `ratio1_gated` 与 `hierafire` 是注册的扩展模式，不是任务书 A–G 核心矩阵的缺口；本次不为它们重复训练。",
        "",
        "## 共同 smoke 划分",
        "",
        "| 文件 | 样本数 | SHA-256 |",
        "|---|---:|---|",
    ]
    lines.extend(f"| `{item['name']}` | {item['count']} | `{item['sha256']}` |" for item in split_info)
    lines.extend(
        [
            "",
            "## 实验状态与指标",
            "",
            "`metric_dir` 与 `train_dir` 不同的项目，是历史上将独立 test 结果写到 sibling 目录的情况；这不是重新训练。",
            "",
            "| 实验 | 训练目录产物 | 主要 test 指标目录 | Precision | Recall | mAP50 | mAP50-95 | 公平 test 目录 | manifest |",
            "|---|---|---|---:|---:|---:|---:|---|---|",
        ]
    )
    for row in rows:
        artifacts = row["artifacts"]
        artifact_state = "完整" if all(artifacts.values()) else "缺 " + ", ".join(
            key for key, present in artifacts.items() if not present
        )
        metrics = row["metrics"] or row["fair_metrics"]
        if metrics:
            values = [f"{metrics[key]:.5f}" for key in METRIC_KEYS]
        else:
            values = ["—"] * 4
        fair_dir = "—" if row["fair_metrics"] is None else "sibling test 目录"
        lines.append(
            f"| `{row['name']}` | {artifact_state}（`{row['train_dir']}`） | `{row['metric_dir']}` | "
            + " | ".join(values)
            + f" | {fair_dir} | {'有' if artifacts['experiment_manifest.json'] else '无'} |"
        )

    lines.extend(
        [
            "",
            "## 阶段式消融记录",
            "",
            "在统一 2∶1、2 epoch smoke 协议下，已有结果可作为流程级阶段式消融。若某实验还有 5 epoch 结果，5 epoch 结果只作为单独记录，不与 2 epoch 数值横向比较：",
            "",
            "| 条件 | Precision | Recall | mAP50 | mAP50-95 |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    comparable = ("ratio2_concat", "ratio2_simple", "ratio2_adapter", "ratio2_gated")
    labels = {
        "ratio2_concat": "直接拼接",
        "ratio2_simple": "固定等权",
        "ratio2_adapter": "特征适配",
        "ratio2_gated": "特征适配 + 通道/空间门控",
    }
    for name in comparable:
        row = next(item for item in rows if item["name"] == name)
        metrics = row["fair_metrics"] or row["metrics"]
        values = [f"{metrics[key]:.5f}" if metrics else "—" for key in METRIC_KEYS]
        lines.append(f"| `{labels[name]}`（{name}） | " + " | ".join(values) + " |")
    lines.extend(
        [
            "",
            "上述表格优先使用目录名明确标注的 2 epoch 公平 test 指标；`ratio2_gated` 的 5 epoch 结果仍保留在主要指标目录中，但不混入这张对比表。",
            "",
            "阶段式定义：",
            "",
            "1. `ratio2_concat`：直接拼接，不使用适配器和门控。",
            "2. `ratio2_adapter`：加入多感受野特征适配。",
            "3. `ratio2_gated`：在适配基础上加入通道 + 空间门控。",
            "4. `ratio2_simple`：固定 0.5/0.5 等权融合，用于与可学习融合区分。",
            "",
            "正式论文消融仍应在完整分组数据、100 epoch、同一 seed/增强/优化器和独立 test 上重跑；至少保留 `concat`、`adapter`、`adapter+channel gate`、`adapter+channel+spatial gate` 和 `aux loss on/off` 五个条件。",
            "",
            "## 下一步",
            "",
            "- 服务器先执行所有模式的 `server_full --dry-run`，确认数据根目录、GPU、batch 和输出命名。",
            "- 正式启动前逐个检查 `last.pt`、`best.pt`、`results.csv`、`results.png`、`test_metrics.json` 和 `experiment_manifest.json`；若有 `last.pt` 则续训，有有效 test 指标则直接登记，不覆盖历史结果。",
            "- 本机继续只做模型/数据/测试流程检查，不新增重复 smoke 训练。",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "docs" / "当前smoke实验审计.md",
        help="报告输出路径，默认为 docs/当前smoke实验审计.md",
    )
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else PROJECT_ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(build_report(), encoding="utf-8")
    print(output.resolve())


if __name__ == "__main__":
    main()
