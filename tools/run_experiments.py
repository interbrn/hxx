"""Run registered experiments under a shared reproducible profile."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.experiment_registry import ALL_EXPERIMENTS, CORE_EXPERIMENTS, OPTIONAL_EXPERIMENTS

EXPERIMENTS = ALL_EXPERIMENTS

# Historical CPU smoke runs used descriptive names before the profile-based
# launcher was introduced.  Keep them in the preflight map so re-running this
# launcher does not silently duplicate completed work.
LEGACY_LOCAL_RUNS = {
    "rgb": "rgb_grouped_smoke_epoch2",
    "ir": "ir_grouped_smoke_epoch2",
    "ratio1_concat": "ratio1_grouped_smoke_epoch5",
    "ratio2_concat": "ratio2_grouped_smoke_epoch2",
    "ratio2_simple": "ratio2_grouped_smoke_simple_epoch2",
    "ratio2_adapter": "ratio2_grouped_smoke_adapter_epoch2",
    "ratio2_gated": "ratio2_grouped_smoke_gated_epoch5_protocol",
}

LEGACY_LOCAL_METRIC_RUNS = {
    "ratio2_simple": "ratio2_grouped_smoke_simple_epoch2_test",
}

REQUIRED_METRIC_KEYS = {
    "metrics/precision(B)",
    "metrics/recall(B)",
    "metrics/mAP50(B)",
    "metrics/mAP50-95(B)",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("local_cpu", "server_full"), required=True)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument(
        "--experiments",
        nargs="+",
        default=["core"],
        help="core（任务书核心矩阵）、all（核心矩阵及扩展模式）或一个/多个模式名",
    )
    parser.add_argument("--device", default=None, help="覆盖 profile 的 device，例如 cpu 或 0")
    parser.add_argument("--batch", type=int, default=None, help="覆盖 profile 的 batch")
    parser.add_argument("--workers", type=int, default=None, help="覆盖 profile 的 workers")
    parser.add_argument("--epochs", type=int, default=None, help="覆盖 profile 的 epoch 数")
    parser.add_argument("--dry-run", action="store_true", help="只打印命令，不启动训练")
    parser.add_argument(
        "--run-tag",
        default=None,
        help="为本次独立运行追加规范化标签，例如 rebuild_20261121；带标签时不会覆盖旧目录",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="忽略已有权重/测试指标并重新训练；默认会先保护已有实验",
    )
    args = parser.parse_args()

    if "all" in args.experiments:
        if len(args.experiments) != 1:
            parser.error("--experiments all 不能和其他模式同时使用")
        selected = list(EXPERIMENTS)
    elif "core" in args.experiments:
        if len(args.experiments) != 1:
            parser.error("--experiments core 不能和其他模式同时使用")
        selected = list(CORE_EXPERIMENTS)
    else:
        unknown = sorted(set(args.experiments) - set(EXPERIMENTS))
        if unknown:
            parser.error(f"未知实验模式: {', '.join(unknown)}")
        selected = args.experiments

    data_root = args.data_root.expanduser().resolve()
    for experiment in selected:
        default_name = f"{experiment}_local_cpu" if args.profile == "local_cpu" else f"{experiment}_server_full"
        safe_tag = None
        if args.run_tag is not None:
            safe_tag = "_".join(part for part in args.run_tag.strip().split() if part)
            if not safe_tag:
                parser.error("--run-tag 不能为空")
            # A tagged run is isolated from default and historical directories,
            # but a repeated invocation must still protect/resume that exact run.
            candidate_names = [f"{experiment}_{args.profile}_{safe_tag}"]
        else:
            candidate_names = [default_name]
            if args.profile == "local_cpu" and experiment in LEGACY_LOCAL_RUNS:
                candidate_names.append(LEGACY_LOCAL_RUNS[experiment])
        metric_names = list(candidate_names)
        if args.run_tag is None and args.profile == "local_cpu" and experiment in LEGACY_LOCAL_METRIC_RUNS:
            metric_names.append(LEGACY_LOCAL_METRIC_RUNS[experiment])

        candidates = [PROJECT_ROOT / "runs" / "detect" / name for name in candidate_names]
        metric_candidates = [PROJECT_ROOT / "runs" / "detect" / name for name in metric_names]
        valid_metric_dir = None
        resume_path = None
        protected_weight_dir = None
        # A run tag isolates the namespace; it does not disable protection.
        protect_existing = not args.force
        if protect_existing:
            for candidate in metric_candidates:
                metric_path = candidate / "test_metrics.json"
                if metric_path.exists():
                    try:
                        metrics = json.loads(metric_path.read_text(encoding="utf-8"))
                    except (OSError, json.JSONDecodeError):
                        metrics = {}
                    if REQUIRED_METRIC_KEYS.issubset(metrics):
                        valid_metric_dir = metric_path
                        break
            for candidate in candidates:
                if valid_metric_dir is not None:
                    break
                last = candidate / "weights" / "last.pt"
                best = candidate / "weights" / "best.pt"
                if last.exists() and resume_path is None:
                    resume_path = last
                    protected_weight_dir = candidate
                elif best.exists() and protected_weight_dir is None:
                    protected_weight_dir = candidate

        if valid_metric_dir is not None:
            print(f"[skip] {experiment}: 已有有效 test_metrics.json，直接登记 {valid_metric_dir}")
            continue

        command = [
            sys.executable,
            str(PROJECT_ROOT / "train.py"),
            "--profile", args.profile,
            "--experiment", experiment,
            "--data-root", str(data_root),
        ]
        if args.device is not None:
            command.extend(["--device", args.device])
        if args.batch is not None:
            command.extend(["--batch", str(args.batch)])
        if args.workers is not None:
            command.extend(["--workers", str(args.workers)])
        if args.epochs is not None:
            if args.epochs < 1:
                parser.error("--epochs 必须为正整数")
            command.extend(["--epochs", str(args.epochs)])

        if args.run_tag is not None:
            command.extend(["--name", f"{experiment}_{args.profile}_{safe_tag}"])

        if resume_path is not None:
            command.extend(["--resume", str(resume_path)])
            command.extend(["--name", protected_weight_dir.name])
            print(f"[resume] {experiment}: 使用已有 last.pt {resume_path}")
        elif protected_weight_dir is not None:
            print(
                f"[hold] {experiment}: 已有 best.pt 但没有有效 test_metrics.json，"
                f"请先用 tools/test_checkpoint.py 测试 {protected_weight_dir}，不自动重训"
            )

        print("[experiment]", experiment)
        print("[command]", " ".join(command))
        if not args.dry_run and (resume_path is not None or protected_weight_dir is None):
            subprocess.run(command, cwd=PROJECT_ROOT, check=True)


if __name__ == "__main__":
    main()
