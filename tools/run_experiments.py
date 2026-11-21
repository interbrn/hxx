"""Run one or all registered experiments with a shared profile."""

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = (
    "rgb",
    "ir",
    "ratio1_concat",
    "ratio1_gated",
    "ratio2_concat",
    "ratio2_simple",
    "ratio2_adapter",
    "ratio2_gated",
    "hierafire",
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("local_cpu", "server_full"), required=True)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--experiments", nargs="+", default=["all"],
                        help="all 或一个/多个已注册模式名")
    parser.add_argument("--device", default=None, help="覆盖 profile 的 device，例如 cpu 或 0")
    parser.add_argument("--batch", type=int, default=None, help="覆盖 profile 的 batch")
    parser.add_argument("--workers", type=int, default=None, help="覆盖 profile 的 workers")
    parser.add_argument("--dry-run", action="store_true", help="只打印命令，不启动训练")
    args = parser.parse_args()

    if "all" in args.experiments:
        if len(args.experiments) != 1:
            parser.error("--experiments all 不能和其他模式同时使用")
        selected = list(EXPERIMENTS)
    else:
        unknown = sorted(set(args.experiments) - set(EXPERIMENTS))
        if unknown:
            parser.error(f"未知实验模式: {', '.join(unknown)}")
        selected = args.experiments

    data_root = args.data_root.expanduser().resolve()
    for experiment in selected:
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

        print("[experiment]", experiment)
        print("[command]", " ".join(command))
        if not args.dry_run:
            subprocess.run(command, cwd=PROJECT_ROOT, check=True)


if __name__ == "__main__":
    main()
