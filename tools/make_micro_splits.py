"""Create deterministic, label-aware and video-grouped CPU smoke splits.

The source files are already partitioned by video. This utility selects a
small, deterministic mixture of labelled and background frames inside each
partition, so validation and test do not accidentally contain only empty
labels. The generated files validate the code path only; they are never a
substitute for the complete paper protocol.
"""

from __future__ import annotations

import argparse
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_COUNTS = {"train": 128, "val": 64, "test": 64}


def read_nonempty(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def video_group(item: str) -> str:
    return item.split("/", 1)[-1].split("_frame_", 1)[0]


def has_label(item: str, data_root: Path, label_dir: str) -> bool:
    subset, stem = item.split("/", 1)
    label_path = data_root / label_dir / subset / f"{stem}.txt"
    if not label_path.exists():
        return False
    return bool(label_path.read_text(encoding="utf-8").strip())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=PROJECT_ROOT / "splits")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "splits")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--label-dir", default="labels_fire_person")
    parser.add_argument("--positive-fraction", type=float, default=0.75)
    parser.add_argument("--train", type=int, default=DEFAULT_COUNTS["train"])
    parser.add_argument("--val", type=int, default=DEFAULT_COUNTS["val"])
    parser.add_argument("--test", type=int, default=DEFAULT_COUNTS["test"])
    args = parser.parse_args()
    source_dir = args.source_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    data_root = args.data_root.expanduser().resolve()
    counts = {"train": args.train, "val": args.val, "test": args.test}
    if any(value < 1 for value in counts.values()):
        raise ValueError("micro split counts must all be positive")
    if not 0.0 < args.positive_fraction <= 1.0:
        raise ValueError("positive-fraction must be in (0, 1]")

    selected_groups: dict[str, set[str]] = {}
    for split, count in counts.items():
        source = source_dir / f"{split}_grouped_smoke.txt"
        rows = read_nonempty(source)
        if len(rows) < count:
            raise ValueError(f"{source} has only {len(rows)} rows, cannot select {count}")
        positives = [row for row in rows if has_label(row, data_root, args.label_dir)]
        backgrounds = [row for row in rows if row not in set(positives)]
        positive_count = min(len(positives), max(1, round(count * args.positive_fraction)))
        background_count = count - positive_count
        if len(backgrounds) < background_count:
            positive_count = count - len(backgrounds)
        if len(positives) < positive_count or len(backgrounds) < count - positive_count:
            raise ValueError(
                f"{source} cannot provide {count} samples with labels/backgrounds; "
                f"positive={len(positives)}, background={len(backgrounds)}"
            )
        selected = positives[:positive_count] + backgrounds[: count - positive_count]
        selected_groups[split] = {video_group(row) for row in selected}
        target = output_dir / f"{split}_grouped_micro.txt"
        target.write_text("\n".join(selected) + "\n", encoding="utf-8")
        print(f"{target.name}: {len(selected)} samples; videos={sorted(selected_groups[split])}")

    splits = tuple(selected_groups)
    for index, left in enumerate(splits):
        for right in splits[index + 1 :]:
            overlap = selected_groups[left] & selected_groups[right]
            if overlap:
                raise RuntimeError(f"video leakage between {left} and {right}: {sorted(overlap)}")


if __name__ == "__main__":
    main()
