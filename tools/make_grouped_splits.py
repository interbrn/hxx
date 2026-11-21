"""按视频生成 RGBT-3M 的 train/val/test 划分，并生成 1/5 调试划分。

每行保存为 ``原始目录/文件名``，例如 ``val/video5_frame_00100``，这样
数据加载器能保留原始 RGB/IR/标签目录关系，同时避免同一视频跨集合。
"""

from pathlib import Path
import argparse


PROJECT_ROOT = Path(__file__).resolve().parents[1]

# 这些视频组不会在 train/val/test 之间交叉。
GROUPS = {
    "train": {"video1", "video2", "video3", "video5"},
    "val": {"video4", "video9"},
    "test": {"video8", "video10"},
}


def read_source_files(root):
    rows = []
    for subset in ("train", "val"):
        source = root / f"{subset}.txt"
        for line in source.read_text(encoding="utf-8").splitlines():
            stem = line.strip()
            if stem:
                rows.append((subset, stem))
    return rows


def video_name(stem):
    return stem.split("_frame_", 1)[0]


def write_list(output_dir, name, rows):
    path = output_dir / name
    path.write_text("\n".join(f"{subset}/{stem}" for subset, stem in rows) + "\n", encoding="utf-8")
    print(f"{name}: {len(rows)} samples")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path,
                        help="RGBT-3M root containing train.txt and val.txt")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "splits")
    args = parser.parse_args()
    root = args.data_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser()
    output_dir = output_dir if output_dir.is_absolute() else PROJECT_ROOT / output_dir
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    grouped = {key: [] for key in GROUPS}
    unknown = []
    for subset, stem in read_source_files(root):
        group = next((key for key, videos in GROUPS.items() if video_name(stem) in videos), None)
        if group is None:
            unknown.append(stem)
        else:
            grouped[group].append((subset, stem))
    if unknown:
        raise RuntimeError(f"未分配视频样本: {unknown[:5]}")

    for key, rows in grouped.items():
        write_list(output_dir, f"{key}_grouped.txt", rows)
        # 调试时每个集合取确定性的前 1/5，仍然保持视频分组不交叉。
        small = rows[: max(1, len(rows) // 5)]
        write_list(output_dir, f"{key}_grouped_smoke.txt", small)


if __name__ == "__main__":
    main()
