"""Save a side-by-side visualization of one aligned RGB/IR pair."""

import argparse
from pathlib import Path

import cv2

from pair_io import load_pair


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--file-name", required=True)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "runs" / "demo" / "rgb_ir_pair.jpg")
    args = parser.parse_args()

    rgb, ir, _ = load_pair(args.data_root, args.split, args.file_name)
    output = args.output if args.output.is_absolute() else PROJECT_ROOT / args.output
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), cv2.hconcat([rgb, ir])):
        raise OSError(f"Failed to write visualization: {output}")
    print(f"Saved pair visualization to: {output}")


if __name__ == "__main__":
    main()
