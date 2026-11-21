"""Build the 6-channel tensor used by the dual-stream detector."""

import argparse

from pair_io import load_pair, pair_tensor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--file-name", required=True)
    parser.add_argument("--rgb-first", action="store_true")
    args = parser.parse_args()

    rgb, ir, _ = load_pair(args.data_root, args.split, args.file_name)
    tensor = pair_tensor(rgb, ir, ir_first=not args.rgb_first)
    print(f"Input tensor: {tuple(tensor.shape)}")
    print(f"Value range: [{tensor.min().item():.3f}, {tensor.max().item():.3f}]")


if __name__ == "__main__":
    main()
