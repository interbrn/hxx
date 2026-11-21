"""Read and inspect one RGB/IR image pair."""

import argparse

from pair_io import load_pair


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--file-name", required=True)
    args = parser.parse_args()

    rgb, ir, label = load_pair(args.data_root, args.split, args.file_name)
    print(f"RGB path: {args.data_root}/RGB/{args.split}/{args.file_name}")
    print(f"IR shape: {ir.shape}; RGB shape: {rgb.shape}")
    print(f"Label path: {label}")


if __name__ == "__main__":
    main()
