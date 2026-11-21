"""Draw YOLO fire/person labels on one RGB/IR image pair."""

import argparse
from pathlib import Path

import cv2

from pair_io import CLASS_NAMES, load_pair


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def draw_labels(image, label_lines):
    height, width = image.shape[:2]
    for line in label_lines:
        values = line.split()
        if len(values) != 5:
            continue
        class_id, x_center, y_center, box_width, box_height = values
        class_id = int(class_id)
        x_center, y_center = float(x_center) * width, float(y_center) * height
        box_width, box_height = float(box_width) * width, float(box_height) * height
        x1 = int(x_center - box_width / 2)
        y1 = int(y_center - box_height / 2)
        x2 = int(x_center + box_width / 2)
        y2 = int(y_center + box_height / 2)
        color = (0, 255, 0)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(image, CLASS_NAMES.get(class_id, str(class_id)), (x1, max(y1 - 5, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    return image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--file-name", required=True)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "runs" / "demo")
    args = parser.parse_args()

    rgb, ir, label_path = load_pair(args.data_root, args.split, args.file_name)
    labels = label_path.read_text(encoding="utf-8").splitlines() if label_path.exists() else []
    output_dir = args.output_dir if args.output_dir.is_absolute() else PROJECT_ROOT / args.output_dir
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_dir / "rgb_labels.jpg"), draw_labels(rgb.copy(), labels))
    cv2.imwrite(str(output_dir / "ir_labels.jpg"), draw_labels(ir.copy(), labels))
    print(f"Saved label visualizations to: {output_dir}")


if __name__ == "__main__":
    main()
