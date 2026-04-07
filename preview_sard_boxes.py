import json
import random
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image, ImageDraw


CLASS_NAMES = [
    "Running",
    "Walking",
    "Laying down",
    "Not defined",
    "Seated",
    "Standing",
]


def load_metadata(jsonl_path):
    rows = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def shorten_text(text):
    replacements = {
        "drone search and rescue image showing a ": "",
        "aerial drone view of a ": "",
        "aerial SAR scene with a ": "",
        "person.": "",
        "human posture present.": "",
    }
    cleaned = text
    for old, new in replacements.items():
        cleaned = cleaned.replace(old, new)
    return cleaned.strip().capitalize()


def yolo_to_xyxy(x_center, y_center, width, height, img_w, img_h):
    x_center *= img_w
    y_center *= img_h
    width *= img_w
    height *= img_h

    x1 = x_center - width / 2
    y1 = y_center - height / 2
    x2 = x_center + width / 2
    y2 = y_center + height / 2

    return int(x1), int(y1), int(x2), int(y2)


def get_label_path(image_path: Path, dataset_root: Path):
    rel = image_path.relative_to(dataset_root)
    parts = list(rel.parts)

    # train/images/abc.jpg -> train/labels/abc.txt
    if "images" in parts:
        idx = parts.index("images")
        parts[idx] = "labels"

    label_path = dataset_root.joinpath(*parts).with_suffix(".txt")
    return label_path


def draw_boxes(image_path: Path, dataset_root: Path):
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)

    img_w, img_h = img.size
    label_path = get_label_path(image_path, dataset_root)

    if label_path.exists():
        with open(label_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue

                class_id = int(float(parts[0]))
                x_center, y_center, width, height = map(float, parts[1:])

                x1, y1, x2, y2 = yolo_to_xyxy(
                    x_center, y_center, width, height, img_w, img_h
                )

                class_name = CLASS_NAMES[class_id] if 0 <= class_id < len(CLASS_NAMES) else f"Class {class_id}"

                draw.rectangle([x1, y1, x2, y2], outline="red", width=3)

                text_x = x1
                text_y = max(0, y1 - 18)
                draw.rectangle([text_x, text_y, text_x + 110, text_y + 18], fill="red")
                draw.text((text_x + 4, text_y + 2), class_name, fill="white")

    return img


def main():
    metadata_path = Path("data/sard_text/train/metadata.jsonl")
    dataset_root = Path("data/search-and-rescue-2")

    rows = load_metadata(metadata_path)
    sample = random.sample(rows, min(9, len(rows)))

    cols = 3
    rows_count = (len(sample) + cols - 1) // cols

    fig, axes = plt.subplots(rows_count, cols, figsize=(13, 4.5 * rows_count))
    axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for ax, row in zip(axes, sample):
        image_path = dataset_root / row["image"]
        img = draw_boxes(image_path, dataset_root)

        label = shorten_text(row["text"])
        label = "\n".join(textwrap.wrap(label, width=24))

        ax.imshow(img)
        ax.axis("off")
        ax.text(
            0.5,
            -0.08,
            label,
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=10,
            bbox=dict(
                facecolor="white",
                alpha=0.9,
                edgecolor="gray",
                boxstyle="round,pad=0.3",
            ),
        )

    for ax in axes[len(sample):]:
        ax.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()