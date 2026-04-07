import json
import random
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

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


def yolo_to_xyxy(x_center, y_center, width, height, img_w, img_h):
    x_center *= img_w
    y_center *= img_h
    width *= img_w
    height *= img_h

    x1 = int(x_center - width / 2)
    y1 = int(y_center - height / 2)
    x2 = int(x_center + width / 2)
    y2 = int(y_center + height / 2)

    return x1, y1, x2, y2


def get_label_path(image_path: Path, dataset_root: Path):
    rel = image_path.relative_to(dataset_root)
    parts = list(rel.parts)
    if "images" in parts:
        parts[parts.index("images")] = "labels"
    return dataset_root.joinpath(*parts).with_suffix(".txt")


def get_font(size=20):
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Supplemental/Helvetica.ttc",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            pass
    return ImageFont.load_default()


def draw_box_with_label(draw, box, label, font):
    x1, y1, x2, y2 = box
    draw.rectangle([x1, y1, x2, y2], outline="red", width=4)

    bbox = draw.textbbox((0, 0), label, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]

    pad = 6
    tx1 = x1
    ty1 = max(0, y1 - th - 2 * pad)
    tx2 = x1 + tw + 2 * pad
    ty2 = ty1 + th + 2 * pad

    draw.rectangle([tx1, ty1, tx2, ty2], fill="red")
    draw.text((tx1 + pad, ty1 + pad), label, fill="white", font=font)


def add_description_panel(img, description, font):
    img = img.convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    wrapped = textwrap.wrap(description, width=42)
    if not wrapped:
        wrapped = ["No description"]

    line_heights = []
    max_width = 0
    for line in wrapped:
        bbox = draw.textbbox((0, 0), line, font=font)
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        max_width = max(max_width, w)
        line_heights.append(h)

    pad = 12
    gap = 6
    panel_w = max_width + 2 * pad
    panel_h = sum(line_heights) + gap * (len(wrapped) - 1) + 2 * pad

    x1 = 12
    y1 = img.height - panel_h - 12
    x2 = x1 + panel_w
    y2 = y1 + panel_h

    draw.rounded_rectangle([x1, y1, x2, y2], radius=12, fill=(0, 0, 0, 165))

    y = y1 + pad
    for line, h in zip(wrapped, line_heights):
        draw.text((x1 + pad, y), line, font=font, fill="white")
        y += h + gap

    combined = Image.alpha_composite(img, overlay)
    return combined.convert("RGB")


def process_one(row, dataset_root: Path, out_dir: Path, box_font, desc_font):
    image_path = dataset_root / row["image"]
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)

    label_path = get_label_path(image_path, dataset_root)

    if label_path.exists():
        with open(label_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue

                class_id = int(float(parts[0]))
                xc, yc, bw, bh = map(float, parts[1:])
                box = yolo_to_xyxy(xc, yc, bw, bh, img.width, img.height)

                class_name = CLASS_NAMES[class_id] if 0 <= class_id < len(CLASS_NAMES) else f"Class {class_id}"
                draw_box_with_label(draw, box, class_name, box_font)

    description = row.get("text", "No description available.")
    img = add_description_panel(img, description, desc_font)

    out_path = out_dir / Path(row["image"]).name
    img.save(out_path)
    print(f"Saved: {out_path}")


def main():
    metadata_path = Path("data/sard_text/train/metadata.jsonl")
    dataset_root = Path("data/search-and-rescue-2")
    out_dir = Path("data/sard_text/boxed_with_description")
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_metadata(metadata_path)
    sample = random.sample(rows, min(12, len(rows)))

    box_font = get_font(20)
    desc_font = get_font(18)

    for row in sample:
        process_one(row, dataset_root, out_dir, box_font, desc_font)

    print(f"\nDone. Open images in: {out_dir}")


if __name__ == "__main__":
    main()