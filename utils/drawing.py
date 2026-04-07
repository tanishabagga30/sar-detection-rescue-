from typing import Dict, List, Tuple
from PIL import ImageDraw, ImageFont, Image

def get_font(size: int = 18):
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Supplemental/Helvetica.ttc",
        "C:\\Windows\\Fonts\\arial.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            continue
    return ImageFont.load_default()

def draw_detections(img: Image.Image, detections: List[Dict]) -> Image.Image:
    draw = ImageDraw.Draw(img)
    font = get_font(16)

    for det in detections:
        x1, y1, x2, y2 = det["bbox_xyxy"]
        score = det["score"]
        label = det["label"]

        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)

        text = f"{label} {score:.2f}"
        bbox = draw.textbbox((0, 0), text, font=font)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]

        y_top = max(0, y1 - th - 8)
        draw.rectangle([x1, y_top, x1 + tw + 8, y1], fill="red")
        draw.text((x1 + 4, y_top + 2), text, fill="white", font=font)

    return img

def add_panel(
    img: Image.Image,
    detections: List[Dict],
    priority_score: int,
    tier: str,
    reason: str,
    terrain: str,
    activity: str
) -> Image.Image:
    draw = ImageDraw.Draw(img)
    font = get_font(18)
    small_font = get_font(15)

    panel_h = 170
    overlay_y = img.height - panel_h
    # Dark semi-transparent or black background panel
    draw.rectangle([0, overlay_y, img.width, img.height], fill="black")

    line1 = f"Objects detected: {len(detections)} | Priority: {priority_score} ({tier})"
    line2 = f"Reason: {reason[:100]}"
    line3 = f"Terrain: {terrain}"
    line4 = f"Activity: {activity}"

    y = overlay_y + 10
    draw.text((12, y), line1, fill="white", font=font)
    y += 30
    draw.text((12, y), line2, fill="white", font=small_font)
    y += 24
    draw.text((12, y), line3, fill="white", font=small_font)
    y += 24
    draw.text((12, y), line4, fill="white", font=small_font)

    return img
