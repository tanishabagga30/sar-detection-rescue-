from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import (
    AutoProcessor,
    AutoModelForZeroShotObjectDetection,
    CLIPModel,
    CLIPProcessor,
)

# --------------------------------------------------
# CONFIG
# --------------------------------------------------
OWL_MODEL = "google/owlv2-base-patch16-ensemble"
CLIP_MODEL_PATH = "./outputs/clip-sard-adapted"

OUTPUT_DIR = Path("outputs/owlv2_clip_results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

OWL_TEXT_LABELS = [
    "person",
    "human"
]

SCENE_PROMPTS = [
    "drone search and rescue image showing a running person",
    "drone search and rescue image showing a walking person",
    "drone search and rescue image showing a laying down person",
    "drone search and rescue image showing a seated person",
    "drone search and rescue image showing a standing person",
    "drone search and rescue image showing possible distressed or injured posture",
    "drone search and rescue image showing active movement",
    "drone search and rescue image showing stationary seated posture",
    "drone search and rescue image showing mixed human postures in an aerial scene",
    "drone search and rescue image showing no identifiable human presence"
]

PROMPT_SHORT_NAMES = {
    "drone search and rescue image showing a running person": "running",
    "drone search and rescue image showing a walking person": "walking",
    "drone search and rescue image showing a laying down person": "laying_down",
    "drone search and rescue image showing a seated person": "seated",
    "drone search and rescue image showing a standing person": "standing",
    "drone search and rescue image showing possible distressed or injured posture": "distress",
    "drone search and rescue image showing active movement": "active_movement",
    "drone search and rescue image showing stationary seated posture": "stationary_seated",
    "drone search and rescue image showing mixed human postures in an aerial scene": "mixed_postures",
    "drone search and rescue image showing no identifiable human presence": "no_human"
}

# --------------------------------------------------
# NEW: TERRAIN PROMPTS
# --------------------------------------------------
OVERALL_TERRAIN_PROMPTS = [
    "drone search and rescue scene with open accessible terrain",
    "drone search and rescue scene with dense vegetation",
    "drone search and rescue scene with rocky or rubble terrain",
    "drone search and rescue scene with visible road or trail access",
    "drone search and rescue scene with water or wet terrain",
    "drone search and rescue scene with steep slope or uneven ground",
    "drone search and rescue scene with obstructed access for responders",
]

OVERALL_TERRAIN_SHORT = {
    "drone search and rescue scene with open accessible terrain": "open_accessible",
    "drone search and rescue scene with dense vegetation": "dense_vegetation",
    "drone search and rescue scene with rocky or rubble terrain": "rocky_rubble",
    "drone search and rescue scene with visible road or trail access": "road_or_trail",
    "drone search and rescue scene with water or wet terrain": "water_or_wet",
    "drone search and rescue scene with steep slope or uneven ground": "steep_or_uneven",
    "drone search and rescue scene with obstructed access for responders": "obstructed_access",
}

VICTIM_TERRAIN_PROMPTS = [
    "drone view of a person on open flat ground",
    "drone view of a person surrounded by dense vegetation or bushes",
    "drone view of a person near rocky terrain or rubble",
    "drone view of a person near a road trail or clear path",
    "drone view of a person near water or wet ground",
    "drone view of a person near slope ditch or uneven terrain",
    "drone view of a person in obstructed surroundings",
]

VICTIM_TERRAIN_SHORT = {
    "drone view of a person on open flat ground": "open_ground",
    "drone view of a person surrounded by dense vegetation or bushes": "dense_vegetation",
    "drone view of a person near rocky terrain or rubble": "rocky_rubble",
    "drone view of a person near a road trail or clear path": "road_or_path",
    "drone view of a person near water or wet ground": "water_or_wet",
    "drone view of a person near slope ditch or uneven terrain": "slope_or_uneven",
    "drone view of a person in obstructed surroundings": "obstructed_access",
}


# --------------------------------------------------
# LOAD MODELS
# --------------------------------------------------
def load_models():
    owl_processor = AutoProcessor.from_pretrained(OWL_MODEL)
    owl_model = AutoModelForZeroShotObjectDetection.from_pretrained(OWL_MODEL).to(DEVICE)
    owl_model.eval()

    clip_model = CLIPModel.from_pretrained(CLIP_MODEL_PATH).to(DEVICE)
    clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_PATH)
    clip_model.eval()

    return owl_processor, owl_model, clip_processor, clip_model


# --------------------------------------------------
# OWLv2 DETECTION
# --------------------------------------------------
def run_owlv2_tiled(
    image: Image.Image,
    processor,
    model,
    text_labels: List[str],
    tile_size: int = 640,
    overlap: int = 160,
    threshold: float = 0.08,
) -> List[Dict]:
    W, H = image.size
    detections = []

    step = tile_size - overlap
    xs = list(range(0, max(1, W - tile_size + 1), step))
    ys = list(range(0, max(1, H - tile_size + 1), step))

    if not xs or xs[-1] != max(0, W - tile_size):
        xs.append(max(0, W - tile_size))
    if not ys or ys[-1] != max(0, H - tile_size):
        ys.append(max(0, H - tile_size))

    for x0 in xs:
        for y0 in ys:
            x1 = min(x0 + tile_size, W)
            y1 = min(y0 + tile_size, H)

            tile = image.crop((x0, y0, x1, y1))

            inputs = processor(text=[text_labels], images=tile, return_tensors="pt")
            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model(**inputs)

            results = processor.post_process_grounded_object_detection(
                outputs=outputs,
                threshold=threshold,
                target_sizes=[(tile.height, tile.width)],
                text_labels=[text_labels],
            )[0]

            for box, score, label in zip(
                results["boxes"], results["scores"], results["text_labels"]
            ):
                bx1, by1, bx2, by2 = [float(v) for v in box.tolist()]

                detections.append(
                    {
                        "bbox_xyxy": [
                            int(round(bx1 + x0)),
                            int(round(by1 + y0)),
                            int(round(bx2 + x0)),
                            int(round(by2 + y0)),
                        ],
                        "score": float(score.item()),
                        "label": str(label),
                    }
                )

    detections = merge_close_detections(detections, iou_thresh=0.25, contain_thresh=0.8)
    return detections


def box_iou_xyxy(a: List[int], b: List[int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter = inter_w * inter_h

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)

    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return inter / union


def box_area(box):
    x1, y1, x2, y2 = box
    return max(0, x2 - x1) * max(0, y2 - y1)


def box_iou_xyxy(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter = inter_w * inter_h

    area_a = box_area(a)
    area_b = box_area(b)
    union = area_a + area_b - inter

    if union <= 0:
        return 0.0
    return inter / union


def box_containment_ratio(inner, outer):
    ix1, iy1, ix2, iy2 = inner
    ox1, oy1, ox2, oy2 = outer

    inter_x1 = max(ix1, ox1)
    inter_y1 = max(iy1, oy1)
    inter_x2 = min(ix2, ox2)
    inter_y2 = min(iy2, oy2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter = inter_w * inter_h

    inner_area = box_area(inner)
    if inner_area <= 0:
        return 0.0

    return inter / inner_area


def merge_close_detections(detections, iou_thresh=0.25, contain_thresh=0.8):
    if not detections:
        return []

    detections = sorted(detections, key=lambda d: d["score"], reverse=True)
    kept = []

    for det in detections:
        curr_box = det["bbox_xyxy"]
        should_keep = True

        for prev in kept:
            prev_box = prev["bbox_xyxy"]

            iou = box_iou_xyxy(curr_box, prev_box)
            contain1 = box_containment_ratio(curr_box, prev_box)
            contain2 = box_containment_ratio(prev_box, curr_box)

            if iou >= iou_thresh or contain1 >= contain_thresh or contain2 >= contain_thresh:
                should_keep = False
                break

        if should_keep:
            kept.append(det)

    return kept


# --------------------------------------------------
# CLIP
# --------------------------------------------------
def run_clip_on_full_image(
    image: Image.Image,
    clip_processor: CLIPProcessor,
    clip_model: CLIPModel,
    prompts: List[str],
) -> Dict[str, float]:
    inputs = clip_processor(
        text=prompts,
        images=image,
        return_tensors="pt",
        padding=True,
    )
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = clip_model(**inputs)
        logits = outputs.logits_per_image
        probs = torch.softmax(logits, dim=1)[0].cpu().tolist()

    return {prompt: float(score) for prompt, score in zip(prompts, probs)}


# --------------------------------------------------
# NEW: TERRAIN HELPERS
# --------------------------------------------------
def expand_box(box: List[int], image_size: Tuple[int, int], context_scale: float = 2.2) -> List[int]:
    x1, y1, x2, y2 = box
    W, H = image_size

    bw = x2 - x1
    bh = y2 - y1
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2

    new_w = bw * context_scale
    new_h = bh * context_scale

    nx1 = max(0, int(round(cx - new_w / 2)))
    ny1 = max(0, int(round(cy - new_h / 2)))
    nx2 = min(W, int(round(cx + new_w / 2)))
    ny2 = min(H, int(round(cy + new_h / 2)))

    return [nx1, ny1, nx2, ny2]


def analyze_overall_terrain(
    image: Image.Image,
    clip_processor: CLIPProcessor,
    clip_model: CLIPModel,
) -> Dict:
    terrain_scores = run_clip_on_full_image(
        image=image,
        clip_processor=clip_processor,
        clip_model=clip_model,
        prompts=OVERALL_TERRAIN_PROMPTS,
    )

    top_prompt, top_score = sorted(terrain_scores.items(), key=lambda x: x[1], reverse=True)[0]

    return {
        "primary_terrain_label": OVERALL_TERRAIN_SHORT[top_prompt],
        "primary_terrain_confidence": round(float(top_score), 6),
        "terrain_scores": {
            OVERALL_TERRAIN_SHORT[k]: round(v, 6) for k, v in terrain_scores.items()
        }
    }


def analyze_victim_surroundings(
    image: Image.Image,
    detections: List[Dict],
    clip_processor: CLIPProcessor,
    clip_model: CLIPModel,
    context_scale: float = 2.2,
) -> List[Dict]:
    victims = []

    for idx, det in enumerate(detections, start=1):
        context_box = expand_box(det["bbox_xyxy"], image.size, context_scale=context_scale)
        crop = image.crop(tuple(context_box))

        terrain_scores = run_clip_on_full_image(
            image=crop,
            clip_processor=clip_processor,
            clip_model=clip_model,
            prompts=VICTIM_TERRAIN_PROMPTS,
        )

        top_prompt, top_score = sorted(terrain_scores.items(), key=lambda x: x[1], reverse=True)[0]

        victims.append({
            "victim_id": idx,
            "bbox_xyxy": det["bbox_xyxy"],
            "context_bbox_xyxy": context_box,
            "terrain_label": VICTIM_TERRAIN_SHORT[top_prompt],
            "terrain_confidence": round(float(top_score), 6),
            "terrain_scores": {
                VICTIM_TERRAIN_SHORT[k]: round(v, 6) for k, v in terrain_scores.items()
            }
        })

    return victims


# --------------------------------------------------
# PRIORITY
# --------------------------------------------------
def compute_scene_priority(person_count: int, clip_scores: Dict[str, float]) -> Tuple[int, str, str]:
    laying = clip_scores["drone search and rescue image showing a laying down person"]
    seated = clip_scores["drone search and rescue image showing a seated person"]
    standing = clip_scores["drone search and rescue image showing a standing person"]
    walking = clip_scores["drone search and rescue image showing a walking person"]
    running = clip_scores["drone search and rescue image showing a running person"]
    distress = clip_scores["drone search and rescue image showing possible distressed or injured posture"]
    active = clip_scores["drone search and rescue image showing active movement"]
    stationary = clip_scores["drone search and rescue image showing stationary seated posture"]
    mixed = clip_scores["drone search and rescue image showing mixed human postures in an aerial scene"]

    score = 0
    reasons = []

    if person_count >= 2 and laying >= 0.18:
        score += 45
        reasons.append("multiple people with lying-down evidence")

    elif person_count >= 1 and laying >= 0.18:
        score += 30
        reasons.append("lying-down evidence present")

    if distress >= 0.18:
        score += 25
        reasons.append("distress-like posture signal")

    if seated >= 0.16 or stationary >= 0.16:
        score += 15
        reasons.append("stationary posture evidence")

    if person_count == 1:
        score += 8
    elif person_count == 2:
        score += 18
    elif person_count >= 3:
        score += 28

    if active >= 0.22:
        score -= 12
        reasons.append("active movement present")

    if running >= 0.18:
        score -= 10

    if walking >= 0.18:
        score -= 6

    if standing >= 0.20 and laying < 0.12 and distress < 0.12:
        score -= 5

    if mixed >= 0.15:
        score += 8
        reasons.append("mixed posture scene")

    score = max(0, min(score, 100))

    if person_count >= 2 and (laying >= 0.18 or distress >= 0.18):
        tier = "P1"
    elif laying >= 0.18 or distress >= 0.18:
        tier = "P2" if person_count <= 1 else "P1"
    elif seated >= 0.16 or stationary >= 0.16:
        tier = "P3"
    else:
        tier = "P4"

    if not reasons:
        reasons.append("low-distress scene pattern")

    return score, tier, ", ".join(reasons).capitalize() + "."


# --------------------------------------------------
# DRAWING
# --------------------------------------------------
def get_font(size=18):
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


def draw_detections(img: Image.Image, detections: List[Dict]) -> Image.Image:
    draw = ImageDraw.Draw(img)
    font = get_font(16)

    for det in detections:
        x1, y1, x2, y2 = det["bbox_xyxy"]
        score = det["score"]
        label = det["label"]

        draw.rectangle([x1, y1, x2, y2], outline="yellow", width=3)

        text = f"{label} {score:.2f}"
        bbox = draw.textbbox((0, 0), text, font=font)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]

        y_top = max(0, y1 - th - 8)
        draw.rectangle([x1, y_top, x1 + tw + 8, y1], fill="yellow")
        draw.text((x1 + 4, y_top + 2), text, fill="black", font=font)

    return img


def add_panel(
    img: Image.Image,
    person_count: int,
    priority_score: int,
    tier: str,
    reason: str,
    clip_scores: Dict[str, float],
    overall_terrain: Dict | None = None,
) -> Image.Image:
    draw = ImageDraw.Draw(img)
    font = get_font(18)
    small_font = get_font(15)

    top_scores = sorted(clip_scores.items(), key=lambda x: x[1], reverse=True)[:4]
    top_lines = [f"{PROMPT_SHORT_NAMES[p]}: {s:.3f}" for p, s in top_scores]

    panel_h = 170
    overlay_y = img.height - panel_h
    draw.rectangle([0, overlay_y, img.width, img.height], fill="black")

    line1 = f"People detected: {person_count} | Scene Priority: {priority_score} ({tier})"
    line2 = reason[:130]

    y = overlay_y + 10
    draw.text((12, y), line1, fill="white", font=font)
    y += 30
    draw.text((12, y), line2, fill="white", font=small_font)
    y += 24

    if overall_terrain is not None:
        terrain_line = (
            f"Overall terrain: "
            f"{overall_terrain['primary_terrain_label']} "
            f"({overall_terrain['primary_terrain_confidence']:.2f})"
        )
        draw.text((12, y), terrain_line, fill="white", font=small_font)
        y += 24

    for line in top_lines:
        draw.text((12, y), line, fill="white", font=small_font)
        y += 22

    return img


# --------------------------------------------------
# PIPELINE
# --------------------------------------------------
def process_image(
    image_path: str,
    owl_processor,
    owl_model,
    clip_processor,
    clip_model,
) -> Dict:
    image = Image.open(image_path).convert("RGB")

    detections = run_owlv2_tiled(
        image=image,
        processor=owl_processor,
        model=owl_model,
        text_labels=OWL_TEXT_LABELS,
        tile_size=512,
        overlap=192,
        threshold=0.08,
    )

    clip_scores = run_clip_on_full_image(
        image=image,
        clip_processor=clip_processor,
        clip_model=clip_model,
        prompts=SCENE_PROMPTS,
    )

    overall_terrain = analyze_overall_terrain(
        image=image,
        clip_processor=clip_processor,
        clip_model=clip_model,
    )

    victim_terrain = analyze_victim_surroundings(
        image=image,
        detections=detections,
        clip_processor=clip_processor,
        clip_model=clip_model,
        context_scale=2.2,
    )

    person_count = len(detections)
    priority_score, tier, reason = compute_scene_priority(person_count, clip_scores)

    vis = image.copy()
    vis = draw_detections(vis, detections)
    vis = add_panel(
        vis,
        person_count,
        priority_score,
        tier,
        reason,
        clip_scores,
        overall_terrain=overall_terrain,
    )

    save_path = OUTPUT_DIR / Path(image_path).name
    vis.save(save_path)

    return {
        "image_name": Path(image_path).name,
        "annotated_image": str(save_path),
        "person_count": person_count,
        "scene_priority_score": priority_score,
        "priority_tier": tier,
        "ranking_reason": reason,
        "clip_scores": {PROMPT_SHORT_NAMES[k]: round(v, 6) for k, v in clip_scores.items()},
        "overall_terrain": overall_terrain,
        "victim_terrain": victim_terrain,
        "detections": detections,
    }


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str)
    parser.add_argument("--image_dir", type=str)
    args = parser.parse_args()

    owl_processor, owl_model, clip_processor, clip_model = load_models()

    results = []
    if args.image:
        results.append(process_image(args.image, owl_processor, owl_model, clip_processor, clip_model))
    elif args.image_dir:
        for p in sorted(Path(args.image_dir).glob("*")):
            if p.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                results.append(process_image(str(p), owl_processor, owl_model, clip_processor, clip_model))
    else:
        raise SystemExit("Provide --image or --image_dir")

    results = sorted(results, key=lambda x: x["scene_priority_score"], reverse=True)

    with open(OUTPUT_DIR / "ranking.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"Done. Results saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()