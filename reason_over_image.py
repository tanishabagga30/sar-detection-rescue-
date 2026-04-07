import argparse
import json
from pathlib import Path
from typing import List, Optional

import requests
import torch
from PIL import Image
from transformers import AutoProcessor, CLIPModel


def score_labels(model, processor, image_path: str, labels: List[str], device: torch.device):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(text=labels, images=[image], return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        probs = outputs.logits_per_image.softmax(dim=-1)[0].detach().cpu().tolist()
    ranked = sorted(zip(labels, probs), key=lambda x: x[1], reverse=True)
    return ranked


def call_ollama(model_name: str, image_path: str, ranked_labels, url: str = "http://localhost:11434/api/generate") -> str:
    structured = {
        "top_matches": [{"label": label, "score": round(score, 4)} for label, score in ranked_labels[:5]],
        "task": "Reason over likely SAR scene semantics from the ranked visual matches. Avoid giving generic rescue advice. Focus on what the image itself likely contains or implies.",
    }
    prompt = (
        "You are analyzing an aerial or disaster-response image. "
        "You are only given structured visual matches from a tuned vision model. "
        "Infer the most plausible scene interpretation, mention uncertainty clearly, and keep it short.\n\n"
        + json.dumps(structured, indent=2)
    )
    payload = {"model": model_name, "prompt": prompt, "stream": False}
    r = requests.post(url, json=payload, timeout=120)
    r.raise_for_status()
    data = r.json()
    return data.get("response", "")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--labels", nargs="+", required=True)
    parser.add_argument("--ollama_model", default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = AutoProcessor.from_pretrained(args.model_dir)
    model = CLIPModel.from_pretrained(args.model_dir).to(device)
    model.eval()

    ranked = score_labels(model, processor, args.image, args.labels, device)
    print("Top matches:")
    for label, score in ranked[:10]:
        print(f"{label:40s} {score:.4f}")

    if args.ollama_model:
        print("\nLLM reasoning:\n")
        print(call_ollama(args.ollama_model, args.image, ranked))


if __name__ == "__main__":
    main()
