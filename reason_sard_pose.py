import argparse
from typing import List

import torch
from transformers import AutoProcessor, CLIPModel
from PIL import Image

DEFAULT_LABELS = [
    'aerial drone view of a running person',
    'aerial drone view of a walking person',
    'aerial drone view of a person laying down',
    'aerial drone view of a seated person',
    'aerial drone view of a standing person',
    'aerial drone view of a person with undefined posture',
    'drone search and rescue image showing possible distressed posture',
    'drone search and rescue image showing active movement',
]


def score(model, processor, image_path: str, labels: List[str], device: torch.device):
    image = Image.open(image_path).convert('RGB')
    inputs = processor(text=labels, images=[image], return_tensors='pt', padding=True, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        probs = outputs.logits_per_image.softmax(dim=-1)[0].cpu().tolist()
    return sorted(zip(labels, probs), key=lambda x: x[1], reverse=True)


def main():
    p = argparse.ArgumentParser(description='Run SARD-style pose/activity retrieval with the adapted CLIP model.')
    p.add_argument('--model_dir', required=True)
    p.add_argument('--image', required=True)
    p.add_argument('--labels', nargs='*', default=DEFAULT_LABELS)
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    processor = AutoProcessor.from_pretrained(args.model_dir)
    model = CLIPModel.from_pretrained(args.model_dir).to(device)
    model.eval()

    ranked = score(model, processor, args.image, args.labels, device)
    print('Top matches:')
    for label, prob in ranked:
        print(f'{label:60s} {prob:.4f}')


if __name__ == '__main__':
    main()
