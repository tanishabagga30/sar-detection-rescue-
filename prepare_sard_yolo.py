import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import yaml

IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}


def load_names(data_yaml: Path) -> List[str]:
    data = yaml.safe_load(data_yaml.read_text())
    names = data.get('names', [])
    if isinstance(names, dict):
        max_idx = max(int(k) for k in names.keys())
        ordered = [''] * (max_idx + 1)
        for k, v in names.items():
            ordered[int(k)] = str(v)
        names = ordered
    return [normalize_label(str(x)) for x in names]


def normalize_label(label: str) -> str:
    return label.strip().lower().replace('_', ' ')


def class_phrase(label: str, count: int) -> str:
    if count == 1:
        article = 'an' if label[0] in 'aeiou' else 'a'
        return f"{article} {label} person"
    return f"{count} {label} people"


def scene_text_from_counts(counts: Counter) -> str:
    total = sum(counts.values())
    if total == 0:
        return 'drone search and rescue image with no labeled person.'
    parts = [class_phrase(label, counts[label]) for label in sorted(counts.keys())]
    joined = ', '.join(parts)
    return f'drone search and rescue image showing {joined}.'


def reasoning_text_from_counts(counts: Counter) -> str:
    total = sum(counts.values())
    labels = set(counts.keys())
    base = []
    if total == 0:
        return 'aerial search and rescue frame without annotated people.'
    if 'laying down' in labels:
        base.append('possible distressed or injured posture present')
    if 'running' in labels:
        base.append('active movement present')
    if 'walking' in labels:
        base.append('slow movement present')
    if 'seated' in labels:
        base.append('stationary seated posture present')
    if 'stands' in labels or 'standing' in labels:
        base.append('upright standing posture present')
    if 'not defined' in labels:
        base.append('ambiguous human posture present')
    if not base:
        base.append('human activity visible from drone view')
    return 'aerial SAR scene; ' + '; '.join(base) + '.'


def parse_label_file(label_path: Path, names: List[str]) -> Counter:
    counts: Counter = Counter()
    if not label_path.exists():
        return counts
    for line in label_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        cls_idx = int(float(parts[0]))
        if 0 <= cls_idx < len(names):
            counts[names[cls_idx]] += 1
    return counts


def collect_pairs(images_dir: Path, labels_dir: Path, names: List[str]) -> List[Tuple[Path, Counter]]:
    rows: List[Tuple[Path, Counter]] = []
    for img_path in sorted(images_dir.rglob('*')):
        if img_path.suffix.lower() not in IMAGE_EXTS:
            continue
        label_path = labels_dir / (img_path.stem + '.txt')
        counts = parse_label_file(label_path, names)
        rows.append((img_path, counts))
    return rows


def write_split(rows: List[Tuple[Path, Counter]], split_root: Path, source_root: Path) -> None:
    split_root.mkdir(parents=True, exist_ok=True)
    metadata_path = split_root / 'metadata.jsonl'
    with metadata_path.open('w', encoding='utf-8') as f:
        for img_path, counts in rows:
            rel_path = img_path.relative_to(source_root).as_posix()
            # duplicate each image with two related but distinct captions for a bit more supervision
            texts = [scene_text_from_counts(counts), reasoning_text_from_counts(counts)]
            for text in texts:
                f.write(json.dumps({'image': rel_path, 'text': text}, ensure_ascii=False) + '\n')


def maybe_limit(rows: List[Tuple[Path, Counter]], max_items: int, seed: int) -> List[Tuple[Path, Counter]]:
    if max_items <= 0 or len(rows) <= max_items:
        return rows
    rnd = random.Random(seed)
    rows = rows.copy()
    rnd.shuffle(rows)
    return sorted(rows[:max_items], key=lambda x: x[0].name)


def main() -> None:
    p = argparse.ArgumentParser(description='Convert SARD YOLO dataset into JSONL image-text data for lightweight CLIP adaptation.')
    p.add_argument('--dataset_dir', required=True, help='Root of exported YOLO dataset containing train/valid/test and data.yaml')
    p.add_argument('--output_dir', required=True, help='Output dir for converted dataset')
    p.add_argument('--train_limit', type=int, default=900, help='Max train images to use; 0 means use all')
    p.add_argument('--val_limit', type=int, default=250, help='Max val images to use; 0 means use all')
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()

    dataset_dir = Path(args.dataset_dir)
    output_dir = Path(args.output_dir)
    names = load_names(dataset_dir / 'data.yaml')

    split_aliases = {'train': 'train', 'valid': 'val', 'val': 'val'}
    used = {}
    for src_split, out_split in split_aliases.items():
        images_dir = dataset_dir / src_split / 'images'
        labels_dir = dataset_dir / src_split / 'labels'
        if not images_dir.exists() or not labels_dir.exists():
            continue
        rows = collect_pairs(images_dir, labels_dir, names)
        if out_split == 'train':
            rows = maybe_limit(rows, args.train_limit, args.seed)
        elif out_split == 'val':
            rows = maybe_limit(rows, args.val_limit, args.seed + 1)
        split_root = output_dir / out_split
        write_split(rows, split_root, dataset_dir)
        used[out_split] = len(rows)

    summary = {
        'classes': names,
        'limits': {'train': args.train_limit, 'val': args.val_limit},
        'images_used': used,
        'notes': 'Each image is written twice with two SAR-specific captions derived from the YOLO labels.'
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
