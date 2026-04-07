import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tif', '.tiff'}


def is_image(p: Path) -> bool:
    return p.suffix.lower() in IMAGE_EXTS


def write_jsonl(rows: List[dict], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')


def flat_with_sidecars(images_dir: Path, output: Path, default_text: str) -> int:
    rows = []
    files = sorted([p for p in images_dir.rglob('*') if is_image(p)])
    for img in files:
        txt_file = img.with_suffix('.txt')
        text = txt_file.read_text(encoding='utf-8').strip() if txt_file.exists() else default_text
        rows.append({'image': str(img.relative_to(output.parent)), 'text': text})
    write_jsonl(rows, output)
    return len(rows)


def class_folders(images_dir: Path, output: Path, template: str) -> int:
    rows = []
    files = sorted([p for p in images_dir.rglob('*') if is_image(p)])
    for img in files:
        label = img.parent.name.replace('_', ' ').replace('-', ' ')
        rows.append({
            'image': str(img.relative_to(output.parent)),
            'text': template.format(label=label)
        })
    write_jsonl(rows, output)
    return len(rows)


def from_csv(csv_path: Path, images_dir: Path, output: Path, image_col: str, text_col: str) -> int:
    rows = []
    with csv_path.open('r', encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        for rec in reader:
            img = images_dir / rec[image_col]
            if not img.exists():
                continue
            rows.append({'image': str(img.relative_to(output.parent)), 'text': rec[text_col].strip()})
    write_jsonl(rows, output)
    return len(rows)


def from_coco(coco_json: Path, images_dir: Path, output: Path, caption_field: str = 'caption') -> int:
    data = json.loads(coco_json.read_text(encoding='utf-8'))
    id_to_name = {img['id']: img['file_name'] for img in data.get('images', [])}
    grouped: Dict[int, List[str]] = {}
    for ann in data.get('annotations', []):
        txt = ann.get(caption_field) or ann.get('text') or ann.get('label')
        if txt:
            grouped.setdefault(ann['image_id'], []).append(str(txt).strip())
    rows = []
    for image_id, texts in grouped.items():
        img = images_dir / id_to_name[image_id]
        if not img.exists():
            continue
        rows.append({'image': str(img.relative_to(output.parent)), 'text': ' ; '.join(texts[:3])})
    write_jsonl(rows, output)
    return len(rows)


def from_vqa(json_path: Path, images_dir: Path, output: Path) -> int:
    data = json.loads(json_path.read_text(encoding='utf-8'))
    rows = []

    if isinstance(data, dict) and 'questions' in data and 'annotations' in data:
        q_by_id = {q['question_id']: q for q in data['questions']}
        for ann in data['annotations']:
            q = q_by_id.get(ann['question_id'])
            if not q:
                continue
            image_name = q.get('image') or q.get('file_name') or q.get('image_name')
            if not image_name and 'image_id' in q:
                image_name = f"{q['image_id']}.jpg"
            if not image_name:
                continue
            img = images_dir / image_name
            if not img.exists():
                continue
            answer = ann.get('multiple_choice_answer') or ann.get('answer') or ''
            prompt = f"Q: {q['question'].strip()} A: {str(answer).strip()}"
            rows.append({'image': str(img.relative_to(output.parent)), 'text': prompt})
    elif isinstance(data, list):
        for rec in data:
            image_name = rec.get('image') or rec.get('file_name') or rec.get('image_name')
            text = rec.get('text') or rec.get('caption')
            if not image_name or not text:
                continue
            img = images_dir / image_name
            if not img.exists():
                continue
            rows.append({'image': str(img.relative_to(output.parent)), 'text': str(text).strip()})
    else:
        raise ValueError('Unsupported VQA/JSON structure')

    write_jsonl(rows, output)
    return len(rows)


def main() -> None:
    p = argparse.ArgumentParser(description='Convert common SAR/remote-sensing dataset layouts into metadata.jsonl')
    p.add_argument('--format', required=True, choices=['flat_sidecar', 'class_folders', 'csv', 'coco', 'vqa'])
    p.add_argument('--images_dir', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--default_text', default='aerial SAR scene')
    p.add_argument('--template', default='aerial SAR scene of {label}')
    p.add_argument('--csv_path')
    p.add_argument('--image_col', default='image')
    p.add_argument('--text_col', default='text')
    p.add_argument('--json_path')
    args = p.parse_args()

    images_dir = Path(args.images_dir)
    output = Path(args.output)

    if args.format == 'flat_sidecar':
        n = flat_with_sidecars(images_dir, output, args.default_text)
    elif args.format == 'class_folders':
        n = class_folders(images_dir, output, args.template)
    elif args.format == 'csv':
        if not args.csv_path:
            raise SystemExit('--csv_path is required for --format csv')
        n = from_csv(Path(args.csv_path), images_dir, output, args.image_col, args.text_col)
    elif args.format == 'coco':
        if not args.json_path:
            raise SystemExit('--json_path is required for --format coco')
        n = from_coco(Path(args.json_path), images_dir, output)
    elif args.format == 'vqa':
        if not args.json_path:
            raise SystemExit('--json_path is required for --format vqa')
        n = from_vqa(Path(args.json_path), images_dir, output)
    else:
        raise SystemExit('Unsupported format')

    print(f'Wrote {n} rows to {output}')


if __name__ == '__main__':
    main()
