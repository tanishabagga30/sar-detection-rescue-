import argparse
import json
import os
import random
from pathlib import Path

from datasets import load_dataset
from PIL import Image


def save_split(dataset, split_name: str, out_dir: Path, limit: int, seed: int) -> None:
    rng = random.Random(seed)
    indices = list(range(len(dataset)))
    rng.shuffle(indices)
    indices = indices[:limit]

    images_dir = out_dir / split_name / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    meta_path = out_dir / split_name / "metadata.jsonl"

    with meta_path.open("w", encoding="utf-8") as f:
        for idx, ds_idx in enumerate(indices):
            row = dataset[ds_idx]
            image = row["image"]
            if not isinstance(image, Image.Image):
                image = Image.open(image).convert("RGB")
            else:
                image = image.convert("RGB")

            captions = row.get("captions") or row.get("sentences") or row.get("caption")
            if isinstance(captions, list):
                text = captions[0]
            else:
                text = str(captions)

            img_name = f"{idx:06d}.jpg"
            image.save(images_dir / img_name, quality=95)
            f.write(json.dumps({"image": f"images/{img_name}", "text": text}, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="arampacha/rsicd")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--train_size", type=int, default=1500)
    parser.add_argument("--val_size", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ds = load_dataset(args.dataset)

    train_ds = ds["train"] if "train" in ds else ds[list(ds.keys())[0]]
    val_ds = ds["validation"] if "validation" in ds else ds.get("test", train_ds)

    save_split(train_ds, "train", out_dir, args.train_size, args.seed)
    save_split(val_ds, "val", out_dir, args.val_size, args.seed + 1)

    print(f"Saved subset to {out_dir}")


if __name__ == "__main__":
    main()
