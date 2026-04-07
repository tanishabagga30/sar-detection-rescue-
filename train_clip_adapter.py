import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List

import torch
import yaml
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoProcessor, CLIPModel


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class ImageTextJsonlDataset(Dataset):
    def __init__(self, root_dir: str):
        root = Path(root_dir)
        meta = root / "metadata.jsonl"
        if not meta.exists():
            raise FileNotFoundError(f"metadata.jsonl not found under {root_dir}")

        # images live in the original SARD dataset
        self.base_dir = Path("data/search-and-rescue-2")

        self.rows = [
            json.loads(line)
            for line in meta.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        row = self.rows[idx]
        image_path = self.base_dir / row["image"]

        if not image_path.exists():
            raise FileNotFoundError(f"Missing image: {image_path}")

        image = Image.open(image_path).convert("RGB")
        text = row["text"]
        return image, text


@dataclass
class BatchCollator:
    processor: AutoProcessor

    def __call__(self, batch):
        images, texts = zip(*batch)
        model_inputs = self.processor(text=list(texts), images=list(images), return_tensors="pt", padding=True, truncation=True)
        return model_inputs


def clip_loss(logits: torch.Tensor) -> torch.Tensor:
    labels = torch.arange(logits.size(0), device=logits.device)
    loss_i = nn.functional.cross_entropy(logits, labels)
    loss_t = nn.functional.cross_entropy(logits.t(), labels)
    return (loss_i + loss_t) / 2


def freeze_for_light_tuning(model: CLIPModel) -> None:
    for p in model.parameters():
        p.requires_grad = False

    # Unfreeze only lightweight parts + last transformer block on both towers.
    trainable_names = []

    for name, param in model.visual_projection.named_parameters():
        param.requires_grad = True
        trainable_names.append(f"visual_projection.{name}")

    for name, param in model.text_projection.named_parameters():
        param.requires_grad = True
        trainable_names.append(f"text_projection.{name}")

    for name, param in model.vision_model.encoder.layers[-1].named_parameters():
        param.requires_grad = True
        trainable_names.append(f"vision_model.encoder.layers[-1].{name}")

    for name, param in model.text_model.encoder.layers[-1].named_parameters():
        param.requires_grad = True
        trainable_names.append(f"text_model.encoder.layers[-1].{name}")

    print("Trainable parameter groups:")
    for n in trainable_names[:12]:
        print("  ", n)
    if len(trainable_names) > 12:
        print(f"  ... and {len(trainable_names) - 12} more")


def make_loader(root: str, processor: AutoProcessor, batch_size: int, num_workers: int, shuffle: bool) -> DataLoader:
    ds = ImageTextJsonlDataset(root)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=BatchCollator(processor),
    )


def run_epoch(model, loader, optimizer, device, scaler, train: bool, log_every: int):
    total = 0.0
    count = 0
    model.train(train)

    autocast_enabled = scaler is not None
    iterator = tqdm(loader, leave=False)
    for step, batch in enumerate(iterator, start=1):
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.set_grad_enabled(train):
            with torch.autocast(device_type=device.type, enabled=autocast_enabled):
                outputs = model(**batch)
                logits = outputs.logits_per_image
                loss = clip_loss(logits)

            if train:
                optimizer.zero_grad(set_to_none=True)
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

        total += loss.item()
        count += 1
        if step % log_every == 0:
            iterator.set_postfix(loss=f"{total / count:.4f}")

    return total / max(count, 1)


def train_stage(model, processor, stage_dir: str, cfg: dict, device: torch.device):
    train_dir = str(Path(stage_dir) / "train")
    val_dir = str(Path(stage_dir) / "val")

    train_loader = make_loader(
        train_dir,
        processor,
        cfg["training"]["batch_size"],
        cfg["training"]["num_workers"],
        True,
    )
    val_loader = make_loader(
        val_dir,
        processor,
        cfg["training"]["batch_size"],
        cfg["training"]["num_workers"],
        False,
    )

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        params,
        lr=float(cfg["training"]["lr"]),
        weight_decay=float(cfg["training"]["weight_decay"]),
    )

    scaler = None
    if cfg["training"].get("mixed_precision", True) and device.type == "cuda":
        scaler = torch.cuda.amp.GradScaler()

    best_val = math.inf
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, cfg["training"]["epochs"] + 1):
        train_loss = run_epoch(model, train_loader, optimizer, device, scaler, True, cfg["training"]["log_every"])
        val_loss = run_epoch(model, val_loader, optimizer, device, scaler, False, cfg["training"]["log_every"])
        print(f"epoch={epoch} train_loss={train_loss:.4f} val_loss={val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            model.save_pretrained(out_dir)
            processor.save_pretrained(out_dir)
            print(f"Saved best checkpoint to {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    set_seed(cfg["seed"])

    device_name = cfg["training"]["device"]
    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)

    processor = AutoProcessor.from_pretrained(cfg["model_name"])
    model = CLIPModel.from_pretrained(cfg["model_name"])
    freeze_for_light_tuning(model)
    model.to(device)

    if cfg["stage1"]["enabled"]:
        train_stage(model, processor, cfg["stage1"]["dataset_dir"], cfg, device)

    if cfg["stage2"]["enabled"]:
        print("Running stage 2 SAR adaptation...")
        train_stage(model, processor, cfg["stage2"]["dataset_dir"], cfg, device)


if __name__ == "__main__":
    main()
