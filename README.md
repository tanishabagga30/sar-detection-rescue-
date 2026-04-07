# SAR Visual Reasoning MVP

A lightweight implementation of the pipeline we discussed:

1. Start from a pretrained CLIP model.
2. Adapt it on a **small subset** of a remote-sensing caption dataset (RSICD by default).
3. Optionally continue adapting on your own SAR image-text pairs.
4. Run inference that produces:
   - top-matching SAR scene labels
   - an optional structured prompt for a local LLM (via Ollama) for visual reasoning

This project is designed for a normal laptop / desktop and avoids full end-to-end multimodal fine-tuning.

## What this repo does

- `prepare_rsicd_subset.py`
  - downloads RSICD from Hugging Face and writes a **small subset** to disk
- `train_clip_adapter.py`
  - fine-tunes only a lightweight subset of CLIP parameters
- `build_sar_metadata.py`
  - converts a local SAR image folder into `metadata.jsonl`
- `reason_over_image.py`
  - runs inference using the tuned model and optionally calls a local LLM through Ollama
- `configs/default.yaml`
  - training and dataset settings

## Expected SAR dataset format

For your custom SAR stage, create a folder like:

```text
my_sar_data/
  images/
    0001.jpg
    0002.jpg
  metadata.jsonl
```

Each line in `metadata.jsonl` should look like:

```json
{"image": "images/0001.jpg", "text": "collapsed structure near road, blocked access path, possible survivor zone"}
```

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python prepare_rsicd_subset.py --output_dir data/rsicd_small --train_size 1500 --val_size 300
python train_clip_adapter.py --config configs/default.yaml
python reason_over_image.py \
  --model_dir outputs/clip-rs-adapted \
  --image path/to/test.jpg \
  --labels "person near debris" "blocked road" "flooded region" "clear access path"
```

## Optional LLM reasoning with Ollama

Install Ollama locally and pull a small model, for example:

```bash
ollama pull llama3.2:3b
```

Then run:

```bash
python reason_over_image.py \
  --model_dir outputs/clip-rs-adapted \
  --image path/to/test.jpg \
  --labels "person near debris" "blocked road" "flooded region" "clear access path" \
  --ollama_model llama3.2:3b
```

The script will first retrieve the top visual matches with the tuned CLIP model and then ask the LLM to reason only over the structured retrieval output.


## Adapting to your exact SAR dataset

Use `adapt_sar_dataset.py` when your dataset is not already in `metadata.jsonl` form. It supports these common layouts:

- `flat_sidecar`: images with optional same-name `.txt` caption files
- `class_folders`: `images/<label_name>/*.jpg`
- `csv`: a CSV with image and text columns
- `coco`: COCO-style JSON with `images` and `annotations`
- `vqa`: VQA-style JSON or a simple JSON list of `{image, text}` rows

Examples:

```bash
# 1) Class folders
python adapt_sar_dataset.py   --format class_folders   --images_dir data/my_sar_data/images   --output data/my_sar_data/metadata.jsonl   --template "aerial SAR scene of {label}"

# 2) CSV
python adapt_sar_dataset.py   --format csv   --images_dir data/my_sar_data/images   --csv_path data/my_sar_data/labels.csv   --image_col filename   --text_col caption   --output data/my_sar_data/metadata.jsonl

# 3) COCO captions / annotations
python adapt_sar_dataset.py   --format coco   --images_dir data/my_sar_data/images   --json_path data/my_sar_data/annotations.json   --output data/my_sar_data/metadata.jsonl
```

Once you have `metadata.jsonl`, point `stage2.dataset_dir` at that folder and set `stage2.enabled: true` in `configs/default.yaml`.


## Exact setup for SARD (the 6-class drone Search And Rescue Dataset)

Since SARD is a YOLO-format detection dataset with six classes — Running, Walking, Laying down, Not defined, Seated, and Stands — the correct adaptation path is:

1. keep the small RSICD bridge stage,
2. convert the YOLO labels into short SAR-specific text captions,
3. run stage 2 on that converted SARD subset.

This is more practical than trying to fine-tune a large VLM end to end. Roboflow's SARD page lists roughly 1.3k train images, 0.3k valid images, and the six posture/activity classes, which is a manageable second-stage adaptation dataset.

### Expected SARD export layout

Export the dataset from Roboflow in **YOLOv8** or **YOLOv5 PyTorch** format so you have something like:

```text
SARD/
  data.yaml
  train/
    images/
    labels/
  valid/
    images/
    labels/
  test/
    images/
    labels/
```

### Convert SARD YOLO labels into image-text pairs

This converter generates `metadata.jsonl` with captions such as:
- `drone search and rescue image showing a laying down person.`
- `aerial SAR scene; possible distressed or injured posture present.`

Use a functional subset by default:

```bash
python prepare_sard_yolo.py   --dataset_dir /path/to/SARD   --output_dir data/sard_text   --train_limit 900   --val_limit 250
```

That keeps training light while still using enough data to be functional.

### Train on RSICD subset + SARD subset

```bash
python prepare_rsicd_subset.py --output_dir data/rsicd_small --train_size 1500 --val_size 300
python train_clip_adapter.py --config configs/sard.yaml
```

### Run SARD-style inference

```bash
python reason_sard_pose.py   --model_dir outputs/clip-sard-adapted   --image /path/to/example.jpg
```

### Why this works for SARD

SARD is not a free-form reasoning dataset. It is a posture/activity detection dataset. So in this repo the model is adapted to map aerial SAR images to language about:
- running
- walking
- laying down
- seated
- standing
- undefined posture

That gives you a lightweight visual-language backbone you can later extend with an LLM for higher-level SAR reasoning over posture patterns, counts, or temporal sequences, without trying to train a giant multimodal model from scratch.
