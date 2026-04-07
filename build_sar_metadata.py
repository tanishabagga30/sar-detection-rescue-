import argparse
import json
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images_dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--default_text",
        default="aerial SAR scene",
        help="Used when no caption file is available.",
    )
    args = parser.parse_args()

    images_dir = Path(args.images_dir)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    files = sorted([p for p in images_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTS])
    with output.open("w", encoding="utf-8") as f:
        for img in files:
            rel = img.relative_to(output.parent)
            txt_file = img.with_suffix(".txt")
            if txt_file.exists():
                text = txt_file.read_text(encoding="utf-8").strip()
            else:
                text = args.default_text
            f.write(json.dumps({"image": str(rel), "text": text}, ensure_ascii=False) + "\n")

    print(f"Wrote {len(files)} rows to {output}")


if __name__ == "__main__":
    main()
