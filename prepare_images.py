"""
prepare_images.py

Processes HAM10000 dataset images spread across multiple source directories.
(e.g., `HAM10000_images_part_1`, `HAM10000_images_part_2`).

For each image:
1. applies centered padding to make it square, preserving aspect ratio.
2. resizes to a fixed size (default: 224×224).
3. saves to the destination directory, preserving the source subfolder structure.

Usage example:

python prepare_images.py \
    --src HAM10000_images_part_1 HAM10000_images_part_2 \
    --dst HAM10000_224 \
    --size 224 \
    --fill black

CLI arguments
-------------

--src       One or more source directories containing original 600×450 images.  
--dst       Destination directory for processed images (created if missing).  
--size      Final square image size after padding and resizing (default: 224).  
--fill      Padding color (e.g., "black", "white", "#RRGGBB"). Default is black.  
--workers   Number of parallel threads for processing. Default is 8.  
--override  If set, overwrite existing files in the destination folder.

"""

import argparse
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path

from PIL import Image, ImageOps
from tqdm import tqdm


def pad_and_resize(img: Image.Image, target_size: int, fill_color: str) -> Image.Image:
    """Applies centered padding and resizes the image to square format."""
    w, h = img.size
    desired = max(w, h)
    delta_w = desired - w
    delta_h = desired - h
    padding = (
        delta_w // 2,
        delta_h // 2,
        delta_w - delta_w // 2,
        delta_h - delta_h // 2,
    )
    padded = ImageOps.expand(img, padding, fill=fill_color)
    return padded.resize((target_size, target_size), Image.Resampling.LANCZOS)


def process_one(target_size: int, fill_color: str, override: bool, pair: tuple[Path, Path], dst_root: Path):
    """Processes a single image.

    pair = (src_root, img_path)
    """
    src_root, path = pair
    rel = path.relative_to(src_root)
    safe_prefix = src_root.name
    dst_path = dst_root / safe_prefix / rel
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    if dst_path.exists() and not override:
        return  # skip if already exists

    with Image.open(path) as img:
        img = img.convert("RGB")  # ensure 3 channels
        out = pad_and_resize(img, target_size, fill_color)
        out.save(dst_path, format="JPEG", quality=95, optimize=True)


def gather_images(src_root: Path):
    """Finds all valid image files in a given source directory."""
    exts = {".jpg", ".jpeg", ".png"}
    return [p for p in src_root.rglob("*") if p.suffix.lower() in exts]


def main():
    parser = argparse.ArgumentParser(description="Apply padding and resizing to HAM10000 images across multiple directories.")
    parser.add_argument("--src", required=True, nargs="+", help="Source directories containing 600x450 images.")
    parser.add_argument("--dst", required=True, help="Output directory for processed images.")
    parser.add_argument("--size", type=int, default=224, help="Final square image size.")
    parser.add_argument("--fill", default="black", help="Padding fill color.")
    parser.add_argument("--workers", type=int, default=8, help="Number of parallel threads.")
    parser.add_argument("--override", action="store_true", help="Overwrite existing files.")

    args = parser.parse_args()
    dst_root = Path(args.dst)
    dst_root.mkdir(parents=True, exist_ok=True)

    tasks = []
    for src_dir in args.src:
        src_root = Path(src_dir)
        imgs = gather_images(src_root)
        tasks.extend([(src_root, p) for p in imgs])

    if not tasks:
        print("No images found in the provided source directories.")
        return

    worker_fn = partial(
        process_one,
        args.size,
        args.fill,
        args.override,
        dst_root=dst_root,
    )

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(tqdm(pool.map(worker_fn, tasks), total=len(tasks), desc="Processing"))

    print(f"\n✔ Conversion complete. {len(tasks)} files written to {dst_root}")


if __name__ == "__main__":
    main()
