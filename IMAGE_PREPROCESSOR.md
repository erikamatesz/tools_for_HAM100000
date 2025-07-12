# Image Preprocessor for HAM10000

Tooling to prepare the **HAM10000** skin‑lesion image dataset for deep‑learning experiments.

## Overview

`prepare_images.py` processes images from one or more source directories by applying centered padding to make them square without distortion, resizing them to a fixed size (default **224×224**), and saving the results to a destination directory while preserving the original folder structure.

This preprocessing enables feeding the images directly into deep learning models or data loaders expecting consistent square inputs, such as Keras `ImageDataGenerator`, MobileNetV2, EfficientNetB0, or custom CNN architectures.

## Features

* **Multi-source support:** Accepts multiple source folders in one run.
* **Aspect-ratio safe:** Adds centered padding instead of stretching images.
* **Threaded processing:** Uses multiple threads for faster image processing.
* **Folder structure preservation:** Keeps original subfolder names inside the output directory to avoid name collisions.

## Requirements

```
pillow
tqdm
```

Install with `pip install pillow tqdm` or `pip install -r requirements.txt`.

## Usage

```bash
python prepare_images.py \
    --src HAM10000_images_part_1 HAM10000_images_part_2 \
    --dst HAM10000_224 \
    --size 224 \
    --fill black \
    --workers 8
```

### Arguments

| Flag         | Description                                                             | Default    |
|--------------|-------------------------------------------------------------------------|------------|
| `--src`      | One or more source directories containing original 600×450 images.      | *Required* |
| `--dst`      | Destination directory for processed images (created if missing).        | *Required* |
| `--size`     | Final square image size after padding and resizing (pixels).            | `224`      |
| `--fill`     | Padding color (e.g., `black`, `white`, `#RRGGBB`).                      | `black`    |
| `--workers`  | Number of parallel threads for processing.                              | `8`        |
| `--override` | Overwrite existing files in the destination folder if set.              | *False*    |

## Example Directory Layout

```
project_root/
│
├─ HAM10000_images_part_1/         # original images
├─ HAM10000_images_part_2/
└─ HAM10000_224/                   # generated processed images
    ├─ HAM10000_images_part_1/
    └─ HAM10000_images_part_2/
```

## After Conversion

Point your `ImageDataGenerator` or any other data loader to the processed folder (`HAM10000_224`) with `target_size=(224, 224)` and start training your model.
