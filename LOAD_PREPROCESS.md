# Dataset Loader & Preprocessor for HAM10000

Utility script to load and preprocess the **HAM10000** skin-lesion dataset for deep learning experiments using TensorFlow/Keras.

## Overview

`load_preprocess.py` reads metadata CSV and processed image folders, then prepares:

- Training and validation image generators with rescaling and simple augmentations.
- Preprocessed clinical features (age, sex, lesion localization) as numeric arrays.
- A summary of dataset splits, class indices, and steps per epoch.

This makes it easy to feed both images and clinical data into multi-input deep learning models.

## Features

* **Metadata-aware:** Reads image IDs and labels from CSV metadata.
* **Multi-folder support:** Maps images from multiple source subfolders.
* **Train/validation split:** Stratified split ensuring balanced classes.
* **Image augmentation:** Rescaling, random horizontal flips, and rotations.
* **Clinical data preprocessing:** Imputation, scaling, and one-hot encoding.
* **Console summary:** Prints dataset statistics and class mappings.

## Requirements

```
pandas
numpy
tensorflow
scikit-learn
```

Install with `pip install pandas numpy tensorflow scikit-learn`.

## Usage

```bash
python load_preprocess.py \
    --csv ham10000_metadata.csv \
    --images ham10000_224 \
    --size 224 \
    --batch 32
```

### Arguments

| Flag       | Description                                        | Default  |
|------------|--------------------------------------------------|----------|
| `--csv`    | Path to CSV metadata file containing labels       | *Required* |
| `--images` | Root directory containing processed image folders | *Required* |
| `--size`   | Target square image size (pixels)                  | `224`    |
| `--batch`  | Batch size for generators                           | `32`     |
| `--val_split` | Fraction of data used for validation (0–1)      | `0.2`    |

## Output

The script returns:

- `train_gen` and `val_gen`: image data generators ready for `model.fit()`.
- `train_clinical` and `val_clinical`: numpy arrays of preprocessed clinical features.
- `stats`: dictionary summarizing dataset sizes, steps, and class indices.

## Example Directory Layout

```
project_root/
│
├─ ham10000_metadata.csv             # metadata CSV with image ids and labels
├─ ham10000_224/                    # processed images (square, 224x224)
│   ├─ HAM10000_images_part_1/
│   └─ HAM10000_images_part_2/
```