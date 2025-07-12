"""
load_preprocess.py

Utility script to load and preprocess the ham10000 dataset using tensorflow/keras
ImageDataGenerator.

What it does and only what is needed:
1. read the csv metadata (image ids + labels).
2. map each image id to its correct location inside `ham10000_224` (two sub‑folders).
3. build training‑ and validation‑image generators with rescaling + simple
   augmentations and a user‑defined batch size.
4. preprocess clinical features (age, sex, localization).
5. print a concise summary: total images, split sizes, expected steps per epoch,
   and class indices.

Usage example:

python load_preprocess.py \
    --csv ham10000_metadata.csv \
    --images ham10000_224 \
    --size 224 \
    --batch 32

CLI arguments
-------------
--csv      path to the csv metadata file (required)
--images   root directory containing the `ham10000_images_part_1/2` folders (required)
--size     target square image size (default 224)
--batch    batch size (default 32)
--val_split  validation split fraction (default 0.2 → 80% train / 20% val)

"""

import argparse
from pathlib import Path
from math import ceil

import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def load_generators(csv_path: Path, images_root: Path, *, img_size=(224, 224), batch_size=32, val_split=0.2):
    """return (train_gen, val_gen, train_clinical, val_clinical, stats_dict)"""
    df = pd.read_csv(csv_path)

    def map_image_path(image_id: str):
        part1 = images_root / 'HAM10000_images_part_1' / f'{image_id}.jpg'
        if part1.exists():
            return f'HAM10000_images_part_1/{image_id}.jpg'
        part2 = images_root / 'HAM10000_images_part_2' / f'{image_id}.jpg'
        if part2.exists():
            return f'HAM10000_images_part_2/{image_id}.jpg'
        return None

    # map image id to relative path for tensorflow
    df['image_path'] = df['image_id'].apply(map_image_path)
    # drop rows where image is not found
    df = df.dropna(subset=['image_path'])

    labels = sorted(df['dx'].unique())

    # split dataframe into train and validation sets, stratifying by label for balance
    train_df, val_df = train_test_split(df, test_size=val_split, stratify=df['dx'], random_state=42)

    # create image data generator with rescaling and simple augmentations
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1.0 / 255,
        horizontal_flip=True,
        rotation_range=15,
    )

    # create training image generator from train_df
    train_gen = datagen.flow_from_dataframe(
        train_df,
        directory=str(images_root),
        x_col='image_path',
        y_col='dx',
        target_size=img_size,
        class_mode='categorical',
        classes=labels,
        batch_size=batch_size,
        shuffle=True,
        seed=42,
    )

    # create validation image generator from val_df
    val_gen = datagen.flow_from_dataframe(
        val_df,
        directory=str(images_root),
        x_col='image_path',
        y_col='dx',
        target_size=img_size,
        class_mode='categorical',
        classes=labels,
        batch_size=batch_size,
        shuffle=False,
        seed=42,
    )

    # preprocess clinical features (age, sex, localization)
    def preprocess_clinical_features(df_subset):
        df_subset = df_subset.copy()
        # fill missing age with mean of the subset
        df_subset['age'] = df_subset['age'].fillna(df_subset['age'].mean())
        # standard scale age
        scaler = StandardScaler()
        age_norm = scaler.fit_transform(df_subset[['age']])
        # one-hot encode sex, drop 'unknown' if present
        sex_ohe = pd.get_dummies(df_subset['sex'])
        if 'unknown' in sex_ohe.columns:
            sex_ohe = sex_ohe.drop(columns=['unknown'])
        # one-hot encode localization
        loc_ohe = pd.get_dummies(df_subset['localization'])
        # concatenate all clinical features as numpy array
        clinical_features = np.hstack([age_norm, sex_ohe.values, loc_ohe.values])
        return clinical_features

    # extract clinical features arrays for train and validation sets
    train_clinical = preprocess_clinical_features(train_df)
    val_clinical = preprocess_clinical_features(val_df)

    stats = {
        'total_images': len(df),
        'train_images': len(train_df),
        'val_images': len(val_df),
        'batch_size': batch_size,
        'train_steps': len(train_gen),
        'val_steps': len(val_gen),
        'num_classes': len(labels),
        'class_indices': train_gen.class_indices,
    }

    return train_gen, val_gen, train_clinical, val_clinical, stats


def parse_args():
    p = argparse.ArgumentParser(description="Load & preprocess ham10000 images + metadata.")
    p.add_argument('--csv', required=True, type=Path, help='csv metadata file')
    p.add_argument('--images', required=True, type=Path, help='Root folder with processed images')
    p.add_argument('--size', type=int, default=224, help='Target image size (square). default 224')
    p.add_argument('--batch', type=int, default=32, help='Batch size. default 32')
    p.add_argument('--val_split', type=float, default=0.2, help='Validation split fraction. default 0.2')
    return p.parse_args()


def main():
    args = parse_args()

    train_gen, val_gen, train_clinical, val_clinical, stats = load_generators(
        csv_path=args.csv,
        images_root=args.images,
        img_size=(args.size, args.size),
        batch_size=args.batch,
        val_split=args.val_split,
    )

    print("\n=== dataset summary ===")
    print(f"total images:           {stats['total_images']}")
    print(f"training images:        {stats['train_images']}")
    print(f"validation images:      {stats['val_images']}")
    print(f"batch size:             {stats['batch_size']}")
    print(f"training steps/epoch:   {stats['train_steps']}  (ceil)")
    print(f"validation steps/epoch: {stats['val_steps']}  (ceil)")
    print(f"number of classes:      {stats['num_classes']}")
    print(f"class indices:          {stats['class_indices']}")

    print("\ngenerators ready — pass them to model.fit(...)")


if __name__ == '__main__':
    main()
