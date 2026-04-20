"""
Consolidate COCO train2017 and val2017 into a single images folder and annotation file.

Input:
  data/coco/train2017/         (118k images)
  data/coco/val2017/           (5k images)
  data/coco/annotations/instances_train2017.json
  data/coco/annotations/instances_val2017.json

Output:
  data/coco/images/                             (all 123k images merged)
  data/coco/annotations/instances_all.json      (combined annotation file)

What this script does:
  - Moves all images from train2017/ and val2017/ into a single images/ folder.
  - Merges the 'images' and 'annotations' lists from both JSONs into one.

COCO filenames are globally unique, so no collisions happen when merging.
The original train2017/ and val2017/ folders end up empty.
Run once after downloading and extracting COCO.
"""

import json
import os
import shutil
import sys
from pathlib import Path

# Add project root to sys.path so we can import from src/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import DATA_DIR

# Paths for this specific script
TRAIN_IMAGES_DIR = DATA_DIR / "train2017"
VAL_IMAGES_DIR = DATA_DIR / "val2017"
ALL_IMAGES_DIR = DATA_DIR / "images"

TRAIN_JSON = DATA_DIR / "annotations" / "instances_train2017.json"
VAL_JSON = DATA_DIR / "annotations" / "instances_val2017.json"
COMBINED_JSON = DATA_DIR / "annotations" / "instances_all.json"


def move_images(src_dir, dst_dir):
    """Move all .jpg files from src_dir to dst_dir."""
    files = [f for f in os.listdir(src_dir) if f.endswith(".jpg")]
    print(f"Moving {len(files)} images from {src_dir} to {dst_dir}...")
    for i, filename in enumerate(files):
        shutil.move(os.path.join(src_dir, filename), os.path.join(dst_dir, filename))
        if (i + 1) % 10000 == 0:
            print(f"  {i + 1}/{len(files)} moved")
    print(f"  Done: {len(files)} files moved.\n")


def combine_annotations(train_path, val_path, out_path):
    """Merge train and val JSONs into a single COCO-format file."""
    print(f"Loading {train_path}...")
    with open(train_path, "r") as f:
        train_data = json.load(f)

    print(f"Loading {val_path}...")
    with open(val_path, "r") as f:
        val_data = json.load(f)

    combined = {
        "info": train_data.get("info", {}),
        "licenses": train_data.get("licenses", []),
        "images": train_data["images"] + val_data["images"],
        "annotations": train_data["annotations"] + val_data["annotations"],
        "categories": train_data["categories"],
    }

    print(f"Writing combined JSON to {out_path}...")
    with open(out_path, "w") as f:
        json.dump(combined, f)

    print(f"  Total images: {len(combined['images'])}")
    print(f"  Total annotations: {len(combined['annotations'])}")
    print(f"  Total categories: {len(combined['categories'])}\n")


def main():
    os.makedirs(ALL_IMAGES_DIR, exist_ok=True)
    move_images(TRAIN_IMAGES_DIR, ALL_IMAGES_DIR)
    move_images(VAL_IMAGES_DIR, ALL_IMAGES_DIR)
    combine_annotations(TRAIN_JSON, VAL_JSON, COMBINED_JSON)

    print("Done.")
    print(f"All images now in: {ALL_IMAGES_DIR}")
    print(f"Combined annotations: {COMBINED_JSON}")


if __name__ == "__main__":
    main()