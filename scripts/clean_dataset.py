"""
Clean the COCO annotations file.

Input:  data/coco/annotations/instances_all.json
Output: data/coco/annotations/instances_all_clean.json

What this script does:
  - Removes metadata we don't need: info, licenses.
  - Keeps only essential fields per entry:
      images       → id, file_name, width, height
      annotations  → image_id, bbox, category_id
      categories   → id, name
  - Filters out crowd annotations (iscrowd=1), which are noisy and hard to train on.

Run once after downloading COCO. The cleaned file is ~7x smaller and loads much faster.
"""

import json
import os
import sys
from pathlib import Path

# Add project root to sys.path so we can import from src/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import DATA_DIR

# Paths for this specific script
INPUT_JSON = DATA_DIR / "annotations" / "instances_all.json"
OUTPUT_JSON = DATA_DIR / "annotations" / "instances_all_clean.json"


def clean_image(img):
    """Keep only the fields we need from an image entry."""
    return {
        "id": img["id"],
        "file_name": img["file_name"],
        "width": img["width"],
        "height": img["height"],
    }


def clean_annotation(ann):
    """Keep only the fields we need from an annotation."""
    return {
        "image_id": ann["image_id"],
        "bbox": ann["bbox"],
        "category_id": ann["category_id"],
    }


def clean_category(cat):
    """Keep only the fields we need from a category."""
    return {
        "id": cat["id"],
        "name": cat["name"],
    }


def main():
    print(f"Loading {INPUT_JSON}...")
    with open(INPUT_JSON, "r") as f:
        data = json.load(f)

    print(f"Original: {len(data['images'])} images, {len(data['annotations'])} annotations, {len(data['categories'])} categories")

    non_crowd = [a for a in data["annotations"] if a["iscrowd"] == 0]
    crowd_removed = len(data["annotations"]) - len(non_crowd)
    print(f"Removed {crowd_removed} crowd annotations ({len(non_crowd)} remain)")

    clean_data = {
        "images": [clean_image(img) for img in data["images"]],
        "annotations": [clean_annotation(ann) for ann in non_crowd],
        "categories": [clean_category(cat) for cat in data["categories"]],
    }

    print(f"Writing {OUTPUT_JSON}...")
    with open(OUTPUT_JSON, "w") as f:
        json.dump(clean_data, f)

    old_size = os.path.getsize(INPUT_JSON) / 1024 / 1024
    new_size = os.path.getsize(OUTPUT_JSON) / 1024 / 1024
    print(f"\nSize: {old_size:.1f} MB → {new_size:.1f} MB ({100 * new_size / old_size:.1f}% of original)")


if __name__ == "__main__":
    main()