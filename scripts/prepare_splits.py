"""
Split the cleaned COCO annotations into train / val / test JSONs.

Input:
    data/coco/annotations/instances_all_clean.json

Output:
    data/coco/annotations/instances_train.json  (90% of images)
    data/coco/annotations/instances_val.json    (5% of images)
    data/coco/annotations/instances_test.json   (5% of images)

The split is deterministic (fixed seed) so re-running produces the same result.
Each output file keeps the same COCO schema as the input, containing only
the images and annotations that belong to that split.

Run once after cleaning the dataset:
    python scripts/prepare_splits.py
"""

import json
import sys
from pathlib import Path

from sklearn.model_selection import train_test_split

# Add project root to sys.path so we can import from src/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import DATA_DIR

# Paths
INPUT_JSON = DATA_DIR / "annotations" / "instances_all_clean.json"
TRAIN_JSON = DATA_DIR / "annotations" / "instances_train.json"
VAL_JSON = DATA_DIR / "annotations" / "instances_val.json"
TEST_JSON = DATA_DIR / "annotations" / "instances_test.json"

# Split proportions (must sum to 1.0)
TRAIN_FRAC = 0.90
VAL_FRAC = 0.05
TEST_FRAC = 0.05

# seed for reproducibility
SEED = 42


def save_split(images, all_annotations, categories, out_path):
    """Build and save a COCO-format JSON for one split."""
    # Keep only the annotations whose image_id belongs to this split.
    image_ids = {img["id"] for img in images}
    anns = [a for a in all_annotations if a["image_id"] in image_ids]

    split_data = {
        "images": images,
        "annotations": anns,
        "categories": categories,
    }

    with open(out_path, "w") as f:
        json.dump(split_data, f)

    print(f"  {out_path.name}: {len(images)} images, {len(anns)} annotations")


def main():
    print(f"Loading {INPUT_JSON}...")
    with open(INPUT_JSON, "r") as f:
        data = json.load(f)

    all_images = data["images"]
    all_annotations = data["annotations"]
    categories = data["categories"]

    print(f"Total: {len(all_images)} images, {len(all_annotations)} annotations")

    # First split: separate test (5%) from the rest (90%).
    train_val_images, test_images = train_test_split(
        all_images,
        test_size=TEST_FRAC,
        random_state=SEED,
    )

    # Second split: from the remaining 95%, take val (5% of 90% = 4,5% of total).
    val_relative = VAL_FRAC / (TRAIN_FRAC + VAL_FRAC)
    train_images, val_images = train_test_split(
        train_val_images,
        test_size=val_relative,
        random_state=SEED,
    )

    print(f"\nSplit sizes:")
    print(f"  train: {len(train_images)} ({100 * len(train_images) / len(all_images):.1f}%)")
    print(f"  val:   {len(val_images)} ({100 * len(val_images) / len(all_images):.1f}%)")
    print(f"  test:  {len(test_images)} ({100 * len(test_images) / len(all_images):.1f}%)")

    print(f"\nWriting split files:")
    save_split(train_images, all_annotations, categories, TRAIN_JSON)
    save_split(val_images, all_annotations, categories, VAL_JSON)
    save_split(test_images, all_annotations, categories, TEST_JSON)

    print("\nDone.")


if __name__ == "__main__":
    main()