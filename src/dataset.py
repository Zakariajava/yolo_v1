"""
COCO dataset loader for YOLOv1 training.

Converts COCO annotations (absolute pixel bounding boxes) into YOLO's grid format:
a tensor of shape (S, S, C + 5*B) where each grid cell encodes class probabilities,
object confidence, and bounding box coordinates.
"""

import json
import os
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from .config import (
    IMAGE_SIZE,
    IMAGES_DIR,
    NUM_BOXES,
    NUM_CLASSES,
    SPLIT_SIZE
)


class COCODataset(Dataset):
    """
    COCO dataset in YOLOv1 grid format.

    Loads images and their annotations from a cleaned COCO JSON file
    (see scripts/clean_dataset.py) and yields (image, target) pairs
    ready for YOLOv1 training.

    Args:
        annotations_file: path to the COCO-format JSON file.
        images_dir:       path to the folder containing the .jpg files.
        S: grid size.
        B: boxes predicted per cell.
        C: number of classes.
        image_size: side length of resized images.
    """

    def __init__(
        self,
        annotations_file,
        images_dir=IMAGES_DIR,
        S=SPLIT_SIZE,
        B=NUM_BOXES,
        C=NUM_CLASSES,
        image_size=IMAGE_SIZE,
    ):
        self.images_dir = Path(images_dir)
        self.S = S
        self.B = B
        self.C = C
        self.image_size = image_size

        # Load the full annotations file.
        with open(annotations_file, "r") as f:
            data = json.load(f)

        # Keep the image list as the main iterable; __len__ is len(images).
        self.images = data["images"]

        # Build a lookup: image_id -> list of annotations for that image.
        # This avoids scanning all 900k annotations on every __getitem__ call.
        self.annotations_by_image = {}
        for ann in data["annotations"]:
            self.annotations_by_image.setdefault(ann["image_id"], []).append(ann)

        # COCO category ids are not 0..79 — they go from 1 to 90 with gaps.
        # Build a mapping to dense indices so we can use them as tensor positions.
        self.coco_id_to_class_idx = {
            cat["id"]: idx for idx, cat in enumerate(data["categories"])
        }

        # Image preprocessing: resize + to tensor (values scaled to [0, 1]).
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        """
        Load image idx and build its YOLO target tensor.

        Returns:
            image:  tensor of shape (3, image_size, image_size), values in [0, 1].
            target: tensor of shape (S, S, C + 5*B) with the grid-format labels.
        """
        img_info = self.images[idx]
        image_id = img_info["id"]
        original_width = img_info["width"]
        original_height = img_info["height"]

        # Load image and apply transforms (resize + tensor conversion).
        image_path = self.images_dir / img_info["file_name"]
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)

        # Build the target grid, initialized to zeros.
        # Layout per cell: [C class probs | conf1, x, y, w, h | conf2, x, y, w, h]
        target = torch.zeros((self.S, self.S, self.C + 5 * self.B))

        # Iterate over all annotations for this image.
        for ann in self.annotations_by_image.get(image_id, []):
            class_idx = self.coco_id_to_class_idx[ann["category_id"]]

            # COCO bbox is [x_top_left, y_top_left, width, height] in absolute pixels.
            x_tl, y_tl, w_px, h_px = ann["bbox"]

            # Convert top-left format to center format, still in pixels.
            x_center_px = x_tl + w_px / 2
            y_center_px = y_tl + h_px / 2

            # Normalize to [0, 1] using the ORIGINAL image dimensions.
            # These values stay valid after resizing because they're relative.
            x_norm = x_center_px / original_width
            y_norm = y_center_px / original_height
            w_norm = w_px / original_width
            h_norm = h_px / original_height

            # Find which grid cell the object's center falls into.
            # For a 7x7 grid, x_norm=0.37 falls in column int(0.37 * 7) = 2.
            cell_col = int(self.S * x_norm)
            cell_row = int(self.S * y_norm)

            # Guard against the edge case where x_norm or y_norm equals exactly 1.0
            # (center on the far edge), which would give cell_col == S (out of bounds).
            cell_col = min(cell_col, self.S - 1)
            cell_row = min(cell_row, self.S - 1)

            # Skip if this cell already has an object. YOLOv1 assumes one object
            # per cell — if two centers fall into the same cell, only one is kept.
            if target[cell_row, cell_col, self.C] == 1:
                continue

            # Convert x, y to coordinates RELATIVE to the cell (values in [0, 1]).
            # Example: if S=7 and x_norm=0.37, cell_col=2, x_in_cell = 7*0.37 - 2 = 0.59
            x_in_cell = self.S * x_norm - cell_col
            y_in_cell = self.S * y_norm - cell_row

            # Width and height stay relative to the whole image (not the cell).
            # Objects can span multiple cells, so cell-relative sizes wouldn't fit in [0, 1].
            box_coords = torch.tensor([x_in_cell, y_in_cell, w_norm, h_norm])

            # Fill the target cell:
            #   [0:C]          one-hot class vector
            #   [C]            object confidence = 1
            #   [C+1 : C+5]    box coordinates
            # Positions [C+5 : C+10] stay at zero — the second box slot is unused
            # in the ground truth (only predictions have two candidate boxes).
            target[cell_row, cell_col, class_idx] = 1
            target[cell_row, cell_col, self.C] = 1
            target[cell_row, cell_col, self.C + 1:self.C + 5] = box_coords

        return image, target