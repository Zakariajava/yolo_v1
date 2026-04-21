"""
Tests for COCODataset.

tests verify the full data pipeline — from loading the JSON, to building
the image/annotation lookups, to producing (image, target) pairs ready for training:

    - Dataset size and class mapping are correct.
    - The annotations lookup is built and queryable.
    - __getitem__ returns tensors with the expected shapes.
    - Image values are normalized to [0, 1] by ToTensor().
    - Target confidence slots are binary (0 or 1), with no double-assignment.
    - Bounding box coordinates stay within the valid [0, 1] range.
    - Class positions form valid one-hot vectors.
    - The second box slot in the target remains zero (YOLO v1 uses one box per cell).
    - Image files referenced in the JSON actually exist on disk.
    - Multiple samples at different indices can be loaded without errors.
"""

import pytest
import torch
from PIL import Image

from src.config import (
    ANNOTATIONS_FILE,
    IMAGE_SIZE,
    IMAGES_DIR,
    NUM_BOXES,
    NUM_CLASSES,
    SPLIT_SIZE,
)
from src.dataset import COCODataset


# Load the dataset once per test session (reuses the JSON parse across tests).
@pytest.fixture(scope="module")
def dataset():
    return COCODataset(annotations_file=ANNOTATIONS_FILE)


def test_dataset_is_not_empty(dataset):
    """The cleaned COCO JSON should contain 123k images."""
    assert len(dataset) > 100_000, f"Expected 100k+ images, got {len(dataset)}"


def test_coco_id_mapping_has_80_classes(dataset):
    """The COCO id -> dense index mapping must cover all 80 classes."""
    assert len(dataset.coco_id_to_class_idx) == NUM_CLASSES

    # Dense indices should cover 0 .. NUM_CLASSES-1 with no gaps.
    indices = sorted(dataset.coco_id_to_class_idx.values())
    assert indices == list(range(NUM_CLASSES))


def test_annotations_lookup_is_built(dataset):
    """annotations_by_image should map image ids to lists of annotations."""
    # Grab any image id and verify its entry is a list of dicts.
    any_image_id = dataset.images[0]["id"]
    anns = dataset.annotations_by_image.get(any_image_id, [])
    assert isinstance(anns, list)
    for ann in anns:
        assert "bbox" in ann
        assert "category_id" in ann


def test_getitem_returns_correct_shapes(dataset):
    """__getitem__ must return tensors with the expected shapes."""
    image, target = dataset[0]

    assert image.shape == (3, IMAGE_SIZE, IMAGE_SIZE), \
        f"Expected image shape (3, {IMAGE_SIZE}, {IMAGE_SIZE}), got {image.shape}"

    expected_target_shape = (SPLIT_SIZE, SPLIT_SIZE, NUM_CLASSES + 5 * NUM_BOXES)
    assert target.shape == expected_target_shape, \
        f"Expected target shape {expected_target_shape}, got {target.shape}"


def test_image_values_are_normalized(dataset):
    """ToTensor() should scale pixel values from 0-255 into 0-1."""
    image, _ = dataset[0]
    assert image.dtype == torch.float32
    assert image.min() >= 0.0
    assert image.max() <= 1.0


def test_target_has_object_confidence_at_most_one_per_cell(dataset):
    """
    For any image with annotations, the target's confidence slot (index 80 for COCO)
    should be either 0 or 1 — never a larger value (no double-assignment bug).
    """
    _, target = dataset[0]
    confidence_slice = target[..., NUM_CLASSES]   # shape (S, S)
    assert torch.all((confidence_slice == 0) | (confidence_slice == 1))


def test_target_has_valid_bbox_coordinates(dataset):
    """
    Whenever a cell has an object (conf=1), its box coordinates should be
    reasonable: x, y in [0, 1] (relative to the cell), w, h in [0, 1] (relative to image).
    """
    _, target = dataset[0]

    # Find cells where there's an object.
    conf = target[..., NUM_CLASSES]
    object_cells = torch.nonzero(conf == 1)

    for cell in object_cells:
        row, col = cell[0].item(), cell[1].item()
        x, y, w, h = target[row, col, NUM_CLASSES + 1:NUM_CLASSES + 5].tolist()

        assert 0 <= x <= 1, f"x={x} out of [0, 1] at cell ({row}, {col})"
        assert 0 <= y <= 1, f"y={y} out of [0, 1] at cell ({row}, {col})"
        assert 0 <= w <= 1, f"w={w} out of [0, 1] at cell ({row}, {col})"
        assert 0 <= h <= 1, f"h={h} out of [0, 1] at cell ({row}, {col})"


def test_target_has_one_hot_class(dataset):
    """
    In each cell with an object, exactly ONE of the first C positions should be 1,
    and the rest should be 0 (valid one-hot encoding).
    """
    _, target = dataset[0]

    conf = target[..., NUM_CLASSES]
    object_cells = torch.nonzero(conf == 1)

    for cell in object_cells:
        row, col = cell[0].item(), cell[1].item()
        class_probs = target[row, col, :NUM_CLASSES]

        ones = (class_probs == 1).sum().item()
        zeros = (class_probs == 0).sum().item()

        assert ones == 1, f"Expected exactly 1 one in class slot, got {ones}"
        assert zeros == NUM_CLASSES - 1, f"Expected {NUM_CLASSES - 1} zeros, got {zeros}"


def test_second_box_slot_is_always_zero(dataset):
    """
    The ground truth only contains one box per cell.
    The second box slot (positions C+5 through C+10) should always be zero,
    even in cells with objects.
    """
    _, target = dataset[0]
    second_box_slice = target[..., NUM_CLASSES + 5:NUM_CLASSES + 10]
    assert torch.all(second_box_slice == 0)


def test_image_file_exists_on_disk(dataset):
    """Sanity check: the file_name in the JSON should actually exist in images/."""
    img_info = dataset.images[0]
    image_path = IMAGES_DIR / img_info["file_name"]
    assert image_path.exists(), f"Image file not found: {image_path}"


def test_can_load_multiple_samples(dataset):
    """Load a handful of samples to make sure __getitem__ is stable across indices."""
    for idx in [0, 100, 1000, 10000]:
        image, target = dataset[idx]
        assert image.shape == (3, IMAGE_SIZE, IMAGE_SIZE)
        assert target.shape == (SPLIT_SIZE, SPLIT_SIZE, NUM_CLASSES + 5 * NUM_BOXES)