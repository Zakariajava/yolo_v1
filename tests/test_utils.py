import torch
from src.utils import (
    intersection_over_union,
    non_max_suppression,
    convert_cellboxes,
    cellboxes_to_boxes,
)

def test_identical_boxes():
    # Two identical boxes should give IoU = 1.
    box = torch.tensor([[0.5, 0.5, 0.4, 0.4]])
    iou = intersection_over_union(box, box).item()
    assert abs(iou - 1.0) < 1e-4, f"Expected IoU ≈ 1.0, got {iou}"


def test_disjoint_boxes():
    # Two boxes that don't touch should give IoU = 0.
    a = torch.tensor([[0.1, 0.1, 0.1, 0.1]])
    b = torch.tensor([[0.9, 0.9, 0.1, 0.1]])
    iou = intersection_over_union(a, b).item()
    assert iou < 1e-4, f"Expected IoU ≈ 0, got {iou}"


def test_half_overlap():
    # Two equal-size boxes offset by half their width along x.
    # Intersection area is half of each box's area.
    # IoU = 0.5 * A / (A + A - 0.5 * A) = 0.5 / 1.5 = 1/3.
    a = torch.tensor([[0.5, 0.5, 0.4, 0.4]])
    b = torch.tensor([[0.7, 0.5, 0.4, 0.4]])
    iou = intersection_over_union(a, b).item()
    expected = 1 / 3
    assert abs(iou - expected) < 1e-4, f"Expected IoU ≈ {expected:.4f}, got {iou}"


def test_one_inside_other():
    # Small box fully inside a larger one.
    # Intersection = small box area. Union = large box area.
    # IoU = small_area / large_area.
    big = torch.tensor([[0.5, 0.5, 1.0, 1.0]])    # area = 1
    small = torch.tensor([[0.5, 0.5, 0.5, 0.5]])  # area = 0.25
    iou = intersection_over_union(small, big).item()
    expected = 0.25
    assert abs(iou - expected) < 1e-4, f"Expected IoU ≈ {expected}, got {iou}"


def test_batched_boxes():
    # Verify the function works on a batched tensor and returns the right shape.
    preds = torch.rand((4, 7, 7, 4))
    labels = torch.rand((4, 7, 7, 4))
    iou = intersection_over_union(preds, labels)
    assert iou.shape == (4, 7, 7, 1), f"Expected shape (4, 7, 7, 1), got {iou.shape}"


def test_corners_format():
    # Same boxes as test_identical_boxes but in corners format.
    # (x1=0.3, y1=0.3, x2=0.7, y2=0.7) is the same as midpoint (0.5, 0.5, 0.4, 0.4).
    box = torch.tensor([[0.3, 0.3, 0.7, 0.7]])
    iou = intersection_over_union(box, box, box_format="corners").item()
    assert abs(iou - 1.0) < 1e-4, f"Expected IoU ≈ 1.0, got {iou}"

# ============================================================
# Tests for non_max_suppression
# ============================================================

def test_nms_keeps_non_overlapping_boxes():
    """Boxes that don't overlap should all survive NMS, regardless of class."""
    boxes = [
        [0, 0.9, 0.1, 0.1, 0.1, 0.1],   # class 0, top-left corner
        [0, 0.8, 0.9, 0.9, 0.1, 0.1],   # class 0, bottom-right corner
        [1, 0.7, 0.5, 0.5, 0.1, 0.1],   # class 1, center
    ]
    result = non_max_suppression(boxes, iou_threshold=0.5, prob_threshold=0.4)
    assert len(result) == 3, f"Expected 3 boxes to survive, got {len(result)}"


def test_nms_removes_duplicate_same_class():
    """When two identical boxes have the same class, only the higher-prob one survives."""
    boxes = [
        [0, 0.9, 0.5, 0.5, 0.4, 0.4],   # higher prob
        [0, 0.6, 0.5, 0.5, 0.4, 0.4],   # lower prob, identical box, same class
    ]
    result = non_max_suppression(boxes, iou_threshold=0.5, prob_threshold=0.4)
    assert len(result) == 1, f"Expected 1 box to survive, got {len(result)}"
    assert result[0][1] == 0.9, "The higher-probability box should be kept"


def test_nms_keeps_overlapping_different_classes():
    """Two identical boxes of different classes should BOTH survive (NMS is per-class)."""
    boxes = [
        [0, 0.9, 0.5, 0.5, 0.4, 0.4],   # class 0
        [1, 0.8, 0.5, 0.5, 0.4, 0.4],   # class 1, same box
    ]
    result = non_max_suppression(boxes, iou_threshold=0.5, prob_threshold=0.4)
    assert len(result) == 2, f"Expected 2 boxes (different classes), got {len(result)}"


def test_nms_filters_low_probability_boxes():
    """Boxes below the probability threshold should be discarded before NMS."""
    boxes = [
        [0, 0.9, 0.1, 0.1, 0.1, 0.1],   # above threshold
        [0, 0.3, 0.5, 0.5, 0.1, 0.1],   # below threshold (0.3 < 0.4)
        [0, 0.2, 0.9, 0.9, 0.1, 0.1],   # below threshold
    ]
    result = non_max_suppression(boxes, iou_threshold=0.5, prob_threshold=0.4)
    assert len(result) == 1, f"Expected 1 box to survive, got {len(result)}"
    assert result[0][1] == 0.9


def test_nms_empty_input_returns_empty():
    """Empty input should return empty output without errors."""
    result = non_max_suppression([], iou_threshold=0.5, prob_threshold=0.4)
    assert result == []


# ============================================================
# Tests for convert_cellboxes
# ============================================================

def test_convert_cellboxes_output_shape():
    """Output shape should be (batch_size, S*S, 6)."""
    batch_size = 2
    S, C, B = 7, 80, 2
    predictions = torch.randn(batch_size, S * S * (C + 5 * B))
    result = convert_cellboxes(predictions, S=S, C=C, B=B)
    assert result.shape == (batch_size, S * S, 6), \
        f"Expected shape ({batch_size}, {S*S}, 6), got {result.shape}"


def test_convert_cellboxes_class_indices_in_range():
    """Predicted class indices should be valid (0 to C-1)."""
    batch_size = 2
    S, C, B = 7, 80, 2
    predictions = torch.randn(batch_size, S * S * (C + 5 * B))
    result = convert_cellboxes(predictions, S=S, C=C, B=B)
    classes = result[..., 0]
    assert classes.min() >= 0
    assert classes.max() < C


# ============================================================
# Tests for cellboxes_to_boxes
# ============================================================

def test_cellboxes_to_boxes_returns_python_lists():
    """Output should be a Python list of lists of lists (not tensors)."""
    batch_size = 2
    S, C, B = 7, 80, 2
    predictions = torch.randn(batch_size, S * S * (C + 5 * B))
    result = cellboxes_to_boxes(predictions, S=S)
    
    assert isinstance(result, list)
    assert isinstance(result[0], list)
    assert isinstance(result[0][0], list)
    # Each box should have exactly 6 elements
    assert len(result[0][0]) == 6


def test_cellboxes_to_boxes_correct_number_of_boxes():
    """Each image should produce exactly S*S boxes."""
    batch_size = 3
    S, C, B = 7, 80, 2
    predictions = torch.randn(batch_size, S * S * (C + 5 * B))
    result = cellboxes_to_boxes(predictions, S=S)
    
    assert len(result) == batch_size, f"Expected {batch_size} images, got {len(result)}"
    for image_boxes in result:
        assert len(image_boxes) == S * S, f"Expected {S*S} boxes per image, got {len(image_boxes)}"
        
if __name__ == "__main__":
    # IoU tests
    test_identical_boxes()
    test_disjoint_boxes()
    test_half_overlap()
    test_one_inside_other()
    test_batched_boxes()
    test_corners_format()
    
    # NMS tests
    test_nms_keeps_non_overlapping_boxes()
    test_nms_removes_duplicate_same_class()
    test_nms_keeps_overlapping_different_classes()
    test_nms_filters_low_probability_boxes()
    test_nms_empty_input_returns_empty()
    
    # convert_cellboxes tests
    test_convert_cellboxes_output_shape()
    test_convert_cellboxes_class_indices_in_range()
    
    # cellboxes_to_boxes tests
    test_cellboxes_to_boxes_returns_python_lists()
    test_cellboxes_to_boxes_correct_number_of_boxes()
    print("All tests passed.")