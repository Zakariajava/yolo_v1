import torch
from utils import intersection_over_union


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


if __name__ == "__main__":
    test_identical_boxes()
    test_disjoint_boxes()
    test_half_overlap()
    test_one_inside_other()
    test_batched_boxes()
    test_corners_format()
    print("All tests passed.")