"""
Tests for mean_average_precision.

Each test sets up a specific scenario and checks that mAP returns the expected value.
We use small toy examples (1-2 classes, 1-3 images) so we can compute the expected
mAP by hand.

Box format: [image_idx, class, prob, x_center, y_center, w, h]
All coordinates in (0, 1) range (midpoint format).
"""

import pytest
import torch

from src.metrics import mean_average_precision


def test_perfect_predictions_give_map_one():
    """
    All predictions match GTs exactly with high confidence.
    Expected: mAP = 1.0 (perfect detector).
    """
    pred_boxes = [
        [0, 0, 0.9, 0.5, 0.5, 0.4, 0.4],   # image 0, class 0
        [1, 0, 0.9, 0.5, 0.5, 0.4, 0.4],   # image 1, class 0
    ]
    true_boxes = [
        [0, 0, 1.0, 0.5, 0.5, 0.4, 0.4],
        [1, 0, 1.0, 0.5, 0.5, 0.4, 0.4],
    ]

    map_value = mean_average_precision(
        pred_boxes, true_boxes,
        iou_threshold=0.5, num_classes=1,
    )
    assert abs(map_value.item() - 1.0) < 1e-4, \
        f"Expected mAP=1.0 for perfect predictions, got {map_value.item()}"


def test_no_predictions_give_map_zero():
    """
    Model predicts nothing, but ground truth exists.
    Expected: mAP = 0.0 (no detections at all).
    """
    pred_boxes = []
    true_boxes = [
        [0, 0, 1.0, 0.5, 0.5, 0.4, 0.4],
        [1, 0, 1.0, 0.5, 0.5, 0.4, 0.4],
    ]

    map_value = mean_average_precision(
        pred_boxes, true_boxes,
        iou_threshold=0.5, num_classes=1,
    )
    assert map_value.item() == 0.0, \
        f"Expected mAP=0 for empty predictions, got {map_value.item()}"


def test_completely_wrong_predictions_give_map_zero():
    """
    All predictions are at totally wrong locations (no IoU with any GT).
    Expected: mAP = 0.0 (all FPs).
    """
    pred_boxes = [
        [0, 0, 0.9, 0.1, 0.1, 0.1, 0.1],   # tiny box in corner
        [1, 0, 0.9, 0.1, 0.1, 0.1, 0.1],
    ]
    true_boxes = [
        [0, 0, 1.0, 0.5, 0.5, 0.4, 0.4],   # GT in center
        [1, 0, 1.0, 0.5, 0.5, 0.4, 0.4],
    ]

    map_value = mean_average_precision(
        pred_boxes, true_boxes,
        iou_threshold=0.5, num_classes=1,
    )
    assert map_value.item() == 0.0, \
        f"Expected mAP=0 for misaligned predictions, got {map_value.item()}"


def test_duplicate_prediction_counts_as_fp():
    """
    Two predictions perfectly match ONE ground truth.
    The higher-confidence prediction is TP; the second is FP (GT already claimed).
    
    Expected: precision_curve goes from 1.0 to 0.5 at recall=1.
    AP ≈ 1.0 * 1.0 = 1.0 (since we add anchor (0, 1) and the recall jumps to 1
    on the first TP, giving area = 1).
    
    Note: the duplicate doesn't reduce mAP because by the time it's considered,
    recall is already at 1.0 (the only GT was found by the first prediction).
    Adding more recall points beyond 1.0 doesn't change the integration.
    """
    pred_boxes = [
        [0, 0, 0.9, 0.5, 0.5, 0.4, 0.4],   # higher prob, TP
        [0, 0, 0.7, 0.5, 0.5, 0.4, 0.4],   # lower prob, duplicate → FP
    ]
    true_boxes = [
        [0, 0, 1.0, 0.5, 0.5, 0.4, 0.4],
    ]

    map_value = mean_average_precision(
        pred_boxes, true_boxes,
        iou_threshold=0.5, num_classes=1,
    )
    # AP should still be ~1.0 because the first prediction captures the only GT
    assert abs(map_value.item() - 1.0) < 1e-3, \
        f"Expected mAP=1.0 (first detection covers all GTs), got {map_value.item()}"


def test_low_iou_prediction_is_fp():
    """
    Prediction overlaps with GT but IoU < 0.5 (below threshold).
    Expected: prediction = FP, mAP = 0.
    """
    pred_boxes = [
        [0, 0, 0.9, 0.7, 0.5, 0.4, 0.4],   # offset to the right, IoU ≈ 0.25
    ]
    true_boxes = [
        [0, 0, 1.0, 0.5, 0.5, 0.4, 0.4],
    ]

    map_value = mean_average_precision(
        pred_boxes, true_boxes,
        iou_threshold=0.5, num_classes=1,
    )
    assert map_value.item() == 0.0, \
        f"Expected mAP=0 for low-IoU predictions, got {map_value.item()}"


def test_classes_are_averaged_correctly():
    """
    Two classes:
    - Class 0: perfect detection (AP = 1.0).
    - Class 1: completely wrong (AP = 0.0).
    Expected: mAP = (1.0 + 0.0) / 2 = 0.5.
    """
    pred_boxes = [
        [0, 0, 0.9, 0.5, 0.5, 0.4, 0.4],   # class 0: perfect
        [0, 1, 0.9, 0.1, 0.1, 0.1, 0.1],   # class 1: wrong
    ]
    true_boxes = [
        [0, 0, 1.0, 0.5, 0.5, 0.4, 0.4],
        [0, 1, 1.0, 0.5, 0.5, 0.4, 0.4],
    ]

    map_value = mean_average_precision(
        pred_boxes, true_boxes,
        iou_threshold=0.5, num_classes=2,
    )
    assert abs(map_value.item() - 0.5) < 1e-3, \
        f"Expected mAP=0.5 (avg of 1.0 and 0.0), got {map_value.item()}"


def test_class_with_no_ground_truths_is_skipped():
    """
    Class 1 has predictions but NO ground truths (model hallucinating).
    Class 0 has perfect detection.
    Expected: class 1 is skipped, mAP = AP_class_0 = 1.0.
    
    This is the correct behavior: AP is undefined for classes with no GT.
    """
    pred_boxes = [
        [0, 0, 0.9, 0.5, 0.5, 0.4, 0.4],   # class 0
        [0, 1, 0.9, 0.5, 0.5, 0.4, 0.4],   # class 1 (no GT for this class)
    ]
    true_boxes = [
        [0, 0, 1.0, 0.5, 0.5, 0.4, 0.4],   # only class 0 has GT
    ]

    map_value = mean_average_precision(
        pred_boxes, true_boxes,
        iou_threshold=0.5, num_classes=2,
    )
    # Class 1 is skipped, mAP = AP for class 0 only
    assert abs(map_value.item() - 1.0) < 1e-3, \
        f"Expected mAP=1.0 (class 1 skipped, class 0 perfect), got {map_value.item()}"


def test_partial_recall_gives_partial_ap():
    """
    Two GTs in same image. Model only detects ONE of them.
    Recall maxes at 0.5. Precision is 1.0 at that point.
    AP = area under curve = 0.5 * 1.0 = 0.5.
    """
    pred_boxes = [
        [0, 0, 0.9, 0.3, 0.3, 0.2, 0.2],   # matches GT 1
    ]
    true_boxes = [
        [0, 0, 1.0, 0.3, 0.3, 0.2, 0.2],   # GT 1 (matched)
        [0, 0, 1.0, 0.7, 0.7, 0.2, 0.2],   # GT 2 (missed)
    ]

    map_value = mean_average_precision(
        pred_boxes, true_boxes,
        iou_threshold=0.5, num_classes=1,
    )
    # AP should be 0.5 (recall caps at 0.5, precision is 1.0)
    assert abs(map_value.item() - 0.5) < 1e-3, \
        f"Expected mAP=0.5 (1 of 2 GTs detected), got {map_value.item()}"