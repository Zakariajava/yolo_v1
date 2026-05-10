"""
Mean Average Precision (mAP) calculation for object detection.

This is the standard metric for evaluating detection models. It captures both
how well boxes are localized (via IoU threshold) and how confident the model
is in its predictions (via the precision-recall curve).
"""

from collections import Counter

import torch

from .utils import intersection_over_union


def mean_average_precision(
    pred_boxes,
    true_boxes,
    iou_threshold=0.5,
    box_format="midpoint",
    num_classes=80,
):
    """
    Compute mean Average Precision over all classes.

    Args:
        pred_boxes:    list of [image_idx, class, prob, x, y, w, h]
                       — all model predictions across the entire test set.
        true_boxes:    list of [image_idx, class, 1.0, x, y, w, h]
                       — all ground truth boxes across the entire test set.
        iou_threshold: IoU threshold above which a prediction counts as TP (default 0.5).
        box_format:    "midpoint" or "corners" — passed to intersection_over_union.
        num_classes:   total number of classes (80 for COCO, 20 for VOC).

    Returns:
        Tensor scalar — the mAP value (between 0 and 1).
    """
    # Will accumulate one AP per class, then average.
    average_precisions = []

    # Used as numerical stabilizer to avoid divide-by-zero.
    epsilon = 1e-6

    # Loop over each class independently. AP is computed per class, then averaged.
    for c in range(num_classes):
        # Filter predictions and ground truths to only this class.
        detections = [box for box in pred_boxes if box[1] == c]
        ground_truths = [box for box in true_boxes if box[1] == c]

        # If this class has no ground truths in the test set, skip it.
        # Including it would either crash (division by zero) or unfairly count as 0.
        if len(ground_truths) == 0:
            continue

        # Count GTs per image. amount_bboxes[image_idx] = number of GT boxes in that image.
        # We use this to track which GTs have been "claimed" by predictions.
        amount_bboxes = Counter([gt[0] for gt in ground_truths])

        # Replace each count with a list of zeros.
        # amount_bboxes[image_idx] is now a tensor like tensor([0, 0, 0])
        # where each 0 represents an unclaimed GT in that image.
        # We flip it to 1 once a prediction claims that GT.
        for key, val in amount_bboxes.items():
            amount_bboxes[key] = torch.zeros(val)

        # Sort detections by confidence, highest first.
        # This is critical: we walk through predictions from "most confident"
        # to "least confident" to build the precision-recall curve.
        detections.sort(key=lambda x: x[2], reverse=True)

        # TP[i] = 1 if detection i is a true positive, 0 otherwise.
        # FP[i] = 1 if detection i is a false positive, 0 otherwise.
        TP = torch.zeros(len(detections))
        FP = torch.zeros(len(detections))

        total_true_bboxes = len(ground_truths)

        # Walk through each detection in confidence order.
        for detection_idx, detection in enumerate(detections):
            # Get all GTs that belong to the same image as this detection.
            ground_truth_img = [
                bbox for bbox in ground_truths if bbox[0] == detection[0]
            ]

            best_iou = 0
            best_gt_idx = 0

            # Find the GT in this image with the highest IoU with our detection.
            for idx, gt in enumerate(ground_truth_img):
                iou = intersection_over_union(
                    torch.tensor(detection[3:]),
                    torch.tensor(gt[3:]),
                    box_format=box_format,
                )

                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = idx

            # Decision: TP or FP?
            if best_iou > iou_threshold:
                # Could be a TP, but only if this GT hasn't been claimed yet.
                if amount_bboxes[detection[0]][best_gt_idx] == 0:
                    # Mark GT as claimed.
                    TP[detection_idx] = 1
                    amount_bboxes[detection[0]][best_gt_idx] = 1
                else:
                    # This GT was already claimed by a higher-confidence detection.
                    # Same object detected twice → second detection is a FP.
                    FP[detection_idx] = 1
            else:
                # No good GT match → FP.
                FP[detection_idx] = 1

        # Cumulative sums: TP_cumsum[i] = how many TPs in detections 0..i (inclusive).
        TP_cumsum = torch.cumsum(TP, dim=0)
        FP_cumsum = torch.cumsum(FP, dim=0)

        # Recall at each detection i: cumulative TPs / total GTs.
        recalls = TP_cumsum / (total_true_bboxes + epsilon)

        # Precision at each detection i: cumulative TPs / cumulative detections seen.
        precisions = TP_cumsum / (TP_cumsum + FP_cumsum + epsilon)

        # Add anchor points (recall=0, precision=1) at the start so the curve
        # starts at the y-axis. This is standard for AP integration.
        precisions = torch.cat((torch.tensor([1]), precisions))
        recalls = torch.cat((torch.tensor([0]), recalls))

        # AP = area under precision-recall curve, computed with trapezoidal rule.
        average_precisions.append(torch.trapz(precisions, recalls))

    # If no class had GTs (unlikely but possible), return 0.
    if len(average_precisions) == 0:
        return torch.tensor(0.0)

    # mAP = mean of per-class APs.
    return sum(average_precisions) / len(average_precisions)