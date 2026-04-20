import torch


def intersection_over_union(boxes_preds, boxes_labels, box_format="midpoint"):
    """
    Compute IoU between two sets of bounding boxes.

    IoU = area_of_intersection / area_of_union.
    It measures how well two boxes overlap. 1.0 means identical, 0.0 means disjoint.

    Args:
        boxes_preds:  tensor of shape (..., 4) with predicted boxes.
        boxes_labels: tensor of shape (..., 4) with ground-truth boxes.
        box_format:   "midpoint" → boxes are (x_center, y_center, w, h)
                      "corners"  → boxes are (x1, y1, x2, y2)

    Returns:
        Tensor of shape (..., 1) — one IoU per pair of boxes.

    Note on slicing: we use [..., i:i+1] instead of [..., i] everywhere.
    Both select the same values, but i:i+1 keeps the last dimension alive
    (shape stays (..., 1) instead of collapsing to (...,)), which is what
    we need so broadcasting works correctly in the caller.
    """

    if box_format == "midpoint":
        # Convert (x_center, y_center, w, h) to (x1, y1, x2, y2).
        # The corners are easier to work with for intersection geometry.
        box1_x1 = boxes_preds[..., 0:1]  - boxes_preds[..., 2:3]  / 2
        box1_y1 = boxes_preds[..., 1:2]  - boxes_preds[..., 3:4]  / 2
        box1_x2 = boxes_preds[..., 0:1]  + boxes_preds[..., 2:3]  / 2
        box1_y2 = boxes_preds[..., 1:2]  + boxes_preds[..., 3:4]  / 2
        box2_x1 = boxes_labels[..., 0:1] - boxes_labels[..., 2:3] / 2
        box2_y1 = boxes_labels[..., 1:2] - boxes_labels[..., 3:4] / 2
        box2_x2 = boxes_labels[..., 0:1] + boxes_labels[..., 2:3] / 2
        box2_y2 = boxes_labels[..., 1:2] + boxes_labels[..., 3:4] / 2

    elif box_format == "corners":
        # Boxes already come as corners — just split the last dimension.
        box1_x1 = boxes_preds[..., 0:1]
        box1_y1 = boxes_preds[..., 1:2]
        box1_x2 = boxes_preds[..., 2:3]
        box1_y2 = boxes_preds[..., 3:4]
        box2_x1 = boxes_labels[..., 0:1]
        box2_y1 = boxes_labels[..., 1:2]
        box2_x2 = boxes_labels[..., 2:3]
        box2_y2 = boxes_labels[..., 3:4]

    # Intersection rectangle corners.
    # The rule: for the intersection to "start", BOTH boxes must have already started.
    # So the intersection's left edge is the right-most of the two left edges (max),
    # and its right edge is the left-most of the two right edges (min). Same idea for y.
    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)

    # Intersection area = width × height.
    # clamp(min=0) handles the "no overlap" case: if the boxes don't touch,
    # (x2 - x1) or (y2 - y1) would be negative. We floor them to 0 so the area
    # is 0 rather than a nonsensical negative number.
    intersection = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)

    # Each box's own area.
    # abs() is a safety net in case the input is malformed (e.g. x2 < x1).
    box1_area = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
    box2_area = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))

    # Union = sum of both areas minus the intersection.
    # We subtract once because the overlapping region was counted in both areas.
    # The 1e-6 epsilon prevents division by zero if both boxes have zero area.
    return intersection / (box1_area + box2_area - intersection + 1e-6)