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

def non_max_suppression(
    bboxes,
    iou_threshold,
    prob_threshold,
    box_format="midpoint",
):
    """
    Apply Non-Maximum Suppression to a list of bounding boxes.

    Each box is a list: [class_idx, prob, x, y, w, h].

    Args:
        bboxes:         list of boxes, each [class, prob, x, y, w, h].
        iou_threshold:  if two boxes of the same class have IoU above this,
                        the one with lower prob is discarded.
        prob_threshold: boxes with prob below this are filtered out before NMS.
        box_format:     "midpoint" (x_center, y_center, w, h) or "corners".

    Returns:
        List of boxes that survived NMS.
    """
    assert isinstance(bboxes, list)

    # filter out low-confidence boxes
    bboxes = [box for box in bboxes if box[1] > prob_threshold]

    # sort by probability, highest first
    bboxes = sorted(bboxes, key=lambda x: x[1], reverse=True)

    bboxes_after_nms = []

    # iteratively pick the highest, suppress overlapping same-class boxes
    while bboxes:
        chosen_box = bboxes.pop(0)

        bboxes = [
            box
            for box in bboxes
            if box[0] != chosen_box[0]
            or intersection_over_union(
                torch.tensor(chosen_box[2:]),
                torch.tensor(box[2:]),
                box_format=box_format,
            ) < iou_threshold
        ]

        bboxes_after_nms.append(chosen_box)

    return bboxes_after_nms

def convert_cellboxes(predictions, S=7, C=80, B=2):
    """
    Convert YOLO output tensor into normalized boxes.

    Input:
        predictions: tensor of shape (batch_size, S*S*(C+5*B))
                     or (batch_size, S, S, C+5*B)
        S: grid size (7 by default)
        C: number of classes (80 for COCO)
        B: number of boxes per cell (2 by default)

    Output:
        tensor of shape (batch_size, S*S, 6) where each row is:
        [class_idx, prob, x, y, w, h]
        with x, y, w, h normalized to (0, 1) over the full image.
    """
    predictions = predictions.to("cpu")
    batch_size = predictions.shape[0]
    predictions = predictions.reshape(batch_size, S, S, C + 5 * B)
    
    # extract the two boxes from each cell
    bboxes1 = predictions[..., C+1:C+5]    # box 1: (x, y, w, h) at indices C+1..C+4
    bboxes2 = predictions[..., C+6:C+10]   # box 2: (x, y, w, h) at indices C+6..C+9
    
    # pick the box with higher confidence per cell
    scores = torch.cat(
        (predictions[..., C].unsqueeze(0), predictions[..., C+5].unsqueeze(0)), dim=0
    )
    best_box = scores.argmax(0).unsqueeze(-1)
    best_boxes = bboxes1 * (1 - best_box) + best_box * bboxes2
    
    # x and y are stored as offsets inside the cell, so we shift by the cell index
    # and divide by S to get image-relative coordinates.
    cell_indices = torch.arange(S).repeat(batch_size, S, 1).unsqueeze(-1)
    x = 1 / S * (best_boxes[..., :1] + cell_indices)
    y = 1 / S * (best_boxes[..., 1:2] + cell_indices.permute(0, 2, 1, 3))

    # w and h are already image-relative in the target (w_norm = w_pixels / image_width),
    # so they pass through unchanged.
    w_h = best_boxes[..., 2:4]

    converted_bboxes = torch.cat((x, y, w_h), dim=-1)
    
    # predicted class is argmax of class probabilities
    predicted_class = predictions[..., :C].argmax(-1).unsqueeze(-1)
    
    # best confidence (from the chosen box)
    best_confidence = torch.max(
        predictions[..., C], predictions[..., C+5]
    ).unsqueeze(-1)
    
    # pack as [class, prob, x, y, w, h]
    converted_preds = torch.cat(
        (predicted_class, best_confidence, converted_bboxes), dim=-1
    )
    
    return converted_preds.reshape(batch_size, S * S, 6)

def cellboxes_to_boxes(out, S=7):
    """
    Convert tensor output from convert_cellboxes to a list of lists of boxes.

    Input:
        out: tensor of shape (batch_size, S*S*(C+5*B)) — the raw model output

    Returns:
        List of length batch_size. Each element is a list of S*S boxes.
        Each box is a Python list: [class_idx, prob, x, y, w, h].

    This is the format expected by non_max_suppression.
    """
    converted_pred = convert_cellboxes(out, S=S).reshape(out.shape[0], S * S, -1)
    converted_pred[..., 0] = converted_pred[..., 0].long()  # class as integer

    all_bboxes = []
    for ex_idx in range(out.shape[0]):
        bboxes = []
        for bbox_idx in range(S * S):
            bboxes.append([x.item() for x in converted_pred[ex_idx, bbox_idx, :]])
        all_bboxes.append(bboxes)

    return all_bboxes