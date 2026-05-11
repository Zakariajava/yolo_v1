"""
Evaluate a trained YOLOv1 model on COCO test set using mAP.

Loads a checkpoint, runs the model on the entire test set, applies NMS,
and computes mAP@0.5 against the ground truth.

Usage:
    python scripts/evaluate.py --checkpoint v2_run/best_v2.pth
    python scripts/evaluate.py --checkpoint v1_run/best_v1.pth --split val
    python scripts/evaluate.py --checkpoint v3_run/best_v3.pth --max-samples 500

Outputs:
    - Overall mAP@0.5
    - Per-class AP (top-10 best and worst)
    - Total inference time
"""

import argparse
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import (
    ANNOTATIONS_FILE,
    BATCH_SIZE,
    NUM_BOXES,
    NUM_CLASSES,
    NUM_WORKERS,
    SPLIT_SIZE,
    TEST_ANNOTATIONS_FILE,
    VAL_ANNOTATIONS_FILE,
)
from src.dataset import COCODataset
from src.metrics import mean_average_precision
from src.model import Yolov1
from src.utils import cellboxes_to_boxes, non_max_suppression


def load_model(checkpoint_path, device):
    """
    Load Yolov1 model and weights from a checkpoint.

    Handles both 'clean' checkpoints (state_dict only) and 'full' checkpoints
    (with optimizer, scaler, epoch, etc.).
    """
    print(f"Loading model from: {checkpoint_path}")

    model = Yolov1(
        split_size=SPLIT_SIZE,
        num_boxes=NUM_BOXES,
        num_classes=NUM_CLASSES,
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"  Loaded full checkpoint (epoch {checkpoint.get('epoch', '?')}, "
              f"val_loss {checkpoint.get('val_loss', '?')})")
    else:
        model.load_state_dict(checkpoint)
        print(f"  Loaded clean weights")

    model.eval()
    return model


def extract_gt_boxes_from_target(target, image_idx, S=SPLIT_SIZE, C=NUM_CLASSES):
    """
    Extract ground truth boxes from a target tensor.

    Args:
        target:    tensor (S, S, C + 5*B) — single-image target.
        image_idx: index assigned to this image (for mAP bookkeeping).

    Returns:
        list of boxes [image_idx, class, 1.0, x, y, w, h] in image-relative coordinates.
    """
    boxes = []
    # Find cells where there is an object (confidence slot at index C is 1).
    conf_mask = target[..., C] == 1
    object_cells = torch.nonzero(conf_mask)

    for cell in object_cells:
        row, col = cell[0].item(), cell[1].item()
        class_idx = int(torch.argmax(target[row, col, :C]).item())

        # Coords stored as (x_in_cell, y_in_cell, w_norm, h_norm)
        x_in_cell, y_in_cell, w_norm, h_norm = target[row, col, C + 1:C + 5].tolist()

        # Convert cell-relative to image-relative
        x_global = (col + x_in_cell) / S
        y_global = (row + y_in_cell) / S

        boxes.append([
            image_idx,
            class_idx,
            1.0,           # prob for GT is 1.0
            x_global,
            y_global,
            w_norm,
            h_norm,
        ])

    return boxes


def get_predictions_and_gt(
    model,
    loader,
    device,
    iou_threshold=0.5,
    prob_threshold=0.2,
    max_samples=None,
):
    """
    Run inference over the full loader and collect predictions + ground truth.

    Returns:
        all_pred_boxes: list of [image_idx, class, prob, x, y, w, h]
        all_true_boxes: list of [image_idx, class, 1.0, x, y, w, h]
    """
    all_pred_boxes = []
    all_true_boxes = []

    model.eval()
    image_idx = 0  # global counter for image identification

    total_images = len(loader.dataset) if max_samples is None else min(max_samples, len(loader.dataset))
    print(f"Processing {total_images} images in batches of {loader.batch_size}...")

    start_time = time.time()

    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(loader):
            # Stop early if max_samples reached
            if max_samples is not None and image_idx >= max_samples:
                break

            images = images.to(device)
            predictions = model(images)

            # Convert raw predictions to list of cell boxes per image
            bboxes = cellboxes_to_boxes(predictions)

            # For each image in the batch
            for i in range(len(bboxes)):
                if max_samples is not None and image_idx >= max_samples:
                    break

                # NMS for this image
                nms_boxes = non_max_suppression(
                    bboxes[i],
                    iou_threshold=iou_threshold,
                    prob_threshold=prob_threshold,
                    box_format="midpoint",
                )

                # Add image_idx prefix and store
                for nms_box in nms_boxes:
                    # nms_box format: [class, prob, x, y, w, h]
                    # We need: [image_idx, class, prob, x, y, w, h]
                    all_pred_boxes.append([image_idx] + nms_box)

                # Extract ground truth from this image's target
                gt_boxes = extract_gt_boxes_from_target(targets[i], image_idx)
                all_true_boxes.extend(gt_boxes)

                image_idx += 1

            # Progress
            if (batch_idx + 1) % 10 == 0:
                elapsed = time.time() - start_time
                eta = elapsed / (image_idx) * (total_images - image_idx)
                print(f"  Processed {image_idx}/{total_images} images "
                      f"({elapsed:.1f}s elapsed, ~{eta:.1f}s remaining)")

    total_time = time.time() - start_time
    print(f"\nInference complete: {image_idx} images in {total_time:.1f}s "
          f"({image_idx/total_time:.1f} img/s)")
    print(f"  Total predictions after NMS: {len(all_pred_boxes)}")
    print(f"  Total ground truths: {len(all_true_boxes)}")

    return all_pred_boxes, all_true_boxes


def compute_per_class_ap(pred_boxes, true_boxes, num_classes=NUM_CLASSES, iou_threshold=0.5):
    """
    Compute AP for each class individually, returning a dict {class_idx: ap}.

    Used to identify which classes the model detects best/worst.
    """
    aps = {}
    for c in range(num_classes):
        # Filter to this class
        preds_c = [box for box in pred_boxes if box[1] == c]
        gts_c = [box for box in true_boxes if box[1] == c]

        if len(gts_c) == 0:
            continue  # AP undefined for classes with no GT

        # Compute mAP with only this class (it will return AP for class c)
        ap = mean_average_precision(
            preds_c, gts_c,
            iou_threshold=iou_threshold,
            num_classes=num_classes,
        )
        aps[c] = ap.item()

    return aps


def main():
    parser = argparse.ArgumentParser(description="Evaluate YOLOv1 model with mAP")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to the trained model checkpoint",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="Which dataset split to evaluate on (default: test)",
    )
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=0.5,
        help="IoU threshold for TP/FP decision (default: 0.5)",
    )
    parser.add_argument(
        "--prob-threshold",
        type=float,
        default=0.2,
        help="NMS probability threshold — boxes with prob below are discarded (default: 0.2)",
    )
    parser.add_argument(
        "--nms-iou-threshold",
        type=float,
        default=0.5,
        help="NMS IoU threshold for duplicate removal (default: 0.5)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help=f"Batch size (default: {BATCH_SIZE})",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Limit evaluation to N samples (for quick testing)",
    )
    args = parser.parse_args()

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # Pick the right annotations file
    if args.split == "test":
        ann_file = TEST_ANNOTATIONS_FILE
    elif args.split == "val":
        ann_file = VAL_ANNOTATIONS_FILE
    else:
        from src.config import TRAIN_ANNOTATIONS_FILE
        ann_file = TRAIN_ANNOTATIONS_FILE

    # Load model
    model = load_model(args.checkpoint, device)

    # Load dataset
    print(f"\nLoading {args.split} dataset from: {ann_file}")
    dataset = COCODataset(annotations_file=str(ann_file))
    print(f"  {len(dataset)} images in {args.split} split")

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,  # IMPORTANT: deterministic image_idx
        num_workers=NUM_WORKERS if device.type == "cuda" else 0,
        pin_memory=(device.type == "cuda"),
    )

    # Get predictions and ground truth
    print(f"\nRunning inference (prob_threshold={args.prob_threshold}, "
          f"nms_iou={args.nms_iou_threshold})...")
    pred_boxes, true_boxes = get_predictions_and_gt(
        model,
        loader,
        device,
        iou_threshold=args.nms_iou_threshold,
        prob_threshold=args.prob_threshold,
        max_samples=args.max_samples,
    )

    # Compute overall mAP
    print(f"\nComputing mAP@{args.iou_threshold}...")
    map_value = mean_average_precision(
        pred_boxes,
        true_boxes,
        iou_threshold=args.iou_threshold,
        num_classes=NUM_CLASSES,
    )

    print(f"\n{'='*60}")
    print(f"RESULT: mAP@{args.iou_threshold} = {map_value.item():.4f}")
    print(f"{'='*60}\n")

    # Per-class breakdown
    print("Computing per-class AP...")
    per_class = compute_per_class_ap(
        pred_boxes, true_boxes,
        num_classes=NUM_CLASSES,
        iou_threshold=args.iou_threshold,
    )

    # Get class names from dataset
    class_names = dataset.class_names

    print(f"\nClasses evaluated: {len(per_class)}/{NUM_CLASSES}")
    print(f"  ({NUM_CLASSES - len(per_class)} classes have no ground truths in this split)")

    if per_class:
        # Sort by AP
        sorted_classes = sorted(per_class.items(), key=lambda x: x[1], reverse=True)

        print(f"\nTOP 10 BEST detected classes:")
        for class_idx, ap in sorted_classes[:10]:
            print(f"  {class_names[class_idx]:20s}: AP = {ap:.4f}")

        print(f"\nTOP 10 WORST detected classes:")
        for class_idx, ap in sorted_classes[-10:]:
            print(f"  {class_names[class_idx]:20s}: AP = {ap:.4f}")


if __name__ == "__main__":
    main()