"""
Export per-class AP for all 80 COCO classes to CSV and TXT files.

Reuses the existing functions from scripts/evaluate.py (load_model,
get_predictions_and_gt, compute_per_class_ap) so the numbers match
exactly what evaluate.py reports.

Generates two output files in artefacts/:
- per_class_ap.csv: machine-readable, sorted by AP descending
- per_class_ap.txt: human-readable, formatted table with summary

Usage:
    python scripts/export_per_class_ap.py \
        --checkpoint v6_run/best_v6.pth \
        --split test \
        --max-samples 6165 \
        --prob-threshold 0.1
"""

import argparse
import csv
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

# Add project root to path (same trick as evaluate.py)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import (
    BATCH_SIZE,
    NUM_BOXES,
    NUM_CLASSES,
    NUM_WORKERS,
    SPLIT_SIZE,
    TEST_ANNOTATIONS_FILE,
    VAL_ANNOTATIONS_FILE,
)
from src.dataset import COCODataset

# Reuse the functions already written in evaluate.py
sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate import (
    compute_per_class_ap,
    get_predictions_and_gt,
    load_model,
)
from src.metrics import mean_average_precision


def main():
    parser = argparse.ArgumentParser(
        description="Export per-class AP for all 80 COCO classes to CSV and TXT"
    )
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint (e.g. v6_run/best_v6.pth)")
    parser.add_argument("--split", type=str, default="test",
                        choices=["val", "test"],
                        help="Which split to evaluate on (default: test)")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Limit evaluation to N samples (default: all)")
    parser.add_argument("--prob-threshold", type=float, default=0.1,
                        help="Confidence threshold for predictions (default: 0.1)")
    parser.add_argument("--iou-threshold", type=float, default=0.5,
                        help="IoU threshold for mAP (default: 0.5)")
    parser.add_argument("--nms-iou-threshold", type=float, default=0.5,
                        help="IoU threshold for NMS (default: 0.5)")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help=f"Batch size (default: {BATCH_SIZE})")
    parser.add_argument("--output-dir", type=str, default="artefacts",
                        help="Directory to save output files (default: artefacts)")
    args = parser.parse_args()

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # Pick the right annotations file
    if args.split == "test":
        ann_file = TEST_ANNOTATIONS_FILE
    else:
        ann_file = VAL_ANNOTATIONS_FILE

    # Load model
    model = load_model(args.checkpoint, device)

    # Load dataset
    print(f"\nLoading {args.split} dataset from: {ann_file}")
    dataset = COCODataset(annotations_file=str(ann_file))
    print(f"  {len(dataset)} images in {args.split} split")

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
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

    # Overall mAP
    print(f"\nComputing overall mAP@{args.iou_threshold}...")
    map_value = mean_average_precision(
        pred_boxes,
        true_boxes,
        iou_threshold=args.iou_threshold,
        num_classes=NUM_CLASSES,
    )
    print(f"\n{'=' * 60}")
    print(f"RESULT: mAP@{args.iou_threshold} = {map_value.item():.4f}")
    print(f"{'=' * 60}")

    # Per-class AP
    print(f"\nComputing per-class AP...")
    per_class = compute_per_class_ap(
        pred_boxes, true_boxes,
        num_classes=NUM_CLASSES,
        iou_threshold=args.iou_threshold,
    )
    print(f"  Classes evaluated: {len(per_class)}/{NUM_CLASSES}")

    # Get class names from dataset
    class_names = dataset.class_names

    # Count ground truths per class
    gt_counts = {}
    for box in true_boxes:
        cls = int(box[1])
        gt_counts[cls] = gt_counts.get(cls, 0) + 1

    # Build full list of (class_idx, class_name, ap, num_gt) for all 80 classes
    # Classes with no GT in the split get AP = None (NA)
    rows = []
    for cls_idx in range(NUM_CLASSES):
        ap = per_class.get(cls_idx, None)
        rows.append({
            "class_idx": cls_idx,
            "class_name": class_names[cls_idx],
            "ap": ap,
            "num_gt": gt_counts.get(cls_idx, 0),
        })

    # Sort by AP descending. Classes with no GT (ap=None) go last.
    rows_sorted = sorted(
        rows,
        key=lambda r: (-1.0 if r["ap"] is None else -r["ap"]),
    )

    # Output paths
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "per_class_ap.csv"
    txt_path = output_dir / "per_class_ap.txt"

    # --- Write CSV (machine-readable) ---
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["class_idx", "class_name", "ap", "num_gt"])
        writer.writeheader()
        for row in rows_sorted:
            writer.writerow({
                "class_idx": row["class_idx"],
                "class_name": row["class_name"],
                "ap": "NA" if row["ap"] is None else f"{row['ap']:.4f}",
                "num_gt": row["num_gt"],
            })
    print(f"\nSaved: {csv_path}")

    # --- Write TXT (human-readable) ---
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("Per-class AP results\n")
        f.write("=" * 70 + "\n")
        f.write(f"Checkpoint:      {args.checkpoint}\n")
        f.write(f"Split:           {args.split}\n")
        f.write(f"Max samples:     {args.max_samples if args.max_samples else 'all'}\n")
        f.write(f"Prob threshold:  {args.prob_threshold}\n")
        f.write(f"NMS IoU:         {args.nms_iou_threshold}\n")
        f.write(f"mAP IoU:         {args.iou_threshold}\n")
        f.write(f"Total preds:     {len(pred_boxes)}\n")
        f.write(f"Total GT boxes:  {len(true_boxes)}\n")
        f.write(f"Classes w/ GT:   {len(per_class)}/{NUM_CLASSES}\n")
        f.write("\n")
        f.write(f"OVERALL mAP@{args.iou_threshold}: {map_value.item():.4f}\n")
        f.write("=" * 70 + "\n\n")

        f.write("All 80 classes sorted by AP (descending):\n")
        f.write("-" * 70 + "\n")
        f.write(f"{'Rank':>4}  {'Idx':>3}  {'Class':<22}  {'AP':>8}  {'GT count':>10}\n")
        f.write("-" * 70 + "\n")
        for rank, row in enumerate(rows_sorted, start=1):
            ap_str = "  NA   " if row["ap"] is None else f"{row['ap']:>8.4f}"
            f.write(
                f"{rank:>4}  "
                f"{row['class_idx']:>3}  "
                f"{row['class_name']:<22}  "
                f"{ap_str}  "
                f"{row['num_gt']:>10}\n"
            )
        f.write("-" * 70 + "\n\n")

        # Summary buckets (only counting classes with GT)
        rows_with_gt = [r for r in rows_sorted if r["ap"] is not None]
        ap_buckets = {
            "AP >= 0.20 (good)":           [r for r in rows_with_gt if r["ap"] >= 0.20],
            "0.10 <= AP < 0.20 (medium)":  [r for r in rows_with_gt if 0.10 <= r["ap"] < 0.20],
            "0.01 <= AP < 0.10 (low)":     [r for r in rows_with_gt if 0.01 <= r["ap"] < 0.10],
            "AP < 0.01 (effectively 0)":   [r for r in rows_with_gt if r["ap"] < 0.01],
        }

        f.write("Summary by AP bucket (only classes with GT in this split):\n")
        f.write("-" * 70 + "\n")
        for bucket_name, bucket_rows in ap_buckets.items():
            f.write(f"\n{bucket_name}: {len(bucket_rows)} classes\n")
            if bucket_rows:
                names = ", ".join(r["class_name"] for r in bucket_rows)
                f.write(f"  {names}\n")
        f.write("\n" + "=" * 70 + "\n")
    print(f"Saved: {txt_path}")

    # --- Quick summary to stdout ---
    rows_with_gt = [r for r in rows_sorted if r["ap"] is not None]
    print(f"\nQuick summary:")
    print(f"  AP >= 0.20:       {sum(1 for r in rows_with_gt if r['ap'] >= 0.20)} classes")
    print(f"  0.10 <= AP <0.20: {sum(1 for r in rows_with_gt if 0.10 <= r['ap'] < 0.20)} classes")
    print(f"  0.01 <= AP <0.10: {sum(1 for r in rows_with_gt if 0.01 <= r['ap'] < 0.10)} classes")
    print(f"  AP < 0.01:        {sum(1 for r in rows_with_gt if r['ap'] < 0.01)} classes")


if __name__ == "__main__":
    main()