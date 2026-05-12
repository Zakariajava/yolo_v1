"""
Debug: print actual predictions vs GTs to see what's happening.
"""

import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import (
    BATCH_SIZE, NUM_BOXES, NUM_CLASSES, NUM_WORKERS,
    SPLIT_SIZE, VAL_ANNOTATIONS_FILE,
)
from src.dataset import COCODataset
from src.model import Yolov1
from src.utils import cellboxes_to_boxes, non_max_suppression, intersection_over_union


def extract_gt_boxes_from_target(target, image_idx, S=SPLIT_SIZE, C=NUM_CLASSES):
    boxes = []
    conf_mask = target[..., C] == 1
    object_cells = torch.nonzero(conf_mask)
    for cell in object_cells:
        row, col = cell[0].item(), cell[1].item()
        class_idx = int(torch.argmax(target[row, col, :C]).item())
        x_in_cell, y_in_cell, w_norm, h_norm = target[row, col, C + 1:C + 5].tolist()
        x_global = (col + x_in_cell) / S
        y_global = (row + y_in_cell) / S
        boxes.append([image_idx, class_idx, 1.0, x_global, y_global, w_norm, h_norm])
    return boxes


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = Yolov1(split_size=SPLIT_SIZE, num_boxes=NUM_BOXES, num_classes=NUM_CLASSES).to(device)

    ckpt = torch.load("v3_run/best_v3.pth", map_location=device, weights_only=False)
    model.load_state_dict(ckpt if "model_state_dict" not in ckpt else ckpt["model_state_dict"])
    model.eval()
    
    dataset = COCODataset(annotations_file=str(VAL_ANNOTATIONS_FILE))
    class_names = dataset.class_names
    
    # Process first 20 images, show pred vs GT
    print(f"{'='*80}")
    print(f"DETAILED INSPECTION: V1 predictions vs Ground Truth")
    print(f"{'='*80}\n")
    
    for img_idx in range(20):
        image_tensor, target = dataset[img_idx]
        
        with torch.no_grad():
            pred = model(image_tensor.unsqueeze(0).to(device))
        
        bboxes = cellboxes_to_boxes(pred)[0]
        nms_boxes = non_max_suppression(
            bboxes,
            iou_threshold=0.5,
            prob_threshold=0.2,
            box_format="midpoint",
        )
        
        gt_boxes = extract_gt_boxes_from_target(target, img_idx)
        
        if not nms_boxes and not gt_boxes:
            continue  # skip if both empty
        
        print(f"\n--- Image {img_idx} ---")
        print(f"Ground truths ({len(gt_boxes)}):")
        for gt in gt_boxes:
            cls = class_names[gt[1]]
            print(f"  {cls:15s} at ({gt[3]:.3f}, {gt[4]:.3f}) size ({gt[5]:.3f}, {gt[6]:.3f})")
        
        print(f"Predictions after NMS ({len(nms_boxes)}):")
        for pred_box in nms_boxes:
            # pred_box: [class, prob, x, y, w, h]
            cls = class_names[int(pred_box[0])]
            print(f"  {cls:15s} prob={pred_box[1]:.3f} at ({pred_box[2]:.3f}, {pred_box[3]:.3f}) "
                  f"size ({pred_box[4]:.3f}, {pred_box[5]:.3f})")
        
        # Compute IoU between each pred and each GT of same class
        if nms_boxes and gt_boxes:
            print(f"IoU matches:")
            for p_idx, pred_box in enumerate(nms_boxes):
                pred_class = int(pred_box[0])
                for gt in gt_boxes:
                    if gt[1] == pred_class:
                        iou = intersection_over_union(
                            torch.tensor(pred_box[2:]),
                            torch.tensor(gt[3:]),
                            box_format="midpoint",
                        ).item()
                        match = "✓ TP" if iou > 0.5 else "✗"
                        print(f"  Pred #{p_idx} ({class_names[pred_class]}) ↔ GT ({class_names[gt[1]]}): IoU={iou:.3f} {match}")


if __name__ == "__main__":
    main()