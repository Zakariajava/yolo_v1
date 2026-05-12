"""
Debug script: print raw model predictions to understand what's happening.
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import ANNOTATIONS_FILE, NUM_BOXES, NUM_CLASSES, SPLIT_SIZE
from src.dataset import COCODataset
from src.model import Yolov1


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model
    model = Yolov1(split_size=SPLIT_SIZE, num_boxes=NUM_BOXES, num_classes=NUM_CLASSES).to(device)
    checkpoint = torch.load("v1_run/best_v1.pth", map_location=device, weights_only=False)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    
    # Load one sample
    dataset = COCODataset(annotations_file=ANNOTATIONS_FILE)
    image_tensor, target = dataset[42]
    
    # Predict
    with torch.no_grad():
        pred = model(image_tensor.unsqueeze(0).to(device))
    
    pred = pred.reshape(1, 7, 7, 90).cpu()
    
    C = 80
    # Extract confidences
    conf1 = pred[0, :, :, C]      # confidence of box 1, shape (7, 7)
    conf2 = pred[0, :, :, C + 5]  # confidence of box 2, shape (7, 7)
    
    print("=" * 60)
    print("CONFIDENCES (box 1) - raw model output:")
    print(f"  min: {conf1.min().item():.4f}")
    print(f"  max: {conf1.max().item():.4f}")
    print(f"  mean: {conf1.mean().item():.4f}")
    print(f"  values > 0.1: {(conf1 > 0.1).sum().item()} / 49")
    print(f"  values > 0.05: {(conf1 > 0.05).sum().item()} / 49")
    
    print("\nCONFIDENCES (box 2):")
    print(f"  min: {conf2.min().item():.4f}")
    print(f"  max: {conf2.max().item():.4f}")
    print(f"  mean: {conf2.mean().item():.4f}")
    
    # Extract class probabilities
    classes = pred[0, :, :, :C]  # shape (7, 7, 80)
    class_max = classes.max(dim=-1).values  # max prob per cell
    print(f"\nCLASS PROBABILITIES (max per cell):")
    print(f"  min: {class_max.min().item():.4f}")
    print(f"  max: {class_max.max().item():.4f}")
    print(f"  mean: {class_max.mean().item():.4f}")
    
    # Combined score (what's used as "prob" in NMS)
    # In convert_cellboxes, prob = max(conf1, conf2) (no multiplication by class)
    combined = torch.max(conf1, conf2)
    print(f"\nCOMBINED CONFIDENCE (max of conf1, conf2) — what NMS uses as 'prob':")
    print(f"  min: {combined.min().item():.4f}")
    print(f"  max: {combined.max().item():.4f}")
    print(f"  mean: {combined.mean().item():.4f}")
    print(f"  values > 0.2: {(combined > 0.2).sum().item()} / 49")
    print(f"  values > 0.1: {(combined > 0.1).sum().item()} / 49")
    print(f"  values > 0.05: {(combined > 0.05).sum().item()} / 49")
    
    # Best combined: conf * class_prob
    full_score = combined * class_max
    print(f"\nFULL SCORE (conf * class_prob):")
    print(f"  min: {full_score.min().item():.4f}")
    print(f"  max: {full_score.max().item():.4f}")
    print(f"  mean: {full_score.mean().item():.4f}")
    
    # Print top 5 cells
    print(f"\n{'='*60}")
    print("TOP 5 CELLS BY COMBINED CONFIDENCE:")
    flat_conf = combined.flatten()
    top_idx = flat_conf.argsort(descending=True)[:5]
    for i, idx in enumerate(top_idx):
        row = (idx // 7).item()
        col = (idx % 7).item()
        c1 = conf1[row, col].item()
        c2 = conf2[row, col].item()
        cls_idx = classes[row, col].argmax().item()
        cls_prob = classes[row, col].max().item()
        print(f"  #{i+1}: cell ({row},{col}) | conf1={c1:.3f} | conf2={c2:.3f} | "
              f"class_idx={cls_idx} | class_prob={cls_prob:.3f}")


if __name__ == "__main__":
    main()