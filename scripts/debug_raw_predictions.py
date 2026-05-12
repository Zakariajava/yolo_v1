"""
Debug: print raw model output (before NMS, before cellboxes_to_boxes).
Compare to what cellboxes_to_boxes produces.
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import NUM_BOXES, NUM_CLASSES, SPLIT_SIZE, VAL_ANNOTATIONS_FILE
from src.dataset import COCODataset
from src.model import Yolov1
from src.utils import cellboxes_to_boxes


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = Yolov1(split_size=SPLIT_SIZE, num_boxes=NUM_BOXES, num_classes=NUM_CLASSES).to(device)
    ckpt = torch.load("v4_run/best_v4.pth", map_location=device, weights_only=False)
    model.load_state_dict(ckpt if "model_state_dict" not in ckpt else ckpt["model_state_dict"])
    model.eval()
    
    dataset = COCODataset(annotations_file=str(VAL_ANNOTATIONS_FILE))
    image_tensor, target = dataset[11]  # the cat image
    
    # Forward
    with torch.no_grad():
        pred = model(image_tensor.unsqueeze(0).to(device))
    
    pred = pred.cpu()
    pred_reshaped = pred.reshape(1, 7, 7, 90)
    
    C = 80
    
    print("RAW MODEL OUTPUT for image 11 (cat in image)")
    print("=" * 80)
    
    # Find top-3 cells by confidence
    conf1 = pred_reshaped[0, :, :, C]      # (7, 7)
    conf2 = pred_reshaped[0, :, :, C + 5]  # (7, 7)
    combined = torch.max(conf1, conf2)
    
    top_idx = combined.flatten().argsort(descending=True)[:5]
    
    print(f"\nTop 5 cells by raw confidence:")
    for idx in top_idx:
        row = (idx // 7).item()
        col = (idx % 7).item()
        c1 = conf1[row, col].item()
        c2 = conf2[row, col].item()
        
        # Get the two boxes
        box1 = pred_reshaped[0, row, col, C+1:C+5].tolist()  # x, y, w, h of box 1
        box2 = pred_reshaped[0, row, col, C+6:C+10].tolist()  # x, y, w, h of box 2
        
        cls_idx = pred_reshaped[0, row, col, :C].argmax().item()
        cls_prob = pred_reshaped[0, row, col, :C].max().item()
        
        print(f"\n  Cell ({row}, {col}):")
        print(f"    conf1 = {c1:.4f}, conf2 = {c2:.4f}")
        print(f"    box1 (raw): x={box1[0]:.4f}, y={box1[1]:.4f}, w={box1[2]:.4f}, h={box1[3]:.4f}")
        print(f"    box2 (raw): x={box2[0]:.4f}, y={box2[1]:.4f}, w={box2[2]:.4f}, h={box2[3]:.4f}")
        print(f"    class_idx={cls_idx}, class_prob={cls_prob:.4f}")
    
    print("\n" + "=" * 80)
    print("WHAT cellboxes_to_boxes PRODUCES")
    print("=" * 80)
    
    all_bboxes = cellboxes_to_boxes(pred)[0]
    
    # Show all with confidence > 0.1
    sig_boxes = [b for b in all_bboxes if b[1] > 0.1]
    print(f"\nCells with confidence > 0.1 ({len(sig_boxes)}):")
    for b in sig_boxes[:10]:
        print(f"  class={int(b[0])} prob={b[1]:.4f} x={b[2]:.4f} y={b[3]:.4f} w={b[4]:.4f} h={b[5]:.4f}")
    
    print("\n" + "=" * 80)
    print("GROUND TRUTH for comparison")
    print("=" * 80)
    print(f"  cat at center (0.345, 0.501) size (0.580, 0.920)")
    print(f"  In which cell? (0.345 * 7 = 2.4 → col 2, 0.501 * 7 = 3.5 → row 3)")


if __name__ == "__main__":
    main()