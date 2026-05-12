"""
Demonstrate that YoloLoss reads VOC indices (20/21:25/25/26:30) on a COCO target
(80 classes, layout indices 80/81:85/85/86:90), making coordinate and confidence
training a near-noop.

We verify three things:
1. Real COCO samples — what fraction of cells does the buggy loss treat as "object" cells?
2. Round-trip: a perfect COCO prediction yields the same loss as one with random
   garbage in the actual COCO box slots, because the loss never reads those slots.
3. The "exists_box" mask the loss uses (target[..., 20]) corresponds to *class index 20*,
   not the real object-confidence slot (target[..., 80]).
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from src.config import NUM_CLASSES, NUM_BOXES, SPLIT_SIZE, TRAIN_ANNOTATIONS_FILE
from src.loss import YoloLoss

S, B, C = SPLIT_SIZE, NUM_BOXES, NUM_CLASSES
print(f"Config: S={S}, B={B}, C={C}, per-cell channels = {C + 5*B}")

# ---------------------------------------------------------------------------
# 1) Real COCO targets — what does the loss actually train on?
# ---------------------------------------------------------------------------
try:
    from src.dataset import COCODataset
    ds = COCODataset(annotations_file=str(TRAIN_ANNOTATIONS_FILE))
    n = min(200, len(ds))
    real_obj_cells = 0
    voc_exists_cells = 0
    class_20_objects = 0
    real_class_counts = torch.zeros(80)
    for i in range(n):
        _, t = ds[i]
        # Real object cells: positions where target[..., 80] (the real conf slot) is 1
        real_obj_cells += int((t[..., C] == 1).sum().item())
        # What the loss calls "exists_box": target[..., 20] (a class probability slot for class index 20)
        voc_exists_cells += int((t[..., 20] == 1).sum().item())
        # Count how often class 20 appears as an annotation
        for r in range(S):
            for c in range(S):
                if t[r, c, C] == 1:
                    real_class_counts[int(torch.argmax(t[r, c, :C]).item())] += 1
        if (t[..., 20] == 1).any():
            class_20_objects += 1
    print(f"\n[1] Over {n} real COCO images:")
    print(f"    Real object cells (target[..., 80] == 1):                 {real_obj_cells}")
    print(f"    Loss-perceived 'exists_box' cells (target[..., 20] == 1): {voc_exists_cells}")
    print(f"    Ratio:                                                    "
          f"{voc_exists_cells/real_obj_cells*100:.1f}% of real objects are seen by the loss")
    print(f"    (the rest contribute ZERO to box_loss and object_loss)")
    top = torch.topk(real_class_counts, 5)
    print(f"    Top 5 most common classes (idx -> count): {[(int(i.item()), int(c.item())) for i, c in zip(top.indices, top.values)]}")
    print(f"    Class index 20 = {ds.class_names[20]!r}  (the only class the buggy loss localizes)")
    print(f"    Classes 0..19 (which the class_loss trains): "
          f"{ds.class_names[:20]}")
    print(f"    Classes 20..79 are NEVER trained by class_loss.")
except FileNotFoundError as e:
    print(f"\n[1] Skipping real-data check ({e})")

# ---------------------------------------------------------------------------
# 2) Round-trip on a synthetic single-class target
# ---------------------------------------------------------------------------
loss_fn = YoloLoss(S=S, B=B, C=C)

# Object of class 7 in cell (3,4) at the COCO positions.
target = torch.zeros(1, S, S, C + 5 * B)
target[0, 3, 4, 7] = 1
target[0, 3, 4, C] = 1
target[0, 3, 4, C+1:C+5] = torch.tensor([0.5, 0.5, 0.3, 0.3])

pred_perfect = target.clone()
loss_perfect = loss_fn(pred_perfect.reshape(1, -1), target).item()

pred_garbage_at_coco_slots = target.clone()
pred_garbage_at_coco_slots[..., C+1:C+5] = 99.0   # corrupt the real box coords
pred_garbage_at_coco_slots[..., C] = -42.0        # corrupt the real conf
pred_garbage_at_coco_slots[..., C+6:C+10] = 99.0  # corrupt box2
loss_garbage = loss_fn(pred_garbage_at_coco_slots.reshape(1, -1), target).item()

print(f"\n[2] Synthetic test (one class-7 object):")
print(f"    Perfect COCO prediction                         -> loss = {loss_perfect:.6f}")
print(f"    Identical, but CORRUPT box coords at [81:85]    -> loss = {loss_garbage:.6f}")
print(f"    Difference: {abs(loss_garbage - loss_perfect):.6f}")
print(f"    -> The loss is INSENSITIVE to the real COCO box coords. Model never learns to localize.")

# ---------------------------------------------------------------------------
# 3) What 'exists_box' refers to on a class-7 target
# ---------------------------------------------------------------------------
exists = target[..., 20:21]
print(f"\n[3] target[..., 20] (what the loss calls 'exists_box') for a class-7 object:")
print(f"    sum = {exists.sum().item()}  (should be 1 if the loss could see the object; it's 0)")
print(f"    -> For this image the loss thinks the entire grid is empty.")
print(f"       box_loss and object_loss are both multiplied by 0 and contribute nothing.")
