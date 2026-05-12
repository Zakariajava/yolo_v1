"""
Round-trip test for convert_cellboxes: encode a known box into a target tensor
the way dataset.py does, then decode it with convert_cellboxes and check that
the coordinates round-trip back to the original.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from src.config import NUM_CLASSES, NUM_BOXES, SPLIT_SIZE
from src.utils import convert_cellboxes

S, B, C = SPLIT_SIZE, NUM_BOXES, NUM_CLASSES

# An object of class 7 centered at image-normalized (0.5, 0.5) with size (0.3, 0.3).
# That center falls in cell (row=3, col=3) for S=7 (since 0.5*7 = 3.5, int = 3).
# x_in_cell = 7*0.5 - 3 = 0.5
# y_in_cell = 7*0.5 - 3 = 0.5
# w_norm = 0.3, h_norm = 0.3

target = torch.zeros(1, S, S, C + 5 * B)
row, col = 3, 3
target[0, row, col, 7] = 1               # class
target[0, row, col, C] = 1               # conf1
target[0, row, col, C+1:C+5] = torch.tensor([0.5, 0.5, 0.3, 0.3])

# We're pretending the model predicted exactly the target.  We also fill box2 so
# the conf2 slot (target[..., C+5]) wins-or-not selection has something to chew on.
pred = target.clone()
pred[0, row, col, C+5] = 0
pred[0, row, col, C+6:C+10] = torch.tensor([0.5, 0.5, 0.3, 0.3])

flat = pred.reshape(1, -1)
out = convert_cellboxes(flat, S=S, C=C, B=B)
# out has shape (1, S*S, 6) per row: [class, prob, x, y, w, h] in image-normalized coords
cell_flat = row * S + col
class_idx, prob, x, y, w, h = out[0, cell_flat].tolist()
print(f"Decoded cell ({row},{col}):")
print(f"  class={class_idx}, prob={prob:.4f}, x={x:.4f}, y={y:.4f}, w={w:.4f}, h={h:.4f}")
print(f"Expected:")
print(f"  class=7,    prob=1.0,    x=0.5,    y=0.5,    w=0.3,    h=0.3")
print(f"\nDiff in w: {w - 0.3:+.4f}   (should be ~0; if it's -0.257, w is divided by S)")
print(f"Diff in h: {h - 0.3:+.4f}")
print(f"Diff in x: {x - 0.5:+.4f}")
print(f"Diff in y: {y - 0.5:+.4f}")
