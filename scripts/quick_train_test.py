"""
Quick training sanity check on a single GPU with mixed precision.
Trains for a few batches with a tiny subset to verify the pipeline works,
including autocast (fp16 forward) and GradScaler (backward scaling).

Usage:
    python scripts/quick_train_test.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Subset

from src.config import MOMENTUM, TRAIN_ANNOTATIONS_FILE, WEIGHT_DECAY
from src.dataset import COCODataset
from src.loss import YoloLoss
from src.model import Yolov1


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if device.type == "cpu":
        print("WARNING: CUDA not available — AMP only works on GPU.")
        return

    # Tiny subset for fast iteration
    full_dataset = COCODataset(annotations_file=str(TRAIN_ANNOTATIONS_FILE))
    dataset = Subset(full_dataset, range(100))

    loader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=0)

    model = Yolov1(split_size=7, num_boxes=2, num_classes=80).to(device)
    loss_fn = YoloLoss().to(device)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=1e-5,                    # warmup-like rate to keep training stable
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY,
    )
    scaler = GradScaler()

    model.train()
    print(f"\nRunning 50 training steps with AMP on {len(dataset)} samples...")

    for step, (images, targets) in enumerate(loader):
        if step >= 50:
            break
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        # Forward and loss in fp16
        with autocast(device_type="cuda", dtype=torch.float16):
            predictions = model(images)
            loss = loss_fn(predictions, targets)

        # Backward with gradient scaling
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # The scaler also exposes the current scale factor — useful to monitor
        print(f"Step {step + 1:>2}/50 | loss: {loss.item():>10.4f} "
              f"| scale: {scaler.get_scale():.0f}")

    print("\nSanity check passed: AMP training works without errors.")


if __name__ == "__main__":
    main()