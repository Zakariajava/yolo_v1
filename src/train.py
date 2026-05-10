"""
Distributed training loop for YOLOv1 on COCO.

Runs on multiple GPUs using PyTorch DistributedDataParallel (DDP).
Uses mixed precision (AMP) with fp16 for compute and fp32 for gradients,
SGD with momentum, and a warmup + cosine learning rate schedule.

Launch with torchrun for multi-GPU training:
    torchrun --nproc_per_node=8 src/train.py

For a single GPU or debug on one machine:
    python src/train.py --max-samples 500

All metrics (train loss, val loss, learning rate) are written to a CSV log
and printed to stdout from rank 0 only. Checkpoints are saved every
CHECKPOINT_EVERY epochs plus a "best" checkpoint tracking the best val loss.
"""

import argparse
import csv
import math
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler

from src.config import (
    BATCH_SIZE,
    CHECKPOINT_EVERY,
    CHECKPOINTS_DIR,
    GRAD_CLIP,
    LEARNING_RATE,
    LOG_EVERY,
    LOGS_DIR,
    MIN_LR,
    MOMENTUM,
    NUM_EPOCHS,
    NUM_WORKERS,
    TRAIN_ANNOTATIONS_FILE,
    VAL_ANNOTATIONS_FILE,
    WARMUP_EPOCHS,
    WEIGHT_DECAY,
    SPLIT_SIZE,
    NUM_BOXES,
    NUM_CLASSES,
)
from src.dataset import COCODataset
from src.loss import YoloLoss
from src.model import Yolov1


# ----------------------------- #
# DDP helpers                   #
# ----------------------------- #

def setup_ddp():
    """
    Initialize the default process group for DDP.
    torchrun sets LOCAL_RANK, WORLD_SIZE, and RANK as environment variables.
    """
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank


def cleanup_ddp():
    dist.destroy_process_group()


def is_main_process(rank):
    return rank == 0


def reduce_tensor(tensor, world_size):
    """Average a tensor across all processes. Used for logging global loss."""
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= world_size
    return rt


# ----------------------------- #
# LR scheduler: warmup + cosine #
# ----------------------------- #

def get_lr(step, total_steps, warmup_steps, base_lr, min_lr):
    """
    Linear warmup for the first `warmup_steps`, cosine decay afterwards.
    """
    if step < warmup_steps:
        return base_lr * (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * progress))


# ----------------------------- #
# Training and validation steps #
# ----------------------------- #

def train_one_epoch(
    model, loader, loss_fn, optimizer, scaler, device, epoch, total_epochs,
    step_counter, total_steps, warmup_steps, csv_writer, rank, world_size,
):
    model.train()
    running_loss = 0.0
    batches = 0

    for batch_idx, (images, targets) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        # Update learning rate for this step.
        lr = get_lr(step_counter[0], total_steps, warmup_steps, LEARNING_RATE, MIN_LR)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        # Forward pass under autocast (fp16 compute, fp32 master weights).
        optimizer.zero_grad(set_to_none=True)
        with autocast(device_type="cuda", dtype=torch.float16):
            predictions = model(images)
            loss = loss_fn(predictions, targets)

        # Backward with gradient scaling to avoid fp16 underflow.
        scaler.scale(loss).backward()
        # Unscale gradients before clipping (so we clip in real magnitude, not scaled).
        scaler.unscale_(optimizer)
        # Clip gradient norm to prevent divergence from a single bad batch.
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP)
        scaler.step(optimizer) # divide the gradients by the scale factor, verify if there is Inf or NaN, then if everting its okey calls optimizer.step() otherwise skip to the next
        scaler.update()

        # Average loss across GPUs for logging.
        reduced_loss = reduce_tensor(loss.detach(), world_size)
        running_loss += reduced_loss.item()
        batches += 1
        step_counter[0] += 1

        # Periodic stdout + CSV log (rank 0 only).
        if is_main_process(rank) and step_counter[0] % LOG_EVERY == 0:
            avg = running_loss / batches
            print(f"Epoch {epoch}/{total_epochs} | Step {step_counter[0]}/{total_steps} "
                  f"| train_loss: {avg:.4f} | lr: {lr:.2e}")
            csv_writer.writerow({
                "epoch": epoch,
                "step": step_counter[0],
                "train_loss": f"{avg:.4f}",
                "val_loss": "",
                "learning_rate": f"{lr:.2e}",
            })

    return running_loss / max(1, batches)


@torch.no_grad()
def validate(model, loader, loss_fn, device, world_size):
    model.eval()
    running_loss = 0.0
    batches = 0

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with autocast(device_type="cuda", dtype=torch.float16):
            predictions = model(images)
            loss = loss_fn(predictions, targets)

        reduced_loss = reduce_tensor(loss.detach(), world_size) # reduce_tensor: tensor to float 
        running_loss += reduced_loss.item()
        batches += 1

    return running_loss / max(1, batches)


# ----------------------------- #
# Checkpointing                 #
# ----------------------------- #

def save_checkpoint(model, optimizer, scaler, epoch, train_loss, val_loss, path):
    """Save a full training checkpoint. Only call from rank 0."""
    # Unwrap DDP to save the plain model state_dict.
    model_sd = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
    torch.save({
        "epoch": epoch,
        "model_state_dict": model_sd,
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "train_loss": train_loss,
        "val_loss": val_loss,
    }, path)


# ----------------------------- #
# Main                          #
# ----------------------------- #

def main():
    parser = argparse.ArgumentParser(description="Train YOLOv1 on COCO with DDP")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Use only the first N train samples (for quick debugging)")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to a checkpoint to resume from")
    args = parser.parse_args()

    rank, world_size, local_rank = setup_ddp()
    device = torch.device(f"cuda:{local_rank}")

    # Make sure output directories exist (only rank 0 creates, then barrier).
    if is_main_process(rank):
        CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
    dist.barrier()

    # Datasets and loaders.
    train_dataset = COCODataset(annotations_file=str(TRAIN_ANNOTATIONS_FILE))
    val_dataset = COCODataset(annotations_file=str(VAL_ANNOTATIONS_FILE))

    if args.max_samples is not None:
        train_dataset = Subset(train_dataset, range(min(args.max_samples, len(train_dataset))))
        if is_main_process(rank):
            print(f"DEBUG MODE: using only {len(train_dataset)} training samples")

    train_sampler = DistributedSampler(train_dataset, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, shuffle=False)

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, sampler=train_sampler,
        num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, sampler=val_sampler,
        num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True,
    )

    # Model, loss, optimizer, scaler.
    model = Yolov1(split_size=SPLIT_SIZE, num_boxes=NUM_BOXES, num_classes=NUM_CLASSES).to(device)
    model = DDP(model, device_ids=[local_rank])

    loss_fn = YoloLoss().to(device)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=LEARNING_RATE, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY,
    )
    scaler = GradScaler()

    # LR schedule math.
    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * NUM_EPOCHS
    warmup_steps = steps_per_epoch * WARMUP_EPOCHS

    # Resume logic.
    start_epoch = 1
    best_val_loss = float("inf")
    if args.resume:
        if is_main_process(rank):
            print(f"Resuming from {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.module.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_val_loss = checkpoint.get("val_loss", float("inf"))

    # CSV logger (rank 0 only).
    csv_file = None
    csv_writer = None
    if is_main_process(rank):
        log_path = LOGS_DIR / "train_log.csv"
        is_new = not log_path.exists()
        csv_file = open(log_path, "a", newline="")
        csv_writer = csv.DictWriter(csv_file, fieldnames=[
            "epoch", "step", "train_loss", "val_loss", "learning_rate",
        ])
        if is_new:
            csv_writer.writeheader()
            csv_file.flush()

    # Training loop.
    step_counter = [steps_per_epoch * (start_epoch - 1)]  # mutable for inner fn
    for epoch in range(start_epoch, NUM_EPOCHS + 1):
        train_sampler.set_epoch(epoch)  # shuffles differently each epoch

        start_time = time.time()
        train_loss = train_one_epoch(
            model, train_loader, loss_fn, optimizer, scaler, device,
            epoch, NUM_EPOCHS, step_counter, total_steps, warmup_steps,
            csv_writer, rank, world_size,
        )
        val_loss = validate(model, val_loader, loss_fn, device, world_size)
        elapsed = time.time() - start_time

        if is_main_process(rank):
            print(f"Epoch {epoch}/{NUM_EPOCHS} done | "
                  f"train_loss: {train_loss:.4f} | val_loss: {val_loss:.4f} | "
                  f"time: {elapsed:.1f}s")
            csv_writer.writerow({
                "epoch": epoch,
                "step": step_counter[0],
                "train_loss": f"{train_loss:.4f}",
                "val_loss": f"{val_loss:.4f}",
                "learning_rate": f"{optimizer.param_groups[0]['lr']:.2e}",
            })
            csv_file.flush()

            # Save periodic checkpoint.
            if epoch % CHECKPOINT_EVERY == 0:
                save_checkpoint(model, optimizer, scaler, epoch, train_loss,
                                val_loss, CHECKPOINTS_DIR / f"epoch_{epoch:03d}.pth")

            # Always save last.
            save_checkpoint(model, optimizer, scaler, epoch, train_loss, val_loss,
                            CHECKPOINTS_DIR / "last.pth")

            # Track best val loss.
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(model, optimizer, scaler, epoch, train_loss, val_loss,
                                CHECKPOINTS_DIR / "best.pth")
                print(f"New best val loss: {best_val_loss:.4f}")

    if is_main_process(rank):
        csv_file.close()
        print("Training complete.")

    cleanup_ddp()


if __name__ == "__main__":
    main()
