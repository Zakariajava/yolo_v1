"""
Visualize what the trained YOLOv1 model predicts on a sample image.

Loads a checkpoint, runs the model on one image from the dataset, applies NMS,
and draws the predicted bounding boxes. Optionally overlays the ground truth
boxes (thicker lines) for visual comparison.

Usage:
    python scripts/visualize_prediction.py --idx 42
    python scripts/visualize_prediction.py --random
    python scripts/visualize_prediction.py --idx 42 --show-gt
    python scripts/visualize_prediction.py --idx 42 --checkpoint v1_run/best_v1.pth
    python scripts/visualize_prediction.py --idx 42 --prob-threshold 0.2

Output is saved to artefacts/predictions/.
"""

import argparse
import random
import sys
from pathlib import Path

import torch

# Add project root to path so we can import from src/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import (
    ANNOTATIONS_FILE,
    NUM_BOXES,
    NUM_CLASSES,
    SPLIT_SIZE,
)
from src.dataset import COCODataset
from src.model import Yolov1
from src.visualization import (
    decode_predictions,
    decode_target,
    draw_boxes,
    generate_class_colors,
    get_class_names,
    tensor_to_pil,
)


PREDICTIONS_DIR = Path("artefacts/predictions")


def load_model(checkpoint_path, device):
    """Load the YOLOv1 model and weights from a checkpoint."""
    print(f"Loading model from: {checkpoint_path}")

    model = Yolov1(
        split_size=SPLIT_SIZE,
        num_boxes=NUM_BOXES,
        num_classes=NUM_CLASSES,
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Checkpoints can be:
    #   (a) just a state_dict (clean weights), or
    #   (b) a full dict with model_state_dict, optimizer_state_dict, etc.
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"  Loaded from full checkpoint (epoch {checkpoint.get('epoch', '?')})")
    else:
        model.load_state_dict(checkpoint)
        print(f"  Loaded from clean weights")

    model.eval()  # set the model to inference mode (no dropout, no BN updates)
    return model


def main():
    parser = argparse.ArgumentParser(description="Visualize YOLOv1 predictions on a sample")
    parser.add_argument("--idx", type=int, help="Index of the sample to visualize")
    parser.add_argument("--random", action="store_true", help="Pick a random sample")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="v2_run/best_v2.pth",
        help="Path to the trained model checkpoint (default: v2_run/best_v2.pth)",
    )
    parser.add_argument(
        "--show-gt",
        action="store_true",
        help="Overlay ground truth boxes (thicker lines) for comparison",
    )
    parser.add_argument(
        "--prob-threshold",
        type=float,
        default=0.2,
        help="NMS probability threshold (default: 0.2)",
    )
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=0.5,
        help="NMS IoU threshold (default: 0.5)",
    )
    parser.add_argument("--output", type=str, default=None, help="Output PNG path (optional)")
    args = parser.parse_args()

    if args.idx is None and not args.random:
        parser.error("Must specify --idx or --random")

    # Device: use CUDA if available, else CPU.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load model and dataset.
    model = load_model(args.checkpoint, device)
    print("\nLoading dataset...")
    dataset = COCODataset(annotations_file=ANNOTATIONS_FILE)

    # Pick a sample index.
    idx = args.idx if args.idx is not None else random.randint(0, len(dataset) - 1)
    if idx < 0 or idx >= len(dataset):
        parser.error(f"Index {idx} out of range [0, {len(dataset) - 1}]")

    print(f"\nVisualizing sample {idx}...")
    img_info = dataset.images[idx]
    print(f"  File: {img_info['file_name']}")
    print(f"  Original size: {img_info['width']} x {img_info['height']}")

    # Load the sample (resize + tensor + target).
    image_tensor, target = dataset[idx]

    # Run inference: model expects a batch dimension.
    with torch.no_grad():
        image_batch = image_tensor.unsqueeze(0).to(device)
        predictions = model(image_batch)

    # Decode predictions into drawable boxes (NMS applied internally).
    pred_boxes = decode_predictions(
        predictions[0].cpu(),
        iou_threshold=args.iou_threshold,
        prob_threshold=args.prob_threshold,
    )

    print(f"  Predicted objects (after NMS): {len(pred_boxes)}")

    # Decode ground truth too if requested.
    gt_boxes = []
    if args.show_gt:
        gt_boxes = decode_target(target)
        print(f"  Ground truth objects: {len(gt_boxes)}")

    # Prepare rendering resources.
    class_names = get_class_names(dataset)
    colors = generate_class_colors()

    # Convert tensor back to PIL for drawing.
    image = tensor_to_pil(image_tensor)

    # Draw ground truth FIRST (thicker line), so predictions go on top.
    if args.show_gt:
        draw_boxes(
            image,
            gt_boxes,
            class_names,
            colors,
            line_width=3,
            draw_debug_overlay=False,
        )

    # Draw predictions (thinner line).
    draw_boxes(
        image,
        pred_boxes,
        class_names,
        colors,
        line_width=1,
        draw_debug_overlay=False,
    )

    # Save.
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    if args.output:
        output_path = Path(args.output)
    else:
        # Extract the model version (e.g. "v1", "v2") from the checkpoint path.
        # Examples:
        #   v2_run/best_v2.pth → "v2"
        #   v1_run/best_v1.pth → "v1"
        #   custom/foo.pth     → "custom"
        checkpoint_path = Path(args.checkpoint)
        if "v1" in checkpoint_path.parent.name:
            model_tag = "v1"
        elif "v2" in checkpoint_path.parent.name:
            model_tag = "v2"
        else:
            model_tag = checkpoint_path.stem  # fallback: filename without extension
        
        gt_suffix = "_with_gt" if args.show_gt else ""
        output_path = PREDICTIONS_DIR / f"pred_{idx:06d}_{model_tag}{gt_suffix}.png"
    image.save(output_path)
    print(f"\nSaved visualization to: {output_path}")

    # Summary of predicted objects.
    if pred_boxes:
        print("\nPredicted objects:")
        for b in pred_boxes:
            name = class_names[b["class_idx"]]
            print(f"  - {name:20s} (prob={b['prob']:.3f})")


if __name__ == "__main__":
    main()