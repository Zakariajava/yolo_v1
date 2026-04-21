"""
Visualize a sample from the COCO dataset with its bounding boxes drawn.

Used to verify that the dataset builds targets correctly, and later to
compare model predictions against ground truth.

Usage:
    python scripts/visualize_sample.py --idx 42
    python scripts/visualize_sample.py --random
    python scripts/visualize_sample.py --idx 42 --no-debug
"""

import argparse
import random
import sys
from pathlib import Path

# Add project root to path so we can import from src/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import ANNOTATIONS_FILE, SAMPLES_DIR
from src.dataset import COCODataset
from src.visualization import (
    decode_target,
    draw_boxes,
    generate_class_colors,
    get_class_names,
    tensor_to_pil,
)


def main():
    parser = argparse.ArgumentParser(description="Visualize a COCO dataset sample")
    parser.add_argument("--idx", type=int, help="Index of the sample to visualize")
    parser.add_argument("--random", action="store_true", help="Pick a random sample")
    parser.add_argument("--output", type=str, default=None, help="Output PNG path (optional)")
    parser.add_argument("--no-debug", action="store_true", help="Disable debug overlay (cells + centers)")
    args = parser.parse_args()

    if args.idx is None and not args.random:
        parser.error("Must specify --idx or --random")

    # Load dataset and metadata.
    print("Loading dataset...")
    dataset = COCODataset(annotations_file=ANNOTATIONS_FILE)

    idx = args.idx if args.idx is not None else random.randint(0, len(dataset) - 1)
    if idx < 0 or idx >= len(dataset):
        parser.error(f"Index {idx} out of range [0, {len(dataset) - 1}]")

    print(f"Visualizing sample {idx}...")
    img_info = dataset.images[idx]
    print(f"  File: {img_info['file_name']}")
    print(f"  Original size: {img_info['width']} x {img_info['height']}")

    # Load the sample through the dataset (resize + tensor + target construction).
    image_tensor, target = dataset[idx]

    # Decode target into drawable boxes.
    boxes = decode_target(target)
    print(f"  Objects in target: {len(boxes)}")

    # Prepare rendering resources.
    class_names = get_class_names(dataset)
    colors = generate_class_colors()

    # Convert tensor back to PIL for drawing.
    image = tensor_to_pil(image_tensor)

    # Draw everything.
    draw_boxes(
        image,
        boxes,
        class_names,
        colors,
        line_width=3,
        draw_debug_overlay=not args.no_debug,
    )

    # Save the result.
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    output_path = Path(args.output) if args.output else SAMPLES_DIR / f"sample_{idx:06d}.png"
    image.save(output_path)
    print(f"\nSaved visualization to: {output_path}")

    # Summary of detected objects.
    if boxes:
        print("\nDetected objects:")
        for b in boxes:
            print(f"  - {class_names[b['class_idx']]:20s} at cell ({b['cell_row']}, {b['cell_col']})")


if __name__ == "__main__":
    main()