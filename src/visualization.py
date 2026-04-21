"""
Visualization helpers for YOLO outputs.

Reusable functions to decode target tensors into human-readable boxes
and draw them on images. Used by both dataset debugging scripts and,
later, prediction visualization scripts.
"""

import colorsys
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont

from .config import IMAGE_SIZE, NUM_CLASSES, SPLIT_SIZE


def generate_class_colors(num_classes=NUM_CLASSES):
    """
    Generate a fixed palette of visually distinct colors, one per class.

    Uses HSV -> RGB conversion to spread colors evenly around the hue wheel.
    Returns a list of (r, g, b) tuples with values 0-255.
    """
    colors = []
    for i in range(num_classes):
        hue = i / num_classes
        rgb = colorsys.hsv_to_rgb(hue, 0.8, 0.9)
        colors.append(tuple(int(c * 255) for c in rgb))
    return colors


def decode_target(target, image_size=IMAGE_SIZE, S=SPLIT_SIZE, C=NUM_CLASSES):
    """
    Convert a YOLO target tensor into a list of boxes in pixel coordinates.

    Args:
        target:     tensor of shape (S, S, C + 5*B) — the ground truth.
        image_size: side length of the (resized) image in pixels.
        S:          grid size.
        C:          number of classes.

    Returns:
        List of dicts, each describing one box:
            {
                "class_idx": int,       # dense class index 0..C-1
                "x1", "y1", "x2", "y2": box corners in pixels (0..image_size)
                "center_x", "center_y": center of the object in pixels
                "cell_row", "cell_col": which grid cell owns the object
            }
    """
    boxes = []
    cell_size = image_size / S

    # Scan every grid cell; keep only those whose confidence slot is 1.
    conf_mask = target[..., C] == 1
    object_cells = torch.nonzero(conf_mask)

    for cell in object_cells:
        row, col = cell[0].item(), cell[1].item()

        # Class is the argmax over the first C positions.
        class_probs = target[row, col, :C]
        class_idx = int(torch.argmax(class_probs).item())

        # Box coords as stored in the target.
        x_in_cell, y_in_cell, w_norm, h_norm = target[row, col, C + 1:C + 5].tolist()

        # Convert back to full-image normalized coordinates (0..1).
        x_norm = (col + x_in_cell) / S
        y_norm = (row + y_in_cell) / S

        # Scale to pixel coordinates.
        center_x = x_norm * image_size
        center_y = y_norm * image_size
        w_px = w_norm * image_size
        h_px = h_norm * image_size

        # Box corners from center + size.
        x1 = center_x - w_px / 2
        y1 = center_y - h_px / 2
        x2 = center_x + w_px / 2
        y2 = center_y + h_px / 2

        boxes.append({
            "class_idx": class_idx,
            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            "center_x": center_x, "center_y": center_y,
            "cell_row": row, "cell_col": col,
        })

    return boxes


def tensor_to_pil(image_tensor):
    """
    Convert a tensor image (C, H, W) with values in [0, 1] to a PIL Image.
    """
    # From (3, H, W) to (H, W, 3) and from [0, 1] to [0, 255].
    array = (image_tensor.permute(1, 2, 0) * 255).byte().numpy()
    return Image.fromarray(array)


def draw_boxes(
    image,
    boxes,
    class_names,
    colors,
    line_width=3,
    draw_debug_overlay=False,
    image_size=IMAGE_SIZE,
    S=SPLIT_SIZE,
):
    """
    Draw bounding boxes, class labels, and optional debug overlays on an image.

    Args:
        image:              PIL Image to draw on (will be modified in place).
        boxes:              list of box dicts from decode_target().
        class_names:        list of 80 class names, indexed by class_idx.
        colors:             list of (r, g, b) tuples, one per class.
        line_width:         thickness of box lines. Use 3 for ground truth, 1 for predictions.
        draw_debug_overlay: if True, highlight the responsible cell and draw the center dot.
        image_size:         side length of the image.
        S:                  grid size (for the cell overlay).

    Returns:
        The same PIL Image with boxes drawn on it.
    """
    draw = ImageDraw.Draw(image)
    cell_size = image_size / S

    # Try to load a readable font; fall back to default if unavailable.
    try:
        font = ImageFont.truetype("arial.ttf", 14)
    except (OSError, IOError):
        font = ImageFont.load_default()

    for box in boxes:
        color = colors[box["class_idx"]]
        name = class_names[box["class_idx"]]

        # Debug overlay: highlight the responsible cell with a semi-transparent fill.
        if draw_debug_overlay:
            cell_x1 = box["cell_col"] * cell_size
            cell_y1 = box["cell_row"] * cell_size
            cell_x2 = cell_x1 + cell_size
            cell_y2 = cell_y1 + cell_size
            draw.rectangle(
                [cell_x1, cell_y1, cell_x2, cell_y2],
                outline=color,
                width=1,
            )

        # Main bounding box.
        draw.rectangle(
            [box["x1"], box["y1"], box["x2"], box["y2"]],
            outline=color,
            width=line_width,
        )

        # Class label with a filled background for readability.
        label_y = max(0, box["y1"] - 18)
        text_bbox = draw.textbbox((box["x1"], label_y), name, font=font)
        draw.rectangle(text_bbox, fill=color)
        draw.text((box["x1"], label_y), name, fill="white", font=font)

        # Debug overlay: draw a red dot at the object's center.
        if draw_debug_overlay:
            r = 4
            draw.ellipse(
                [box["center_x"] - r, box["center_y"] - r,
                 box["center_x"] + r, box["center_y"] + r],
                fill="red",
                outline="white",
                width=1,
            )

    return image


def get_class_names(dataset):
    """
    Return the list of class names from a COCODataset, indexed by class_idx.

    class_names[i] gives the human-readable name of class i (in the 0..79 range).
    """
    return dataset.class_names