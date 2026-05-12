"""
Real-time YOLOv1 detection with live FPS overlay.

Three modes selected by flags:

1) Webcam (default):
       python scripts/realtime_detect.py --checkpoint v6_run/best_v6.pth

2) Webcam + record annotated output to disk:
       python scripts/realtime_detect.py --checkpoint v6_run/best_v6.pth \
           --record runs/realtime.mp4

3) Run on a video file (optionally recording):
       python scripts/realtime_detect.py --checkpoint v6_run/best_v6.pth \
           --source path/to/clip.mp4 --record runs/clip_annotated.mp4

4) Offline benchmark (no webcam, no display) — produces the clean
   inference-latency number for the academic report:
       python scripts/realtime_detect.py --checkpoint v6_run/best_v6.pth \
           --benchmark 1000

Controls during live mode:
    q  quit
    s  save a PNG snapshot of the current annotated frame
    +  raise the probability threshold by 0.05
    -  lower the probability threshold by 0.05
"""

import argparse
import sys
import time
from collections import deque
from pathlib import Path

import cv2
import torch
from torchvision import transforms

# Make `src.*` importable when running from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import IMAGE_SIZE, NUM_BOXES, NUM_CLASSES, SPLIT_SIZE
from src.model import Yolov1
from src.utils import cellboxes_to_boxes, non_max_suppression
from src.visualization import generate_class_colors


# COCO 80 class names in the same order the dataset builds them.
COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag",
    "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon",
    "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
    "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
    "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
    "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
]
assert len(COCO_CLASSES) == 80


# ----------------------------- #
# Model loading                  #
# ----------------------------- #

def load_model(checkpoint_path, device):
    """Load Yolov1 weights from either a clean state_dict or a full checkpoint."""
    print(f"Loading model from: {checkpoint_path}")
    model = Yolov1(
        split_size=SPLIT_SIZE,
        num_boxes=NUM_BOXES,
        num_classes=NUM_CLASSES,
    ).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"  Full checkpoint, epoch {ckpt.get('epoch', '?')}, "
              f"val_loss {ckpt.get('val_loss', '?')}")
    else:
        model.load_state_dict(ckpt)
        print("  Clean state_dict")
    model.eval()
    return model


# ----------------------------- #
# Preprocessing                  #
# ----------------------------- #

def make_transform():
    """Same transform as src/dataset.py — resize + ToTensor + ImageNet normalize."""
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


# ----------------------------- #
# Drawing on cv2 frames          #
# ----------------------------- #

def draw_boxes_cv2(frame, boxes, names, colors, line_width=2):
    """Draw NMS-filtered boxes onto a BGR cv2 frame, in place.

    Each box is [class_idx, prob, x_center, y_center, w, h] in image-normalised
    midpoint format (range [0, 1]).
    """
    h, w = frame.shape[:2]
    for box in boxes:
        class_idx = int(box[0])
        prob = float(box[1])
        x_c, y_c, w_n, h_n = box[2], box[3], abs(box[4]), abs(box[5])

        # Normalised midpoint -> pixel corners.
        x1 = int(round((x_c - w_n / 2) * w))
        y1 = int(round((y_c - h_n / 2) * h))
        x2 = int(round((x_c + w_n / 2) * w))
        y2 = int(round((y_c + h_n / 2) * h))

        # Clip to frame and skip degenerate boxes.
        x1 = max(0, min(w - 1, x1))
        y1 = max(0, min(h - 1, y1))
        x2 = max(0, min(w - 1, x2))
        y2 = max(0, min(h - 1, y2))
        if x2 <= x1 or y2 <= y1:
            continue

        # Our colour palette is RGB; cv2 wants BGR.
        r, g, b = colors[class_idx]
        color_bgr = (b, g, r)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color_bgr, line_width)

        label = f"{names[class_idx]} {prob:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        ytop = max(0, y1 - th - 4)
        cv2.rectangle(frame, (x1, ytop), (x1 + tw + 4, y1), color_bgr, -1)
        cv2.putText(frame, label, (x1 + 2, max(th, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255),
                    1, cv2.LINE_AA)
    return frame


def overlay_hud(frame, fps, inf_ms, n_boxes, prob_thr):
    """Top-left heads-up display with live stats."""
    lines = [
        f"FPS: {fps:5.1f}",
        f"Inf:  {inf_ms:5.1f} ms",
        f"Boxes: {n_boxes:3d}",
        f"p_thr: {prob_thr:.2f}",
    ]
    x, y = 10, 22
    for line in lines:
        # Black outline for contrast on any background.
        cv2.putText(frame, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 255, 0), 1, cv2.LINE_AA)
        y += 22


# ----------------------------- #
# Inference helpers              #
# ----------------------------- #

def infer_frame(model, frame_bgr, transform, device, prob_thr, iou_thr):
    """Run one frame through the full inference pipeline. Returns
    (boxes, inference_seconds). `boxes` is a list ready for draw_boxes_cv2."""
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    tensor = transform(rgb).unsqueeze(0).to(device, non_blocking=True)

    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        preds = model(tensor)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t_inf = time.perf_counter() - t0

    all_boxes = cellboxes_to_boxes(preds, S=SPLIT_SIZE)
    boxes = non_max_suppression(
        all_boxes[0],
        iou_threshold=iou_thr,
        prob_threshold=prob_thr,
        box_format="midpoint",
    )
    return boxes, t_inf


# ----------------------------- #
# Modes                          #
# ----------------------------- #

def open_capture(source):
    """Open webcam (integer source) or a video file. Uses DirectShow on Windows
    for faster webcam initialisation."""
    if source.isdigit():
        backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
        cap = cv2.VideoCapture(int(source), backend)
    else:
        cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {source!r}")
    return cap


def run_live(args, model, device):
    """Webcam or video-file mode with optional recording."""
    cap = open_capture(args.source)
    transform = make_transform()
    colors = generate_class_colors()

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
    print(f"Source: {args.source!r}, native {w}x{h} @ {src_fps:.1f} FPS")

    writer = None
    if args.record:
        Path(args.record).parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.record, fourcc, src_fps, (w, h))
        if not writer.isOpened():
            raise RuntimeError(f"Could not open writer for {args.record!r}")
        print(f"Recording to {args.record} ({w}x{h} @ {src_fps:.1f} FPS)")

    # Sliding-window stats (last 30 frames).
    frame_times = deque(maxlen=30)
    inf_times = deque(maxlen=30)
    prob_thr = args.prob_threshold

    print("Press 'q' to quit, 's' to snapshot, '+'/'-' to adjust prob threshold.")
    try:
        while True:
            t0 = time.perf_counter()
            ok, frame = cap.read()
            if not ok or frame is None:
                print("End of stream.")
                break

            boxes, t_inf = infer_frame(
                model, frame, transform, device, prob_thr, args.iou_threshold
            )
            inf_times.append(t_inf)
            draw_boxes_cv2(frame, boxes, COCO_CLASSES, colors,
                           line_width=args.line_width)

            frame_times.append(time.perf_counter() - t0)
            fps = len(frame_times) / sum(frame_times) if frame_times else 0.0
            inf_ms = 1000.0 * sum(inf_times) / len(inf_times) if inf_times else 0.0
            overlay_hud(frame, fps, inf_ms, len(boxes), prob_thr)

            cv2.imshow("YOLOv1 real-time", frame)
            if writer is not None:
                writer.write(frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("s"):
                stamp = time.strftime("%Y%m%d_%H%M%S")
                outp = Path("runs") / f"snapshot_{stamp}.png"
                outp.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(outp), frame)
                print(f"Saved snapshot {outp}")
            elif key in (ord("+"), ord("=")):
                prob_thr = min(0.95, prob_thr + 0.05)
                print(f"prob_threshold = {prob_thr:.2f}")
            elif key in (ord("-"), ord("_")):
                prob_thr = max(0.00, prob_thr - 0.05)
                print(f"prob_threshold = {prob_thr:.2f}")
    finally:
        cap.release()
        if writer is not None:
            writer.release()
            print(f"Saved video: {args.record}")
        cv2.destroyAllWindows()

    if inf_times:
        mean_fps = len(frame_times) / sum(frame_times)
        mean_inf = 1000.0 * sum(inf_times) / len(inf_times)
        print(f"\nSession summary: {mean_fps:.1f} pipeline FPS, "
              f"{mean_inf:.1f} ms mean inference (forward only).")


def run_benchmark(args, model, device):
    """Offline forward-only benchmark on synthetic batches.

    No webcam, no display, no decoding. Reports mean / median / p95 / p99 of
    the forward latency over `n` iterations after a warmup. This is the
    clean number to quote in an academic report.
    """
    n = args.benchmark
    warmup = max(20, n // 20)

    # Synthetic input matching what the dataset would produce: an ImageNet-normalised
    # tensor of zeros is a fair worst-case for latency since the network has the same
    # cost regardless of input values.
    x = torch.zeros(1, 3, IMAGE_SIZE, IMAGE_SIZE, device=device)

    print(f"Benchmark: {n} forward passes (after {warmup} warmup) "
          f"on {device}, batch=1, input={IMAGE_SIZE}x{IMAGE_SIZE}")

    with torch.no_grad():
        for _ in range(warmup):
            _ = model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()

        samples = []
        for _ in range(n):
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = model(x)
            if device.type == "cuda":
                torch.cuda.synchronize()
            samples.append(time.perf_counter() - t0)

    samples_ms = [s * 1000.0 for s in samples]
    samples_ms.sort()

    def pct(p):
        idx = max(0, min(len(samples_ms) - 1, int(round(p * (len(samples_ms) - 1)))))
        return samples_ms[idx]

    mean_ms = sum(samples_ms) / len(samples_ms)
    median = pct(0.50)
    p95 = pct(0.95)
    p99 = pct(0.99)
    fps_mean = 1000.0 / mean_ms
    fps_median = 1000.0 / median

    print(f"\nForward-only latency (ms): "
          f"mean {mean_ms:.2f}, median {median:.2f}, "
          f"p95 {p95:.2f}, p99 {p99:.2f}")
    print(f"Throughput at batch=1: {fps_mean:.1f} FPS (mean), "
          f"{fps_median:.1f} FPS (median)")

    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        mem_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
        print(f"Device: {gpu_name} | peak alloc: {mem_mb:.0f} MB")


# ----------------------------- #
# Entry point                    #
# ----------------------------- #

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=str, required=True,
                   help="Path to the trained YOLOv1 checkpoint.")
    p.add_argument("--source", type=str, default="0",
                   help="Webcam index (e.g. '0') or video file path. Default '0'.")
    p.add_argument("--record", type=str, default=None,
                   help="If given, save the annotated stream to this MP4 file.")
    p.add_argument("--prob-threshold", type=float, default=0.30,
                   help="NMS probability threshold (boxes below are discarded).")
    p.add_argument("--iou-threshold", type=float, default=0.50,
                   help="NMS IoU threshold for duplicate suppression.")
    p.add_argument("--line-width", type=int, default=2,
                   help="Bounding-box line thickness in pixels.")
    p.add_argument("--benchmark", type=int, default=None, metavar="N",
                   help="Run an offline forward-only benchmark over N iterations "
                        "and exit (no webcam, no display).")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"  {torch.cuda.get_device_name(0)}")

    model = load_model(args.checkpoint, device)

    if args.benchmark is not None:
        run_benchmark(args, model, device)
    else:
        run_live(args, model, device)


if __name__ == "__main__":
    main()
