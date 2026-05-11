# ============================ #
# YOLOv configuration         #
# ============================ #

from pathlib import Path

# Project root, computed from this file's location
ROOT_DIR = Path(__file__).resolve().parent.parent

# Grid and prediction structure
SPLIT_SIZE = 7
NUM_BOXES = 2
NUM_CLASSES = 80

# Input image size
IMAGE_SIZE = 448

# Loss weights (from the original YOLOv1 paper)
LAMBDA_COORD = 5.0
LAMBDA_NOOBJ = 0.5

# ============================ #
# Training hyperparameters     #
# ============================ #

LEARNING_RATE = 1e-4
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4
BATCH_SIZE = 32
NUM_EPOCHS = 50
NUM_WORKERS = 8

WARMUP_EPOCHS = 3
MIN_LR = 1e-5

CHECKPOINT_EVERY = 2
LOG_EVERY = 50

# ============================ #
# Paths                        #
# ============================ #

DATA_DIR = ROOT_DIR / "data" / "coco"
IMAGES_DIR = DATA_DIR / "images"

ANNOTATIONS_FILE = DATA_DIR / "annotations" / "instances_all_clean.json"
TRAIN_ANNOTATIONS_FILE = DATA_DIR / "annotations" / "instances_train.json"
VAL_ANNOTATIONS_FILE = DATA_DIR / "annotations" / "instances_val.json"
TEST_ANNOTATIONS_FILE = DATA_DIR / "annotations" / "instances_test.json"

CHECKPOINTS_DIR = ROOT_DIR / "checkpoints"
LOGS_DIR = ROOT_DIR / "logs"
SAMPLES_DIR = ROOT_DIR / "artefacts" / "samples"

# Gradient clipping (prevents divergence from single bad batches)
GRAD_CLIP = 10.0
