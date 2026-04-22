# ============================ #
# YOLOv1 configuration         #
# ============================ #

from pathlib import Path

# ROOT_DIR = the project root, computed from this file's location
# config.py is at <root>/src/config.py, so parent.parent gives <root>
ROOT_DIR = Path(__file__).resolve().parent.parent


# Grid and prediction structure
SPLIT_SIZE = 7                   # S — grid is S x S cells
NUM_BOXES = 2                    # B — boxes predicted per cell
NUM_CLASSES = 80                 # C — number of object classes (COCO)

# Input image size
IMAGE_SIZE = 448                 # images are resized to IMAGE_SIZE x IMAGE_SIZE

# Loss weights (from the original YOLOv1 paper)
LAMBDA_COORD = 5.0               # upweights localization error
LAMBDA_NOOBJ = 0.5               # downweights confidence error on empty cells

# Training hyperparameters
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.0005
NUM_EPOCHS = 100



# Paths
DATA_DIR = ROOT_DIR / "data" / "coco"
IMAGES_DIR = DATA_DIR / "images"
ANNOTATIONS_FILE = DATA_DIR / "annotations" / "instances_all_clean.json"
CHECKPOINTS_DIR = ROOT_DIR / "checkpoints"


# Visualization outputs
SAMPLES_DIR = ROOT_DIR / "artefacts" / "samples"

# ============================ #
# Training hyperparameters     #
# ============================ #

# Optimizer (SGD with momentum)
LEARNING_RATE = 1e-3
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4

# Batch size per GPU (total batch = BATCH_SIZE * num_gpus)
BATCH_SIZE = 32

# Number of training epochs
NUM_EPOCHS = 100

# DataLoader
NUM_WORKERS = 8                      # per GPU

# Learning rate schedule
WARMUP_EPOCHS = 3                    # linear warmup at the start
MIN_LR = 1e-5                        # lower bound after cosine decay

# Validation and checkpointing
CHECKPOINT_EVERY = 5                 # save a checkpoint every N epochs
LOG_EVERY = 50                       # log training metrics every N steps

# Output directories
LOGS_DIR = ROOT_DIR / "logs"

