"""Shared configuration for the OCT agentic diagnostic pipeline."""
from __future__ import annotations

import os
import platform
import random
from pathlib import Path

# Must be set before PyTorch initializes CUDA — reduces fragmentation on the
# 6GB RTX 3060 during per-image GradCAM calls.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch

# ── Paths (resolved relative to the package, e.g. /home/nebula/projects) ──────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT    = PROJECT_ROOT / "OCT2017"
TRAIN_DIR    = DATA_ROOT / "train"
TEST_DIR     = DATA_ROOT / "test"
OCTC8_ROOT   = PROJECT_ROOT / "octc8"
OCTID_ROOT   = PROJECT_ROOT / "octid"
KB_PATH      = PROJECT_ROOT / "oct_corpus_xl.json"

CKPT_PATH   = PROJECT_ROOT / "oct_best.pth"         # from-scratch OCTNet
CKPT_EMB    = PROJECT_ROOT / "retrieval_encoder.pth"  # from-scratch retrieval encoder

# ── Model / training constants ────────────────────────────────────────────────
IMG_SIZE        = 224
BATCH_SIZE      = 8
NUM_CLASSES     = 4
CLASS_NAMES     = ["CNV", "DME", "DRUSEN", "NORMAL"]

EPOCHS          = 60
LR              = 3e-4
WEIGHT_DECAY    = 1e-4
LABEL_SMOOTHING = 0.05
WARMUP_EPOCHS   = 5
GRAD_CLIP       = 1.0

EMB_EPOCHS      = 80
EMB_LR          = 3e-4
EMB_TEMPERATURE = 0.07
EMB_DIM         = 64
EMB_MAX_LEN     = 64

# ── Runtime ───────────────────────────────────────────────────────────────────
SEED = 42
NUM_WORKERS = 0 if platform.system() == "Windows" else 4
PIN_MEMORY  = not (platform.system() == "Windows") and torch.cuda.is_available()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def autocast_ctx():
    """torch.amp autocast context, enabled only on CUDA."""
    if torch.cuda.is_available():
        return torch.amp.autocast("cuda")
    return torch.amp.autocast("cpu", enabled=False)


def get_scaler():
    return torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())


def allocate_device() -> torch.device:
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")