"""Shared configuration for the TensorStego research demonstration."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"

CLEAN_MODEL = MODEL_DIR / "clean_model.pt"
STEGO_MODEL = MODEL_DIR / "stego_model.pt"

TARGET_TENSOR = "fc1.weight"

RANDOM_SEED = 1337
