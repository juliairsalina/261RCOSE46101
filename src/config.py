from pathlib import Path

# =========================
# Project Root
# =========================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# =========================
# Data Paths
# =========================

RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "dataset_nad.xlsx"
RAW_OOD_PATH = PROJECT_ROOT / "data" / "raw" / "custom_ood_set_150_julia.csv"

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

TRAIN_PATH = PROCESSED_DIR / "train.csv"
VAL_PATH = PROCESSED_DIR / "val.csv"
TEST_PATH = PROCESSED_DIR / "test.csv"
OOD_PATH = PROCESSED_DIR / "ood.csv"

MASKED_TRAIN_PATH = PROCESSED_DIR / "train_masked.csv"
COUNTERFACTUAL_TRAIN_PATH = PROCESSED_DIR / "train_counterfactual.csv"
FULL_TRAIN_PATH = PROCESSED_DIR / "train_full.csv"

# =========================
# Output Paths
# =========================

RESULTS_DIR = PROJECT_ROOT / "results"
MODEL_DIR = PROJECT_ROOT / "models"

# =========================
# Training Settings
# =========================

MODEL_NAME = "roberta-base"

MAX_LEN = 128
BATCH_SIZE = 16
EPOCHS = 3
LEARNING_RATE = 2e-5
SEED = 42