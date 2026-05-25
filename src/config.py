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

REPLAY_PATH = PROCESSED_DIR / "replay_examples.csv"
TRAIN_REPLAY_PATH = PROCESSED_DIR / "train_replay.csv"

# =========================
# Output Paths
# =========================

RESULTS_DIR = PROJECT_ROOT / "results"
MODEL_DIR = PROJECT_ROOT / "saved_models"

# =========================
# Model Settings
# =========================

ROBERTA_MODEL_NAME = "roberta-base"
NLI_MODEL_NAME = "cross-encoder/nli-roberta-base"

# Backward compatibility for older files that still import MODEL_NAME
MODEL_NAME = ROBERTA_MODEL_NAME

# =========================
# Training Settings
# =========================

MAX_LEN = 128
BATCH_SIZE = 16
EPOCHS = 10
LEARNING_RATE = 2e-5
SEED = 42

# =========================
# Label Settings
# =========================

LABEL2ID = {
    "non-risky": 0,
    "risky": 1,
}

ID2LABEL = {
    0: "non-risky",
    1: "risky",
}