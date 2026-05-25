import argparse
import inspect
import json
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    set_seed,
)

from src.config import (
    ROBERTA_MODEL_NAME,
    NLI_MODEL_NAME,
    MAX_LEN,
    BATCH_SIZE,
    EPOCHS,
    LEARNING_RATE,
    SEED,
    VAL_PATH,
    RESULTS_DIR,
    MODEL_DIR,
)


# =========================
# Dataset
# =========================

class TextDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = list(texts)
        self.labels = list(labels)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        text = str(self.texts[index])
        label = int(self.labels[index])

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long),
        }


# =========================
# Utilities
# =========================

def print_disk_usage(path: Path, label: str) -> None:
    """
    Print disk usage for the filesystem containing path.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    total, used, free = shutil.disk_usage(path)

    print(f"\nDisk check — {label}")
    print(f"Path : {path}")
    print(f"Total: {total / (1024 ** 3):.2f} GB")
    print(f"Used : {used / (1024 ** 3):.2f} GB")
    print(f"Free : {free / (1024 ** 3):.2f} GB")


def check_free_space(path: Path, min_free_gb: float = 2.0) -> None:
    """
    Raise an error if there is not enough free space.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    _, _, free = shutil.disk_usage(path)
    free_gb = free / (1024 ** 3)

    if free_gb < min_free_gb:
        raise RuntimeError(
            f"Not enough disk space at {path}.\n"
            f"Free space: {free_gb:.2f} GB\n"
            f"Required minimum: {min_free_gb:.2f} GB\n"
            f"Move MODEL_DIR and RESULTS_DIR to /opt/dlami/nvme."
        )


def load_dataset(path: Path) -> pd.DataFrame:
    print(f"Loading dataset from: {path}")

    df = pd.read_csv(path)

    required_columns = ["text", "label"]
    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        raise ValueError(
            f"Missing required columns in {path}: {missing_columns}\n"
            f"Available columns: {list(df.columns)}"
        )

    df = df.dropna(subset=["text", "label"]).copy()
    df["text"] = df["text"].astype(str)
    df["label"] = df["label"].astype(int)

    print(f"Loaded {len(df)} rows.")
    print("Label distribution:")
    print(df["label"].value_counts().sort_index())

    return df


def make_replay_training_data(
    train_df: pd.DataFrame,
    replay_df: pd.DataFrame | None,
    replay_repeat: int,
) -> pd.DataFrame:
    if replay_df is None or replay_repeat <= 0:
        return train_df.copy()

    replay_repeated = pd.concat(
        [replay_df.copy() for _ in range(replay_repeat)],
        ignore_index=True,
    )

    combined_df = pd.concat(
        [train_df.copy(), replay_repeated],
        ignore_index=True,
    )

    combined_df = combined_df.sample(
        frac=1.0,
        random_state=SEED,
    ).reset_index(drop=True)

    print("\nExperience replay data summary:")
    print(f"Original train rows       : {len(train_df)}")
    print(f"Replay rows               : {len(replay_df)}")
    print(f"Replay repeat             : {replay_repeat}")
    print(f"Replay rows after repeat  : {len(replay_repeated)}")
    print(f"Combined train rows       : {len(combined_df)}")

    print("\nCombined label distribution:")
    print(combined_df["label"].value_counts().sort_index())

    return combined_df


# =========================
# Metrics
# =========================

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=1)

    return {
        "accuracy": accuracy_score(labels, predictions),
        "macro_f1": f1_score(
            labels,
            predictions,
            average="macro",
            zero_division=0,
        ),
        "precision": precision_score(
            labels,
            predictions,
            average="macro",
            zero_division=0,
        ),
        "recall": recall_score(
            labels,
            predictions,
            average="macro",
            zero_division=0,
        ),
    }


# =========================
# Trainer Setup
# =========================

def build_training_args(output_model_dir: Path):
    """
    Build TrainingArguments safely.

    Important:
    save_strategy='no' prevents Hugging Face Trainer from saving checkpoint folders
    during training. This avoids saving optimizer.pt, which was causing disk errors.

    The final model is saved manually after trainer.train().
    """
    kwargs = {
        "output_dir": str(output_model_dir),

        # CRITICAL FIX:
        # Do not save checkpoint-* folders during training.
        # This prevents optimizer.pt and scheduler.pt from filling disk.
        "save_strategy": "no",

        # Training settings.
        "learning_rate": LEARNING_RATE,
        "per_device_train_batch_size": BATCH_SIZE,
        "per_device_eval_batch_size": BATCH_SIZE,
        "num_train_epochs": EPOCHS,
        "weight_decay": 0.01,

        # Must be False when save_strategy='no'.
        "load_best_model_at_end": False,

        # Logging.
        "logging_steps": 20,
        "report_to": "none",
        "seed": SEED,
    }

    signature = inspect.signature(TrainingArguments.__init__)
    parameters = signature.parameters

    # Transformers version compatibility:
    # Newer versions use eval_strategy, older versions use evaluation_strategy.
    if "eval_strategy" in parameters:
        kwargs["eval_strategy"] = "epoch"
    else:
        kwargs["evaluation_strategy"] = "epoch"

    return TrainingArguments(**kwargs)


def build_trainer(
    model,
    training_args,
    train_dataset,
    val_dataset,
    tokenizer,
):
    kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "eval_dataset": val_dataset,
        "compute_metrics": compute_metrics,
    }

    signature = inspect.signature(Trainer.__init__)
    parameters = signature.parameters

    # Transformers version compatibility.
    if "processing_class" in parameters:
        kwargs["processing_class"] = tokenizer
    else:
        kwargs["tokenizer"] = tokenizer

    return Trainer(**kwargs)


# =========================
# Model Selection
# =========================

def get_model_name(model_family: str, model_name: str | None) -> str:
    if model_name:
        return model_name

    if model_family == "roberta":
        return ROBERTA_MODEL_NAME

    if model_family == "nli":
        return NLI_MODEL_NAME

    raise ValueError(
        f"Unknown model_family: {model_family}. "
        "Use model_family='roberta' or model_family='nli'."
    )


# =========================
# Main Training Function
# =========================

def train_transformer(
    experiment_id: str,
    train_file: Path,
    model_family: str,
    model_name: str | None = None,
    replay_file: Path | None = None,
    replay_repeat: int = 0,
):
    set_seed(SEED)

    selected_model_name = get_model_name(model_family, model_name)

    output_model_dir = MODEL_DIR / experiment_id
    metrics_dir = RESULTS_DIR / "metrics"

    output_model_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 80)
    print("TRAINING START")
    print("=" * 80)

    print("\nEnvironment:")
    print(f"Current working directory : {Path.cwd()}")
    print(f"HF_HOME                   : {os.environ.get('HF_HOME', 'not set')}")
    print(f"TRANSFORMERS_CACHE        : {os.environ.get('TRANSFORMERS_CACHE', 'not set')}")

    print("\nOutput paths:")
    print(f"MODEL_DIR                 : {MODEL_DIR}")
    print(f"RESULTS_DIR               : {RESULTS_DIR}")
    print(f"Output model directory    : {output_model_dir}")
    print(f"Metrics directory         : {metrics_dir}")

    print_disk_usage(MODEL_DIR, "MODEL_DIR")
    print_disk_usage(RESULTS_DIR, "RESULTS_DIR")

    # Require some free disk before training.
    # RoBERTa final model is usually around hundreds of MB.
    check_free_space(MODEL_DIR, min_free_gb=2.0)
    check_free_space(RESULTS_DIR, min_free_gb=0.5)

    train_df = load_dataset(train_file)
    val_df = load_dataset(VAL_PATH)

    replay_df = None
    if replay_file is not None:
        replay_df = load_dataset(replay_file)

    final_train_df = make_replay_training_data(
        train_df=train_df,
        replay_df=replay_df,
        replay_repeat=replay_repeat,
    )

    print("\nTraining configuration:")
    print(f"Experiment ID      : {experiment_id}")
    print(f"Model family       : {model_family}")
    print(f"Model name         : {selected_model_name}")
    print(f"Train file         : {train_file}")
    print(f"Validation file    : {VAL_PATH}")
    print(f"Replay file        : {replay_file}")
    print(f"Replay repeat      : {replay_repeat}")
    print(f"Max length         : {MAX_LEN}")
    print(f"Batch size         : {BATCH_SIZE}")
    print(f"Epochs             : {EPOCHS}")
    print(f"Learning rate      : {LEARNING_RATE}")
    print(f"Seed               : {SEED}")

    print("\nLoading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(selected_model_name)

    model = AutoModelForSequenceClassification.from_pretrained(
        selected_model_name,
        num_labels=2,
        ignore_mismatched_sizes=True,
    )

    train_dataset = TextDataset(
        texts=final_train_df["text"],
        labels=final_train_df["label"],
        tokenizer=tokenizer,
        max_len=MAX_LEN,
    )

    val_dataset = TextDataset(
        texts=val_df["text"],
        labels=val_df["label"],
        tokenizer=tokenizer,
        max_len=MAX_LEN,
    )

    training_args = build_training_args(output_model_dir)

    print("\nTrainingArguments check:")
    print(f"save_strategy          : {training_args.save_strategy}")
    print(f"load_best_model_at_end : {training_args.load_best_model_at_end}")
    print("Expected behavior      : no checkpoint-* folders during training")

    trainer = build_trainer(
        model=model,
        training_args=training_args,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        tokenizer=tokenizer,
    )

    print("\nStarting fine-tuning...")
    train_result = trainer.train()

    print("\nTraining completed.")
    print(train_result)

    # Check space before final model save.
    print_disk_usage(MODEL_DIR, "MODEL_DIR before final save")
    check_free_space(MODEL_DIR, min_free_gb=2.0)

    print("\nSaving final model and tokenizer only...")
    trainer.save_model(output_model_dir)
    tokenizer.save_pretrained(output_model_dir)

    print(f"Saved final model to: {output_model_dir}")

    # Check whether checkpoint folders were accidentally created.
    checkpoint_dirs = list(output_model_dir.glob("checkpoint-*"))
    if checkpoint_dirs:
        print("\nWarning: checkpoint folders were found even though save_strategy='no'.")
        print("Removing checkpoint folders to save disk:")
        for checkpoint_dir in checkpoint_dirs:
            print(f"Removing: {checkpoint_dir}")
            shutil.rmtree(checkpoint_dir, ignore_errors=True)
    else:
        print("\nNo checkpoint folders found. Good.")

    train_log = {
        "experiment_id": experiment_id,
        "model_family": model_family,
        "model_name": selected_model_name,
        "train_file": str(train_file),
        "validation_file": str(VAL_PATH),
        "replay_file": str(replay_file) if replay_file is not None else None,
        "replay_repeat": replay_repeat,
        "original_train_rows": len(train_df),
        "final_train_rows": len(final_train_df),
        "max_len": MAX_LEN,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "learning_rate": LEARNING_RATE,
        "seed": SEED,
        "save_strategy": str(training_args.save_strategy),
        "load_best_model_at_end": training_args.load_best_model_at_end,
        "log_history": trainer.state.log_history,
    }

    train_log_path = metrics_dir / f"{experiment_id}_train_log.json"

    with open(train_log_path, "w", encoding="utf-8") as file:
        json.dump(train_log, file, indent=4)

    print(f"Saved training log to: {train_log_path}")

    print_disk_usage(MODEL_DIR, "MODEL_DIR after training")
    print_disk_usage(RESULTS_DIR, "RESULTS_DIR after training")

    print("\nTraining finished successfully.")


# =========================
# CLI
# =========================

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--experiment_id", type=str, required=True)
    parser.add_argument("--train_file", type=str, required=True)

    parser.add_argument(
        "--model_family",
        type=str,
        choices=["roberta", "nli"],
        required=True,
    )

    parser.add_argument(
        "--model_name",
        type=str,
        default=None,
        help="Optional custom model checkpoint.",
    )

    parser.add_argument(
        "--replay_file",
        type=str,
        default=None,
        help="Optional replay examples CSV.",
    )

    parser.add_argument(
        "--replay_repeat",
        type=int,
        default=0,
        help="How many times to repeat replay examples.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    replay_file = Path(args.replay_file) if args.replay_file else None

    train_transformer(
        experiment_id=args.experiment_id,
        train_file=Path(args.train_file),
        model_family=args.model_family,
        model_name=args.model_name,
        replay_file=replay_file,
        replay_repeat=args.replay_repeat,
    )


if __name__ == "__main__":
    main()