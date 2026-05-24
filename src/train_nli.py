import argparse
import json
import inspect
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
    MAX_LEN,
    BATCH_SIZE,
    EPOCHS,
    LEARNING_RATE,
    SEED,
    VAL_PATH,
    RESULTS_DIR,
)


SAVED_MODELS_DIR = Path("saved_models")
DEFAULT_NLI_MODEL = "cross-encoder/nli-roberta-base"


class TextDataset(torch.utils.data.Dataset):
    """
    Simple PyTorch Dataset for binary risky-intent classification.
    """

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


def load_dataset(path):
    """
    Load CSV dataset.

    Required columns:
    - text
    - label

    Labels must already be numeric:
    0 = non-risky
    1 = risky
    """
    print(f"Loading dataset from: {path}")

    df = pd.read_csv(path)

    required_columns = ["text", "label"]
    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        raise ValueError(f"Missing required columns in {path}: {missing_columns}")

    df = df.dropna(subset=["text", "label"]).copy()
    df["text"] = df["text"].astype(str)
    df["label"] = df["label"].astype(int)

    print(f"Loaded {len(df)} rows.")

    return df


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=1)

    return {
        "accuracy": accuracy_score(labels, predictions),
        "macro_f1": f1_score(labels, predictions, average="macro"),
        "precision": precision_score(labels, predictions, average="macro", zero_division=0),
        "recall": recall_score(labels, predictions, average="macro", zero_division=0),
    }


def build_training_args(output_model_dir):
    """
    Handles Transformers version differences:
    - older versions use evaluation_strategy
    - newer versions use eval_strategy
    """
    training_args_kwargs = {
        "output_dir": str(output_model_dir),
        "save_strategy": "epoch",
        "learning_rate": LEARNING_RATE,
        "per_device_train_batch_size": BATCH_SIZE,
        "per_device_eval_batch_size": BATCH_SIZE,
        "num_train_epochs": EPOCHS,
        "weight_decay": 0.01,
        "load_best_model_at_end": True,
        "metric_for_best_model": "macro_f1",
        "greater_is_better": True,
        "logging_steps": 20,
        "save_total_limit": 2,
        "report_to": "none",
        "seed": SEED,
    }

    signature = inspect.signature(TrainingArguments.__init__)

    if "eval_strategy" in signature.parameters:
        training_args_kwargs["eval_strategy"] = "epoch"
    else:
        training_args_kwargs["evaluation_strategy"] = "epoch"

    return TrainingArguments(**training_args_kwargs)


def build_trainer(
    model,
    training_args,
    train_dataset,
    val_dataset,
    tokenizer,
):
    """
    Handles Transformers version differences:
    - older versions use tokenizer=
    - newer versions use processing_class=
    """
    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "eval_dataset": val_dataset,
        "compute_metrics": compute_metrics,
    }

    signature = inspect.signature(Trainer.__init__)

    if "processing_class" in signature.parameters:
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer

    return Trainer(**trainer_kwargs)


def train_nli(
    experiment_id,
    train_file,
    model_name,
):
    """
    E7:
    NLI initialization + keyword masking.

    This uses an NLI-trained RoBERTa checkpoint as initialization,
    but fine-tunes it for binary risky/non-risky classification.
    """
    set_seed(SEED)

    train_df = load_dataset(train_file)
    val_df = load_dataset(VAL_PATH)

    print("\nTraining configuration:")
    print(f"Experiment ID: {experiment_id}")
    print(f"NLI model name: {model_name}")
    print(f"Train file: {train_file}")
    print(f"Validation file: {VAL_PATH}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
        ignore_mismatched_sizes=True,
    )

    train_dataset = TextDataset(
        texts=train_df["text"],
        labels=train_df["label"],
        tokenizer=tokenizer,
        max_len=MAX_LEN,
    )

    val_dataset = TextDataset(
        texts=val_df["text"],
        labels=val_df["label"],
        tokenizer=tokenizer,
        max_len=MAX_LEN,
    )

    output_model_dir = SAVED_MODELS_DIR / experiment_id
    output_model_dir.mkdir(parents=True, exist_ok=True)

    training_args = build_training_args(output_model_dir)

    trainer = build_trainer(
        model=model,
        training_args=training_args,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        tokenizer=tokenizer,
    )

    print("\nStarting NLI fine-tuning...")
    trainer.train()

    print("\nSaving model and tokenizer...")
    trainer.save_model(output_model_dir)
    tokenizer.save_pretrained(output_model_dir)

    print(f"Saved model to: {output_model_dir}")

    train_log = {
        "experiment_id": experiment_id,
        "experiment_name": "NLI + Keyword Masking",
        "model_name": model_name,
        "train_file": str(train_file),
        "validation_file": str(VAL_PATH),
        "max_len": MAX_LEN,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "learning_rate": LEARNING_RATE,
        "seed": SEED,
        "log_history": trainer.state.log_history,
    }

    metrics_dir = RESULTS_DIR / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    train_log_path = metrics_dir / f"{experiment_id}_train_log.json"

    with open(train_log_path, "w", encoding="utf-8") as file:
        json.dump(train_log, file, indent=4)

    print(f"Saved training log to: {train_log_path}")
    print("\nTraining finished.")


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--experiment_id",
        type=str,
        required=True,
        help="Experiment ID, for example E7.",
    )

    parser.add_argument(
        "--train_file",
        type=str,
        required=True,
        help="Training CSV file path. For E7, use data/processed/train_masked.csv.",
    )

    parser.add_argument(
        "--model_name",
        type=str,
        default=DEFAULT_NLI_MODEL,
        help="NLI model checkpoint.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    train_nli(
        experiment_id=args.experiment_id,
        train_file=Path(args.train_file),
        model_name=args.model_name,
    )


if __name__ == "__main__":
    main()