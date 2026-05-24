import argparse
import json
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
    MODEL_NAME,
    MAX_LEN,
    BATCH_SIZE,
    EPOCHS,
    LEARNING_RATE,
    SEED,
    VAL_PATH,
    RESULTS_DIR,
)


SAVED_MODELS_DIR = Path("saved_models")


class TextDataset(torch.utils.data.Dataset):
    """
    Simple PyTorch Dataset for text classification.
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


class ConfidenceRegularizationTrainer(Trainer):
    """
    Custom Trainer for confidence regularization.

    Normal loss:
        cross entropy loss

    Confidence regularized loss:
        final_loss = cross_entropy_loss - lambda_conf * entropy_mean

    Higher entropy means the model is less overconfident.
    """

    def __init__(
        self,
        lambda_conf=0.05,
        use_confidence_regularization=False,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.lambda_conf = lambda_conf
        self.use_confidence_regularization = use_confidence_regularization

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")

        outputs = model(**inputs)
        logits = outputs.get("logits")

        cross_entropy_loss = torch.nn.functional.cross_entropy(logits, labels)

        if self.use_confidence_regularization:
            probabilities = torch.softmax(logits, dim=-1)
            log_probabilities = torch.log_softmax(logits, dim=-1)

            entropy = -torch.sum(probabilities * log_probabilities, dim=-1)
            entropy_mean = entropy.mean()

            final_loss = cross_entropy_loss - self.lambda_conf * entropy_mean
        else:
            final_loss = cross_entropy_loss

        if return_outputs:
            return final_loss, outputs

        return final_loss


def str_to_bool(value):
    """
    Convert command line string to boolean.

    Example:
        "true" -> True
        "false" -> False
    """
    if isinstance(value, bool):
        return value

    value = value.lower()

    if value in ["true", "1", "yes", "y"]:
        return True

    if value in ["false", "0", "no", "n"]:
        return False

    raise argparse.ArgumentTypeError("Boolean value expected: true or false")


def load_dataset(path):
    """
    Load CSV dataset.

    Required columns:
    - text
    - label
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
    """
    Metrics used during validation.
    """
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=1)

    return {
        "accuracy": accuracy_score(labels, predictions),
        "macro_f1": f1_score(labels, predictions, average="macro"),
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


def train_roberta(
    experiment_id,
    train_file,
    confidence_regularization,
    lambda_conf,
):
    """
    Train RoBERTa model and save it to saved_models/{experiment_id}/.
    """
    set_seed(SEED)

    train_df = load_dataset(train_file)
    val_df = load_dataset(VAL_PATH)

    print("\nTraining configuration:")
    print(f"Experiment ID: {experiment_id}")
    print(f"Model name: {MODEL_NAME}")
    print(f"Train file: {train_file}")
    print(f"Confidence regularization: {confidence_regularization}")
    print(f"Lambda confidence: {lambda_conf}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,
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

    training_args = TrainingArguments(
        output_dir=str(output_model_dir),
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=EPOCHS,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        logging_steps=20,
        save_total_limit=2,
        report_to="none",
        seed=SEED,
    )

    trainer = ConfidenceRegularizationTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
        lambda_conf=lambda_conf,
        use_confidence_regularization=confidence_regularization,
    )

    print("\nStarting training...")
    trainer.train()

    print("\nSaving model and tokenizer...")
    trainer.save_model(output_model_dir)
    tokenizer.save_pretrained(output_model_dir)

    print(f"Saved model to: {output_model_dir}")

    train_log = {
        "experiment_id": experiment_id,
        "model_name": MODEL_NAME,
        "train_file": str(train_file),
        "validation_file": str(VAL_PATH),
        "confidence_regularization": confidence_regularization,
        "lambda_conf": lambda_conf,
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
        help="Experiment ID, for example E1, E2, E3, E4, or E5.",
    )

    parser.add_argument(
        "--train_file",
        type=str,
        required=True,
        help="Training CSV file path.",
    )

    parser.add_argument(
        "--confidence_regularization",
        type=str_to_bool,
        default=False,
        help="Use confidence regularization: true or false.",
    )

    parser.add_argument(
        "--lambda_conf",
        type=float,
        default=0.05,
        help="Lambda value for confidence regularization.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    train_roberta(
        experiment_id=args.experiment_id,
        train_file=Path(args.train_file),
        confidence_regularization=args.confidence_regularization,
        lambda_conf=args.lambda_conf,
    )


if __name__ == "__main__":
    main()