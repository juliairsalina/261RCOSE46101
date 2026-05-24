import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix,
)

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
    FULL_TRAIN_PATH,
    VAL_PATH,
    TEST_PATH,
    OOD_PATH,
    RESULTS_DIR,
    MODEL_DIR,
)


class TextDataset(torch.utils.data.Dataset):
    """
    Simple PyTorch dataset for text classification.
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
    Trainer with confidence regularization.

    Normal RoBERTa can become too confident on shortcut words.

    This adds an entropy regularization term:
    - Cross entropy teaches the correct label.
    - Entropy regularization discourages extreme overconfidence.

    Total loss:
        loss = classification_loss - lambda * entropy

    A small lambda is important.
    """

    def __init__(self, confidence_lambda=0.01, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.confidence_lambda = confidence_lambda

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")

        outputs = model(**inputs)
        logits = outputs.get("logits")

        classification_loss = torch.nn.functional.cross_entropy(logits, labels)

        probs = torch.softmax(logits, dim=-1)
        log_probs = torch.log_softmax(logits, dim=-1)

        entropy = -torch.sum(probs * log_probs, dim=-1).mean()

        loss = classification_loss - self.confidence_lambda * entropy

        if return_outputs:
            return loss, outputs

        return loss


def load_dataset(path):
    df = pd.read_csv(path)

    required_columns = ["text", "label"]
    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        raise ValueError(f"Missing columns in {path}: {missing_columns}")

    df = df.dropna(subset=["text", "label"]).copy()
    df["text"] = df["text"].astype(str)
    df["label"] = df["label"].astype(int)

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


def evaluate_and_save_predictions(model, tokenizer, df, output_path, device):
    model.eval()

    true_labels = df["label"].tolist()

    predicted_labels = []
    risky_probs = []

    for text in df["text"].tolist():
        encoding = tokenizer(
            str(text),
            truncation=True,
            padding="max_length",
            max_length=MAX_LEN,
            return_tensors="pt",
        )

        encoding = {key: value.to(device) for key, value in encoding.items()}

        with torch.no_grad():
            outputs = model(**encoding)
            probs = torch.softmax(outputs.logits, dim=1)

        risky_prob = probs[0][1].item()
        predicted_label = int(torch.argmax(probs, dim=1).item())

        risky_probs.append(risky_prob)
        predicted_labels.append(predicted_label)

    metrics = {
        "accuracy": accuracy_score(true_labels, predicted_labels),
        "macro_f1": f1_score(true_labels, predicted_labels, average="macro"),
        "precision": precision_score(true_labels, predicted_labels, average="macro", zero_division=0),
        "recall": recall_score(true_labels, predicted_labels, average="macro", zero_division=0),
        "classification_report": classification_report(
            true_labels,
            predicted_labels,
            target_names=["non-risky", "risky"],
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(true_labels, predicted_labels).tolist(),
    }

    predictions_df = pd.DataFrame(
        {
            "text": df["text"],
            "true_label": true_labels,
            "predicted_label": predicted_labels,
            "risky_prob": risky_probs,
            "correct": np.array(true_labels) == np.array(predicted_labels),
        }
    )

    if "category" in df.columns:
        predictions_df["category"] = df["category"]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    predictions_df.to_csv(output_path, index=False)

    print(f"Saved predictions to: {output_path}")

    return metrics


def train_with_confidence_regularization(
    experiment_id,
    train_path,
    val_path,
    test_path,
    ood_path,
    confidence_lambda,
):
    set_seed(SEED)

    train_df = load_dataset(train_path)
    val_df = load_dataset(val_path)
    test_df = load_dataset(test_path)
    ood_df = load_dataset(ood_path)

    print(f"Training rows: {len(train_df)}")
    print(f"Validation rows: {len(val_df)}")
    print(f"Test rows: {len(test_df)}")
    print(f"OOD rows: {len(ood_df)}")
    print(f"Confidence lambda: {confidence_lambda}")

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

    output_model_dir = MODEL_DIR / experiment_id

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
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        confidence_lambda=confidence_lambda,
    )

    trainer.train()

    output_model_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(output_model_dir)
    tokenizer.save_pretrained(output_model_dir)

    print(f"Saved model to: {output_model_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    predictions_dir = RESULTS_DIR / "predictions"
    metrics_dir = RESULTS_DIR / "metrics"

    id_predictions_path = predictions_dir / f"{experiment_id}_id_predictions.csv"
    ood_predictions_path = predictions_dir / f"{experiment_id}_ood_predictions.csv"

    print("\nEvaluating ID test set...")
    id_metrics = evaluate_and_save_predictions(
        model=model,
        tokenizer=tokenizer,
        df=test_df,
        output_path=id_predictions_path,
        device=device,
    )

    print("\nEvaluating OOD set...")
    ood_metrics = evaluate_and_save_predictions(
        model=model,
        tokenizer=tokenizer,
        df=ood_df,
        output_path=ood_predictions_path,
        device=device,
    )

    all_metrics = {
        "experiment_id": experiment_id,
        "model": MODEL_NAME,
        "train_path": str(train_path),
        "confidence_lambda": confidence_lambda,
        "id_test": id_metrics,
        "ood": ood_metrics,
    }

    metrics_path = metrics_dir / f"{experiment_id}_confidence_regularization.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    with open(metrics_path, "w", encoding="utf-8") as file:
        json.dump(all_metrics, file, indent=4)

    print(f"Saved metrics to: {metrics_path}")

    print("\nDone.")
    print(f"ID macro F1: {id_metrics['macro_f1']:.4f}")
    print(f"OOD macro F1: {ood_metrics['macro_f1']:.4f}")


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--experiment_id", type=str, default="E5")

    parser.add_argument("--train_path", type=str, default=str(FULL_TRAIN_PATH))
    parser.add_argument("--val_path", type=str, default=str(VAL_PATH))
    parser.add_argument("--test_path", type=str, default=str(TEST_PATH))
    parser.add_argument("--ood_path", type=str, default=str(OOD_PATH))

    parser.add_argument(
        "--confidence_lambda",
        type=float,
        default=0.01,
        help="Strength of confidence regularization.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    train_with_confidence_regularization(
        experiment_id=args.experiment_id,
        train_path=Path(args.train_path),
        val_path=Path(args.val_path),
        test_path=Path(args.test_path),
        ood_path=Path(args.ood_path),
        confidence_lambda=args.confidence_lambda,
    )


if __name__ == "__main__":
    main()