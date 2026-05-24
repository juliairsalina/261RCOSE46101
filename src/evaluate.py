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

from transformers import AutoTokenizer, AutoModelForSequenceClassification

from src.config import (
    MAX_LEN,
    TEST_PATH,
    OOD_PATH,
    RESULTS_DIR,
)


SAVED_MODELS_DIR = Path("saved_models")
CONFIDENCE_THRESHOLD = 0.80


def load_dataset(path):
    """
    Load evaluation CSV dataset.
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


def predict_dataset(model, tokenizer, df, device):
    """
    Predict labels and probabilities for a dataset.
    """
    model.eval()

    predicted_labels = []
    risky_probs = []
    confidences = []

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
            probabilities = torch.softmax(outputs.logits, dim=1)

        risky_prob = probabilities[0][1].item()
        confidence = torch.max(probabilities, dim=1).values.item()
        predicted_label = torch.argmax(probabilities, dim=1).item()

        risky_probs.append(risky_prob)
        confidences.append(confidence)
        predicted_labels.append(int(predicted_label))

    return predicted_labels, risky_probs, confidences


def compute_basic_metrics(true_labels, predicted_labels, confidences):
    """
    Compute standard evaluation metrics.
    """
    correct_array = np.array(true_labels) == np.array(predicted_labels)
    confidence_array = np.array(confidences)

    confident_wrong_count = int(
        np.sum((correct_array == False) & (confidence_array >= CONFIDENCE_THRESHOLD))
    )

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
        "average_confidence": float(np.mean(confidences)),
        "confident_wrong_count": confident_wrong_count,
        "confidence_threshold": CONFIDENCE_THRESHOLD,
    }

    return metrics


def compute_category_metrics(df, predicted_labels):
    """
    For OOD dataset:
    If category column exists, compute category-wise accuracy and macro-F1.
    """
    if "category" not in df.columns:
        return None

    category_results = {}

    temp_df = df.copy()
    temp_df["predicted_label"] = predicted_labels

    for category, group in temp_df.groupby("category"):
        true_labels = group["label"].tolist()
        category_predictions = group["predicted_label"].tolist()

        category_results[str(category)] = {
            "count": int(len(group)),
            "accuracy": accuracy_score(true_labels, category_predictions),
            "macro_f1": f1_score(
                true_labels,
                category_predictions,
                average="macro",
                zero_division=0,
            ),
        }

    return category_results


def create_predictions_dataframe(df, predicted_labels, risky_probs, confidences):
    """
    Create predictions CSV dataframe.
    """
    true_labels = df["label"].tolist()

    predictions_df = pd.DataFrame(
        {
            "text": df["text"],
            "true_label": true_labels,
            "predicted_label": predicted_labels,
            "risky_prob": risky_probs,
            "confidence": confidences,
            "correct": np.array(true_labels) == np.array(predicted_labels),
        }
    )

    if "category" in df.columns:
        predictions_df["category"] = df["category"]

    return predictions_df


def evaluate_one_dataset(
    model,
    tokenizer,
    df,
    device,
    predictions_output_path,
    compute_categories=False,
):
    """
    Evaluate one dataset and save prediction CSV.
    """
    true_labels = df["label"].tolist()

    predicted_labels, risky_probs, confidences = predict_dataset(
        model=model,
        tokenizer=tokenizer,
        df=df,
        device=device,
    )

    metrics = compute_basic_metrics(
        true_labels=true_labels,
        predicted_labels=predicted_labels,
        confidences=confidences,
    )

    if compute_categories:
        category_metrics = compute_category_metrics(df, predicted_labels)

        if category_metrics is not None:
            metrics["category_metrics"] = category_metrics

    predictions_df = create_predictions_dataframe(
        df=df,
        predicted_labels=predicted_labels,
        risky_probs=risky_probs,
        confidences=confidences,
    )

    predictions_output_path.parent.mkdir(parents=True, exist_ok=True)
    predictions_df.to_csv(predictions_output_path, index=False)

    print(f"Saved predictions to: {predictions_output_path}")

    return metrics


def evaluate_experiment(experiment_id):
    """
    Load saved model and evaluate on ID test set and OOD set.
    """
    model_path = SAVED_MODELS_DIR / experiment_id

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model directory not found: {model_path}\n"
            f"Please train first using: python -m src.train_roberta"
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")
    print(f"Loading model from: {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.to(device)

    test_df = load_dataset(TEST_PATH)
    ood_df = load_dataset(OOD_PATH)

    predictions_dir = RESULTS_DIR / "predictions"
    metrics_dir = RESULTS_DIR / "metrics"

    id_predictions_path = predictions_dir / f"{experiment_id}_id_predictions.csv"
    ood_predictions_path = predictions_dir / f"{experiment_id}_ood_predictions.csv"

    print("\nEvaluating ID test set...")
    id_metrics = evaluate_one_dataset(
        model=model,
        tokenizer=tokenizer,
        df=test_df,
        device=device,
        predictions_output_path=id_predictions_path,
        compute_categories=False,
    )

    print("\nEvaluating OOD set...")
    ood_metrics = evaluate_one_dataset(
        model=model,
        tokenizer=tokenizer,
        df=ood_df,
        device=device,
        predictions_output_path=ood_predictions_path,
        compute_categories=True,
    )

    all_metrics = {
        "experiment_id": experiment_id,
        "model_path": str(model_path),
        "id_test": id_metrics,
        "ood": ood_metrics,
    }

    metrics_dir.mkdir(parents=True, exist_ok=True)

    metrics_output_path = metrics_dir / f"{experiment_id}.json"

    with open(metrics_output_path, "w", encoding="utf-8") as file:
        json.dump(all_metrics, file, indent=4)

    print(f"Saved metrics to: {metrics_output_path}")

    print("\nEvaluation summary:")
    print(f"ID accuracy: {id_metrics['accuracy']:.4f}")
    print(f"ID macro F1: {id_metrics['macro_f1']:.4f}")
    print(f"OOD accuracy: {ood_metrics['accuracy']:.4f}")
    print(f"OOD macro F1: {ood_metrics['macro_f1']:.4f}")
    print(f"OOD confident wrong count: {ood_metrics['confident_wrong_count']}")


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--experiment_id",
        type=str,
        required=True,
        help="Experiment ID, for example E1, E2, E3, E4, or E5.",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    evaluate_experiment(experiment_id=args.experiment_id)


if __name__ == "__main__":
    main()