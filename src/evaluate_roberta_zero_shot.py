import argparse
import json

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
    ROBERTA_MODEL_NAME,
    MAX_LEN,
    TEST_PATH,
    OOD_PATH,
    RESULTS_DIR,
    ID2LABEL,
)

CONFIDENCE_THRESHOLD = 0.80
LABEL_IDS = [0, 1]


def load_dataset(path):
    df = pd.read_csv(path)
    df = df.dropna(subset=["text", "label"]).copy()
    df["text"] = df["text"].astype(str)
    df["label"] = df["label"].astype(int)
    print(f"Loaded {len(df)} rows from {path}")
    return df


def predict_dataset(model, tokenizer, df, device):
    model.eval()
    predicted_labels, risky_probs, confidences = [], [], []

    for text in df["text"].tolist():
        encoding = tokenizer(
            str(text),
            truncation=True,
            padding="max_length",
            max_length=MAX_LEN,
            return_tensors="pt",
        )
        encoding = {k: v.to(device) for k, v in encoding.items()}

        with torch.no_grad():
            outputs = model(**encoding)
            probs = torch.softmax(outputs.logits, dim=1)

        risky_prob = probs[0][1].item()
        confidence = torch.max(probs, dim=1).values.item()
        predicted_label = torch.argmax(probs, dim=1).item()

        risky_probs.append(risky_prob)
        confidences.append(confidence)
        predicted_labels.append(int(predicted_label))

    return predicted_labels, risky_probs, confidences


def compute_metrics(true_labels, predicted_labels, confidences):
    correct_array = np.array(true_labels) == np.array(predicted_labels)
    confidence_array = np.array(confidences)
    confident_wrong_count = int(
        np.sum((correct_array == False) & (confidence_array >= CONFIDENCE_THRESHOLD))
    )

    return {
        "accuracy": accuracy_score(true_labels, predicted_labels),
        "macro_f1": f1_score(true_labels, predicted_labels, average="macro",
                             labels=LABEL_IDS, zero_division=0),
        "precision": precision_score(true_labels, predicted_labels, average="macro",
                                     labels=LABEL_IDS, zero_division=0),
        "recall": recall_score(true_labels, predicted_labels, average="macro",
                               labels=LABEL_IDS, zero_division=0),
        "classification_report": classification_report(
            true_labels, predicted_labels, labels=LABEL_IDS,
            target_names=[ID2LABEL[0], ID2LABEL[1]],
            output_dict=True, zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(
            true_labels, predicted_labels, labels=LABEL_IDS,
        ).tolist(),
        "average_confidence": float(np.mean(confidences)),
        "confident_wrong_count": confident_wrong_count,
        "confidence_threshold": CONFIDENCE_THRESHOLD,
    }


def evaluate_zero_shot_roberta(experiment_id):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Loading untrained RoBERTa from: {ROBERTA_MODEL_NAME}")

    # Load roberta-base with a RANDOM (untrained) classification head.
    # ignore_mismatched_sizes=True initializes the head with random weights.
    tokenizer = AutoTokenizer.from_pretrained(ROBERTA_MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        ROBERTA_MODEL_NAME,
        num_labels=2,
        ignore_mismatched_sizes=True,
    )
    model.to(device)

    test_df = load_dataset(TEST_PATH)
    ood_df = load_dataset(OOD_PATH)

    predictions_dir = RESULTS_DIR / "predictions"
    metrics_dir = RESULTS_DIR / "metrics"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    for split_name, df in [("id_test", test_df), ("ood", ood_df)]:
        print(f"\nEvaluating {split_name}...")
        predicted_labels, risky_probs, confidences = predict_dataset(
            model, tokenizer, df, device
        )
        true_labels = df["label"].tolist()
        metrics = compute_metrics(true_labels, predicted_labels, confidences)
        results[split_name] = metrics

        # Save predictions CSV
        preds_df = pd.DataFrame({
            "text": df["text"],
            "true_label": true_labels,
            "predicted_label": predicted_labels,
            "risky_prob": risky_probs,
            "confidence": confidences,
            "correct": np.array(true_labels) == np.array(predicted_labels),
        })
        for col in ["id", "keyword", "category"]:
            if col in df.columns:
                preds_df[col] = df[col].values

        preds_path = predictions_dir / f"{experiment_id}_{split_name}_predictions.csv"
        preds_df.to_csv(preds_path, index=False)
        print(f"Saved predictions: {preds_path}")

    all_metrics = {
        "experiment_id": experiment_id,
        "model": ROBERTA_MODEL_NAME,
        "note": "Zero-shot: randomly initialized classification head, no fine-tuning.",
        "id_test": results["id_test"],
        "ood": results["ood"],
    }

    metrics_path = metrics_dir / f"{experiment_id}.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=4)

    print(f"\nSaved metrics: {metrics_path}")
    print(f"ID  accuracy: {results['id_test']['accuracy']:.4f}  macro-F1: {results['id_test']['macro_f1']:.4f}")
    print(f"OOD accuracy: {results['ood']['accuracy']:.4f}  macro-F1: {results['ood']['macro_f1']:.4f}")
    print("Expected: ~0.50 accuracy (random baseline)")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_id", type=str, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    evaluate_zero_shot_roberta(experiment_id=args.experiment_id)


if __name__ == "__main__":
    main()