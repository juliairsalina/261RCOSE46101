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
    NLI_MODEL_NAME,
    MAX_LEN,
    TEST_PATH,
    OOD_PATH,
    RESULTS_DIR,
    ID2LABEL,
)


RISKY_HYPOTHESIS = "This text expresses risky intent."
NON_RISKY_HYPOTHESIS = "This text does not express risky intent."


def load_dataset(path):
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
    return df


def find_entailment_index(model):
    id2label = model.config.id2label

    for idx, label in id2label.items():
        if "entail" in str(label).lower():
            return int(idx)

    # fallback for common 3-class NLI heads
    if model.config.num_labels == 3:
        return 2

    raise ValueError(f"Could not find entailment label from id2label: {id2label}")


def get_entailment_score(text, hypothesis, tokenizer, model, device, entailment_index):
    encoding = tokenizer(
        text,
        hypothesis,
        truncation=True,
        padding="max_length",
        max_length=MAX_LEN,
        return_tensors="pt",
    )

    encoding = {key: value.to(device) for key, value in encoding.items()}

    with torch.no_grad():
        outputs = model(**encoding)
        probs = torch.softmax(outputs.logits, dim=1)

    return probs[0][entailment_index].item()


def predict_zero_shot(df, tokenizer, model, device, entailment_index):
    predicted_labels = []
    risky_scores = []
    non_risky_scores = []
    confidences = []

    for text in df["text"].tolist():
        risky_score = get_entailment_score(
            text=text,
            hypothesis=RISKY_HYPOTHESIS,
            tokenizer=tokenizer,
            model=model,
            device=device,
            entailment_index=entailment_index,
        )

        non_risky_score = get_entailment_score(
            text=text,
            hypothesis=NON_RISKY_HYPOTHESIS,
            tokenizer=tokenizer,
            model=model,
            device=device,
            entailment_index=entailment_index,
        )

        if risky_score >= non_risky_score:
            predicted_label = 1
        else:
            predicted_label = 0

        confidence = float(max(risky_score, non_risky_score))

        predicted_labels.append(predicted_label)
        risky_scores.append(risky_score)
        non_risky_scores.append(non_risky_score)
        confidences.append(confidence)

    return predicted_labels, risky_scores, non_risky_scores, confidences


def compute_metrics(true_labels, predicted_labels, confidences):
    correct_array = np.array(true_labels) == np.array(predicted_labels)
    confidence_array = np.array(confidences)

    confident_wrong_count = int(
        np.sum((correct_array == False) & (confidence_array >= 0.80))
    )

    return {
        "accuracy": accuracy_score(true_labels, predicted_labels),
        "macro_f1": f1_score(true_labels, predicted_labels, average="macro"),
        "precision": precision_score(
            true_labels,
            predicted_labels,
            average="macro",
            zero_division=0,
        ),
        "recall": recall_score(
            true_labels,
            predicted_labels,
            average="macro",
            zero_division=0,
        ),
        "classification_report": classification_report(
            true_labels,
            predicted_labels,
            target_names=[ID2LABEL[0], ID2LABEL[1]],
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(true_labels, predicted_labels).tolist(),
        "average_confidence": float(np.mean(confidences)),
        "confident_wrong_count": confident_wrong_count,
        "confidence_threshold": 0.80,
    }


def compute_category_metrics(df, predicted_labels):
    if "category" not in df.columns:
        return None

    temp_df = df.copy()
    temp_df["predicted_label"] = predicted_labels

    category_results = {}

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


def evaluate_dataset(df, tokenizer, model, device, entailment_index, output_path, compute_categories):
    true_labels = df["label"].tolist()

    predicted_labels, risky_scores, non_risky_scores, confidences = predict_zero_shot(
        df=df,
        tokenizer=tokenizer,
        model=model,
        device=device,
        entailment_index=entailment_index,
    )

    metrics = compute_metrics(
        true_labels=true_labels,
        predicted_labels=predicted_labels,
        confidences=confidences,
    )

    if compute_categories:
        category_metrics = compute_category_metrics(df, predicted_labels)

        if category_metrics is not None:
            metrics["category_metrics"] = category_metrics

    predictions_df = pd.DataFrame(
        {
            "text": df["text"],
            "true_label": true_labels,
            "predicted_label": predicted_labels,
            "risky_entailment_score": risky_scores,
            "non_risky_entailment_score": non_risky_scores,
            "confidence": confidences,
            "correct": np.array(true_labels) == np.array(predicted_labels),
        }
    )

    for col in ["id", "keyword", "category"]:
        if col in df.columns:
            predictions_df[col] = df[col]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    predictions_df.to_csv(output_path, index=False)

    print(f"Saved predictions to: {output_path}")

    return metrics


def evaluate_nli_zero_shot(experiment_id, model_name):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")
    print(f"Loading NLI model: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.to(device)
    model.eval()

    entailment_index = find_entailment_index(model)

    print(f"Entailment index: {entailment_index}")
    print(f"Risky hypothesis: {RISKY_HYPOTHESIS}")
    print(f"Non-risky hypothesis: {NON_RISKY_HYPOTHESIS}")

    test_df = load_dataset(TEST_PATH)
    ood_df = load_dataset(OOD_PATH)

    predictions_dir = RESULTS_DIR / "predictions"
    metrics_dir = RESULTS_DIR / "metrics"

    id_predictions_path = predictions_dir / f"{experiment_id}_id_predictions.csv"
    ood_predictions_path = predictions_dir / f"{experiment_id}_ood_predictions.csv"

    print("\nEvaluating ID test set...")
    id_metrics = evaluate_dataset(
        df=test_df,
        tokenizer=tokenizer,
        model=model,
        device=device,
        entailment_index=entailment_index,
        output_path=id_predictions_path,
        compute_categories=True,
    )

    print("\nEvaluating OOD set...")
    ood_metrics = evaluate_dataset(
        df=ood_df,
        tokenizer=tokenizer,
        model=model,
        device=device,
        entailment_index=entailment_index,
        output_path=ood_predictions_path,
        compute_categories=True,
    )

    all_metrics = {
        "experiment_id": experiment_id,
        "model_name": model_name,
        "mode": "nli_zero_shot",
        "risky_hypothesis": RISKY_HYPOTHESIS,
        "non_risky_hypothesis": NON_RISKY_HYPOTHESIS,
        "id_test": id_metrics,
        "ood": ood_metrics,
    }

    metrics_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = metrics_dir / f"{experiment_id}.json"

    with open(metrics_path, "w", encoding="utf-8") as file:
        json.dump(all_metrics, file, indent=4)

    print(f"Saved metrics to: {metrics_path}")

    print("\nEvaluation summary:")
    print(f"ID accuracy: {id_metrics['accuracy']:.4f}")
    print(f"ID macro F1: {id_metrics['macro_f1']:.4f}")
    print(f"OOD accuracy: {ood_metrics['accuracy']:.4f}")
    print(f"OOD macro F1: {ood_metrics['macro_f1']:.4f}")


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--experiment_id", type=str, default="E6")
    parser.add_argument("--model_name", type=str, default=NLI_MODEL_NAME)

    return parser.parse_args()


def main():
    args = parse_args()

    evaluate_nli_zero_shot(
        experiment_id=args.experiment_id,
        model_name=args.model_name,
    )


if __name__ == "__main__":
    main()