import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from transformers import AutoTokenizer, AutoModelForSequenceClassification

from src.config import MAX_LEN, OOD_PATH


SAVED_MODELS_DIR = Path("saved_models")
OUTPUT_DIR = Path("results") / "uncertainty"


def load_ood_data():
    df = pd.read_csv(OOD_PATH)

    required_columns = ["text", "label"]
    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        raise ValueError(f"Missing columns in OOD data: {missing_columns}")

    df = df.dropna(subset=["text", "label"]).copy()
    df["text"] = df["text"].astype(str)
    df["label"] = df["label"].astype(int)

    return df


def enable_dropout(model):
    """
    Keep dropout active during inference.
    """
    for module in model.modules():
        if "Dropout" in module.__class__.__name__:
            module.train()


def predictive_entropy(probabilities):
    """
    Entropy = -sum(p log p)
    """
    probabilities = np.clip(probabilities, 1e-12, 1.0)
    return float(-np.sum(probabilities * np.log(probabilities)))


def mc_dropout_predict(text, tokenizer, model, device, mc_passes):
    encoding = tokenizer(
        str(text),
        truncation=True,
        padding="max_length",
        max_length=MAX_LEN,
        return_tensors="pt",
    )

    encoding = {key: value.to(device) for key, value in encoding.items()}

    all_probs = []

    model.eval()
    enable_dropout(model)

    with torch.no_grad():
        for _ in range(mc_passes):
            outputs = model(**encoding)
            probs = torch.softmax(outputs.logits, dim=1)
            all_probs.append(probs.cpu().numpy()[0])

    all_probs = np.array(all_probs)

    mean_probs = all_probs.mean(axis=0)
    risky_probs = all_probs[:, 1]

    mean_risky_prob = float(mean_probs[1])
    variance = float(np.var(risky_probs))
    entropy = predictive_entropy(mean_probs)

    predicted_label = int(np.argmax(mean_probs))
    confidence = float(np.max(mean_probs))

    return mean_risky_prob, variance, entropy, predicted_label, confidence


def run_mc_dropout(experiment_id, mc_passes):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    model_path = SAVED_MODELS_DIR / experiment_id

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found: {model_path}\n"
            f"Train model first before running MC Dropout."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Loading model from: {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.to(device)

    df = load_ood_data()

    rows = []

    for _, row in df.iterrows():
        text = row["text"]
        true_label = int(row["label"])

        mean_risky_prob, variance, entropy, predicted_label, confidence = mc_dropout_predict(
            text=text,
            tokenizer=tokenizer,
            model=model,
            device=device,
            mc_passes=mc_passes,
        )

        result = {
            "text": text,
            "true_label": true_label,
            "mean_risky_prob": mean_risky_prob,
            "variance": variance,
            "predictive_entropy": entropy,
            "predicted_label": predicted_label,
            "confidence": confidence,
            "correct": true_label == predicted_label,
        }

        if "category" in df.columns:
            result["category"] = row["category"]

        rows.append(result)

    output_df = pd.DataFrame(rows)

    prediction_path = OUTPUT_DIR / f"{experiment_id}_mc_dropout_ood.csv"
    output_df.to_csv(prediction_path, index=False)

    summary = {
        "experiment_id": experiment_id,
        "mc_passes": mc_passes,
        "num_examples": len(output_df),
        "accuracy": float(output_df["correct"].mean()),
        "average_confidence": float(output_df["confidence"].mean()),
        "average_variance": float(output_df["variance"].mean()),
        "average_predictive_entropy": float(output_df["predictive_entropy"].mean()),
    }

    if "category" in output_df.columns:
        category_summary = {}

        for category, group in output_df.groupby("category"):
            category_summary[str(category)] = {
                "count": int(len(group)),
                "accuracy": float(group["correct"].mean()),
                "average_confidence": float(group["confidence"].mean()),
                "average_variance": float(group["variance"].mean()),
                "average_predictive_entropy": float(group["predictive_entropy"].mean()),
            }

        summary["category_summary"] = category_summary

    summary_path = OUTPUT_DIR / f"{experiment_id}_mc_dropout_summary.json"

    with open(summary_path, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=4)

    print(f"Saved MC Dropout predictions to: {prediction_path}")
    print(f"Saved MC Dropout summary to: {summary_path}")


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--experiment_id", type=str, required=True)
    parser.add_argument("--mc_passes", type=int, default=30)

    return parser.parse_args()


def main():
    args = parse_args()
    run_mc_dropout(
        experiment_id=args.experiment_id,
        mc_passes=args.mc_passes,
    )


if __name__ == "__main__":
    main()