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


LABEL_MAP = {
    0: "non-risky",
    1: "risky",
}


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
    prediction = LABEL_MAP.get(predicted_label, str(predicted_label))
    confidence = float(np.max(mean_probs))

    return {
        "mean_risky_prob": mean_risky_prob,
        "variance": variance,
        "predictive_entropy": entropy,
        "uncertainty": entropy,
        "predicted_label": predicted_label,
        "prediction": prediction,
        "confidence": confidence,
    }


def print_examples(output_df, title, n=10):
    """
    Print readable examples like:
    Text        :
    Prediction  :
    Confidence  :
    Uncertainty :
    """
    if output_df.empty:
        print(f"\nNo examples to print for: {title}")
        return

    print("\n" + "#" * 80)
    print(title)
    print("#" * 80)

    for _, row in output_df.head(n).iterrows():
        print("=" * 65)
        print(f"Text        : {row['text']}")
        print(f"Prediction  : {row['prediction']}")
        print(f"Confidence  : {row['confidence']:.4f}")
        print(f"Uncertainty : {row['uncertainty']:.6f}")

        if "category" in row:
            print(f"Category    : {row['category']}")

        if "correct" in row:
            print(f"Correct     : {row['correct']}")


def run_mc_dropout(experiment_id, mc_passes, print_top_n):
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

        prediction_result = mc_dropout_predict(
            text=text,
            tokenizer=tokenizer,
            model=model,
            device=device,
            mc_passes=mc_passes,
        )

        result = {
            "text": text,
            "true_label": true_label,
            "true_label_name": LABEL_MAP.get(true_label, str(true_label)),
            "mean_risky_prob": prediction_result["mean_risky_prob"],
            "variance": prediction_result["variance"],
            "predictive_entropy": prediction_result["predictive_entropy"],
            "uncertainty": prediction_result["uncertainty"],
            "predicted_label": prediction_result["predicted_label"],
            "prediction": prediction_result["prediction"],
            "confidence": prediction_result["confidence"],
            "correct": true_label == prediction_result["predicted_label"],
        }

        if "keyword" in df.columns:
            result["keyword"] = row["keyword"]

        if "category" in df.columns:
            result["category"] = row["category"]

        rows.append(result)

    output_df = pd.DataFrame(rows)

    prediction_path = OUTPUT_DIR / f"{experiment_id}_mc_dropout_ood.csv"
    output_df.to_csv(prediction_path, index=False)

    wrong_df = output_df[output_df["correct"] == False].copy()
    confident_wrong_df = wrong_df[wrong_df["confidence"] >= 0.8].copy()

    summary = {
        "experiment_id": experiment_id,
        "mc_passes": mc_passes,
        "num_examples": len(output_df),
        "accuracy": float(output_df["correct"].mean()),
        "average_confidence": float(output_df["confidence"].mean()),
        "average_variance": float(output_df["variance"].mean()),
        "average_predictive_entropy": float(output_df["predictive_entropy"].mean()),
        "wrong_count": int(len(wrong_df)),
        "confident_wrong_count": int(len(confident_wrong_df)),
        "confidence_threshold": 0.8,
    }

    if "category" in output_df.columns:
        category_summary = {}

        for category, group in output_df.groupby("category"):
            group_wrong = group[group["correct"] == False]
            group_confident_wrong = group_wrong[group_wrong["confidence"] >= 0.8]

            category_summary[str(category)] = {
                "count": int(len(group)),
                "accuracy": float(group["correct"].mean()),
                "average_confidence": float(group["confidence"].mean()),
                "average_variance": float(group["variance"].mean()),
                "average_predictive_entropy": float(group["predictive_entropy"].mean()),
                "wrong_count": int(len(group_wrong)),
                "confident_wrong_count": int(len(group_confident_wrong)),
            }

        summary["category_summary"] = category_summary

    summary_path = OUTPUT_DIR / f"{experiment_id}_mc_dropout_summary.json"

    with open(summary_path, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=4)

    print(f"\nSaved MC Dropout predictions to: {prediction_path}")
    print(f"Saved MC Dropout summary to: {summary_path}")

    print("\nMC Dropout Summary")
    print("-" * 65)
    print(f"Experiment ID                  : {experiment_id}")
    print(f"MC passes                      : {mc_passes}")
    print(f"Number of OOD examples          : {summary['num_examples']}")
    print(f"Accuracy                       : {summary['accuracy']:.4f}")
    print(f"Average confidence             : {summary['average_confidence']:.4f}")
    print(f"Average variance               : {summary['average_variance']:.6f}")
    print(f"Average predictive entropy      : {summary['average_predictive_entropy']:.6f}")
    print(f"Wrong count                    : {summary['wrong_count']}")
    print(f"Confident wrong count           : {summary['confident_wrong_count']}")

    # Print examples like teammate screenshot
    print_examples(
        output_df=output_df.head(print_top_n),
        title=f"First {print_top_n} OOD MC Dropout Examples",
        n=print_top_n,
    )

    high_uncertainty_df = output_df.sort_values(
        "uncertainty",
        ascending=False,
    ).head(print_top_n)

    print_examples(
        output_df=high_uncertainty_df,
        title=f"Top {print_top_n} Highest-Uncertainty Examples",
        n=print_top_n,
    )

    if not confident_wrong_df.empty:
        confident_wrong_df = confident_wrong_df.sort_values(
            "confidence",
            ascending=False,
        ).head(print_top_n)

        print_examples(
            output_df=confident_wrong_df,
            title=f"Top {print_top_n} Confident Wrong Examples",
            n=print_top_n,
        )
    else:
        print("\nNo confident wrong examples found at confidence >= 0.8.")


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--experiment_id", type=str, required=True)
    parser.add_argument("--mc_passes", type=int, default=30)
    parser.add_argument(
        "--print_top_n",
        type=int,
        default=10,
        help="Number of readable examples to print.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    run_mc_dropout(
        experiment_id=args.experiment_id,
        mc_passes=args.mc_passes,
        print_top_n=args.print_top_n,
    )


if __name__ == "__main__":
    main()