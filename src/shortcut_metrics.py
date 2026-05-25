import argparse
import re

import numpy as np
import pandas as pd
import torch

from transformers import AutoTokenizer, AutoModelForSequenceClassification

from src.config import (
    MAX_LEN,
    OOD_PATH,
    RESULTS_DIR,
    MODEL_DIR,
)


OUTPUT_DIR = RESULTS_DIR / "shortcut_metrics"


def load_ood_data() -> pd.DataFrame:
    df = pd.read_csv(OOD_PATH)

    required_columns = ["text", "keyword", "label"]
    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        raise ValueError(
            f"Missing columns in OOD data: {missing_columns}\n"
            f"Available columns: {list(df.columns)}"
        )

    df = df.dropna(subset=["text", "label"]).copy()
    df["text"] = df["text"].astype(str)
    df["keyword"] = df["keyword"].astype(str)
    df["label"] = df["label"].astype(int)

    return df


def mask_keyword(text, keyword, mask_token="[MASK]"):
    if pd.isna(keyword):
        return text

    text = str(text)
    keyword = str(keyword).strip()

    if keyword == "":
        return text

    pattern = re.compile(re.escape(keyword), re.IGNORECASE)

    if not pattern.search(text):
        return text

    return pattern.sub(mask_token, text)


def load_model(experiment_id, device):
    model_path = MODEL_DIR / experiment_id

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found: {model_path}\n"
            f"Train the model first before running shortcut metrics."
        )

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)

    model.to(device)
    model.eval()

    return tokenizer, model


def predict_risky_prob(text, tokenizer, model, device):
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

    return risky_prob, predicted_label


def evaluate_keyword_sensitivity_for_model(experiment_id, df, device):
    tokenizer, model = load_model(experiment_id, device)

    rows = []

    for _, row in df.iterrows():
        text = row["text"]
        keyword = row.get("keyword", "")
        true_label = int(row["label"])

        masked_text = mask_keyword(text, keyword)

        original_prob, original_pred = predict_risky_prob(
            text=text,
            tokenizer=tokenizer,
            model=model,
            device=device,
        )

        masked_prob, masked_pred = predict_risky_prob(
            text=masked_text,
            tokenizer=tokenizer,
            model=model,
            device=device,
        )

        probability_change = abs(original_prob - masked_prob)
        prediction_flip = int(original_pred != masked_pred)

        result = {
            "experiment_id": experiment_id,
            "text": text,
            "keyword": keyword,
            "masked_text": masked_text,
            "true_label": true_label,
            "original_risky_prob": original_prob,
            "masked_risky_prob": masked_prob,
            "absolute_probability_change": probability_change,
            "original_predicted_label": original_pred,
            "masked_predicted_label": masked_pred,
            "prediction_flip": prediction_flip,
        }

        for col in ["id", "category"]:
            if col in df.columns:
                result[col] = row[col]

        rows.append(result)

    return pd.DataFrame(rows)


def summarize_shortcut_metrics(details_df):
    summaries = []

    for experiment_id, group in details_df.groupby("experiment_id"):
        summaries.append(
            {
                "experiment_id": experiment_id,
                "keyword_sensitivity_score": group["absolute_probability_change"].mean(),
                "prediction_flip_rate": group["prediction_flip"].mean(),
                "num_examples": len(group),
            }
        )

    return pd.DataFrame(summaries)


def run_shortcut_metrics(experiment_ids):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    ood_df = load_ood_data()

    all_details = []

    for experiment_id in experiment_ids:
        print(f"\nRunning shortcut sensitivity for {experiment_id}...")

        details_df = evaluate_keyword_sensitivity_for_model(
            experiment_id=experiment_id,
            df=ood_df,
            device=device,
        )

        all_details.append(details_df)

    combined_details_df = pd.concat(all_details, ignore_index=True)
    summary_df = summarize_shortcut_metrics(combined_details_df)

    details_path = OUTPUT_DIR / "keyword_sensitivity_details.csv"
    summary_path = OUTPUT_DIR / "shortcut_summary.csv"

    combined_details_df.to_csv(details_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    print(f"\nSaved details to: {details_path}")
    print(f"Saved summary to: {summary_path}")

    print("\nShortcut summary:")
    print(summary_df)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--experiment_ids",
        nargs="+",
        default=["E1", "E10"],
        help="Experiment IDs to compare. Default: E1 E10",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    run_shortcut_metrics(experiment_ids=args.experiment_ids)


if __name__ == "__main__":
    main()