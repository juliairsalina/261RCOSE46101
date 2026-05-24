import argparse
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import shap

from transformers import AutoTokenizer, AutoModelForSequenceClassification

from src.config import MAX_LEN, OOD_PATH, SEED


SAVED_MODELS_DIR = Path("saved_models")
OUTPUT_DIR = Path("results") / "shap"


FIXED_EXAMPLES = [
    {
        "text": "I do not want to die.",
        "keyword": "die",
        "source": "fixed",
    },
    {
        "text": "I want to die in my hometown.",
        "keyword": "die",
        "source": "fixed",
    },
    {
        "text": "I want to kill myself.",
        "keyword": "kill",
        "source": "fixed",
    },
    {
        "text": "I want to eat ice cream.",
        "keyword": "",
        "source": "fixed",
    },
    {
        "text": "I want to kill my mom because she made me mad.",
        "keyword": "kill",
        "source": "fixed",
    },
]


def load_model(experiment_id, device):
    model_path = SAVED_MODELS_DIR / experiment_id

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.to(device)
    model.eval()

    return tokenizer, model


def load_shap_examples():
    examples = FIXED_EXAMPLES.copy()

    try:
        ood_df = pd.read_csv(OOD_PATH)

        if "text" in ood_df.columns:
            sample_size = min(50, len(ood_df))
            sampled_df = ood_df.sample(n=sample_size, random_state=SEED)

            for _, row in sampled_df.iterrows():
                examples.append(
                    {
                        "text": str(row["text"]),
                        "keyword": str(row["keyword"]) if "keyword" in row and not pd.isna(row["keyword"]) else "",
                        "source": "ood",
                    }
                )

    except Exception as error:
        print(f"Warning: Could not load OOD examples for SHAP: {error}")

    return pd.DataFrame(examples)


def make_predict_function(tokenizer, model, device):
    """
    SHAP calls this function with a list/array of texts.
    It must return probability scores.
    """

    def predict_proba(texts):
        if isinstance(texts, str):
            texts_list = [texts]
        else:
            texts_list = [str(text) for text in texts]

        all_probs = []

        for text in texts_list:
            encoding = tokenizer(
                text,
                truncation=True,
                padding="max_length",
                max_length=MAX_LEN,
                return_tensors="pt",
            )

            encoding = {key: value.to(device) for key, value in encoding.items()}

            with torch.no_grad():
                outputs = model(**encoding)
                probs = torch.softmax(outputs.logits, dim=1)

            all_probs.append(probs.cpu().numpy()[0])

        return np.array(all_probs)

    return predict_proba


def safe_filename(text, max_len=40):
    text = re.sub(r"[^a-zA-Z0-9]+", "_", text)
    text = text.strip("_")
    return text[:max_len]


def extract_token_level_values(shap_values, example_index, class_index=1):
    """
    Extract tokens and SHAP values for one example.

    class_index=1 means risky class.
    """
    tokens = shap_values.data[example_index]

    values = shap_values.values[example_index]

    if values.ndim == 2:
        token_values = values[:, class_index]
    else:
        token_values = values

    rows = []

    for token, value in zip(tokens, token_values):
        rows.append(
            {
                "token": str(token),
                "shap_value": float(value),
                "abs_shap_value": float(abs(value)),
            }
        )

    return pd.DataFrame(rows)


def compute_keyword_shap_ratio(token_df, keyword):
    """
    Compute how much of total token attribution belongs to the shortcut keyword.
    """
    if keyword is None or str(keyword).strip() == "":
        return np.nan

    keyword = str(keyword).lower().strip()

    total_abs_shap = token_df["abs_shap_value"].sum()

    if total_abs_shap == 0:
        return np.nan

    keyword_mask = token_df["token"].str.lower().str.contains(keyword, regex=False)
    keyword_abs_shap = token_df.loc[keyword_mask, "abs_shap_value"].sum()

    return float(keyword_abs_shap / total_abs_shap)


def save_waterfall_plot(shap_values, example_index, output_path):
    """
    Save SHAP waterfall plot.

    If the environment fails to render the plot, do not crash.
    """
    try:
        import matplotlib.pyplot as plt

        shap.plots.waterfall(shap_values[example_index, :, 1], show=False)
        plt.tight_layout()
        plt.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close()

    except Exception as error:
        print(f"Warning: Could not save waterfall plot {output_path}: {error}")


def run_shap_analysis(experiment_id):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tokenizer, model = load_model(experiment_id, device)

    examples_df = load_shap_examples()

    texts = examples_df["text"].astype(str).tolist()

    predict_proba = make_predict_function(tokenizer, model, device)

    print(f"Running SHAP for {len(texts)} examples...")

    try:
        masker = shap.maskers.Text(tokenizer)
        explainer = shap.Explainer(predict_proba, masker)
        shap_values = explainer(texts)

    except Exception as error:
        raise RuntimeError(
            "SHAP explainer failed. Try running on Colab with shap installed.\n"
            f"Original error: {error}"
        )

    all_token_rows = []
    keyword_rows = []

    for i, row in examples_df.iterrows():
        text = row["text"]
        keyword = row.get("keyword", "")
        source = row.get("source", "")

        token_df = extract_token_level_values(
            shap_values=shap_values,
            example_index=i,
            class_index=1,
        )

        token_df["experiment_id"] = experiment_id
        token_df["example_index"] = i
        token_df["text"] = text
        token_df["keyword"] = keyword
        token_df["source"] = source

        keyword_ratio = compute_keyword_shap_ratio(token_df, keyword)

        keyword_rows.append(
            {
                "experiment_id": experiment_id,
                "example_index": i,
                "text": text,
                "keyword": keyword,
                "source": source,
                "keyword_shap_ratio": keyword_ratio,
            }
        )

        all_token_rows.append(token_df)

        filename = f"{experiment_id}_{i}_{safe_filename(text)}.png"
        plot_path = OUTPUT_DIR / filename
        save_waterfall_plot(shap_values, i, plot_path)

    all_tokens_df = pd.concat(all_token_rows, ignore_index=True)
    keyword_df = pd.DataFrame(keyword_rows)

    token_path = OUTPUT_DIR / f"{experiment_id}_shap_token_values.csv"
    keyword_path = OUTPUT_DIR / f"{experiment_id}_shap_keyword_reliance.csv"

    all_tokens_df.to_csv(token_path, index=False)
    keyword_df.to_csv(keyword_path, index=False)

    print(f"Saved token SHAP values to: {token_path}")
    print(f"Saved keyword reliance to: {keyword_path}")

    update_shap_summary()


def update_shap_summary():
    """
    If E1 and E5 SHAP files both exist, create a summary comparison.
    """
    summary_rows = []

    for experiment_id in ["E1", "E5"]:
        path = OUTPUT_DIR / f"{experiment_id}_shap_keyword_reliance.csv"

        if not path.exists():
            continue

        df = pd.read_csv(path)
        valid_df = df.dropna(subset=["keyword_shap_ratio"])

        if len(valid_df) == 0:
            continue

        summary_rows.append(
            {
                "experiment_id": experiment_id,
                "average_keyword_shap_ratio": valid_df["keyword_shap_ratio"].mean(),
                "num_examples": len(valid_df),
            }
        )

    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        summary_path = OUTPUT_DIR / "shap_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        print(f"Saved SHAP summary to: {summary_path}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_id", type=str, required=True)
    return parser.parse_args()


def main():
    warnings.filterwarnings("ignore")
    args = parse_args()

    try:
        run_shap_analysis(args.experiment_id)
    except Exception as error:
        print("\nWarning: SHAP analysis did not finish successfully.")
        print(error)
        print("The script stopped safely instead of crashing the whole pipeline.")


if __name__ == "__main__":
    main()