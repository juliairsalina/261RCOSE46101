import argparse
import json

import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix,
)
from sklearn.pipeline import Pipeline

from src.config import (
    TRAIN_PATH,
    TEST_PATH,
    OOD_PATH,
    RESULTS_DIR,
    SEED,
)


def load_dataset(path):
    """
    Load dataset CSV.
    """
    print(f"Loading dataset: {path}")

    df = pd.read_csv(path)

    required_columns = ["text", "label"]
    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        raise ValueError(f"Missing columns in {path}: {missing_columns}")

    df = df.dropna(subset=["text", "label"]).copy()

    df["text"] = df["text"].astype(str)
    df["label"] = df["label"].astype(int)

    return df


def build_tfidf_model():
    """
    Create TF-IDF + Logistic Regression model.
    """
    model = Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    stop_words="english",
                    ngram_range=(1, 2),
                    max_features=10000,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    random_state=SEED,
                    class_weight="balanced",
                ),
            ),
        ]
    )

    return model


def compute_metrics(true_labels, predicted_labels):
    """
    Compute evaluation metrics.
    """
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

    return metrics


def create_predictions_dataframe(df, predicted_labels, risky_probs):
    """
    Create prediction dataframe for saving.
    """
    predictions_df = pd.DataFrame(
        {
            "text": df["text"],
            "true_label": df["label"],
            "predicted_label": predicted_labels,
            "risky_prob": risky_probs,
            "correct": df["label"].values == predicted_labels,
        }
    )

    if "category" in df.columns:
        predictions_df["category"] = df["category"]

    return predictions_df


def evaluate_model(model, df, output_predictions_path):
    """
    Evaluate model and save predictions.
    """
    texts = df["text"]
    true_labels = df["label"]

    predicted_labels = model.predict(texts)

    # Probability for class 1 = risky
    risky_probs = model.predict_proba(texts)[:, 1]

    metrics = compute_metrics(true_labels, predicted_labels)

    predictions_df = create_predictions_dataframe(
        df=df,
        predicted_labels=predicted_labels,
        risky_probs=risky_probs,
    )

    output_predictions_path.parent.mkdir(parents=True, exist_ok=True)
    predictions_df.to_csv(output_predictions_path, index=False)

    print(f"Saved predictions to: {output_predictions_path}")

    return metrics


def save_metrics(metrics, path):
    """
    Save metrics as JSON.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=4)

    print(f"Saved metrics to: {path}")


def train_and_evaluate(experiment_id="E0"):
    """
    Train TF-IDF baseline and evaluate on ID test set and OOD set.
    """
    train_df = load_dataset(TRAIN_PATH)
    test_df = load_dataset(TEST_PATH)
    ood_df = load_dataset(OOD_PATH)

    print("\nTraining TF-IDF + Logistic Regression model...")

    model = build_tfidf_model()
    model.fit(train_df["text"], train_df["label"])

    print("Training complete.")

    metrics_dir = RESULTS_DIR / "metrics"
    predictions_dir = RESULTS_DIR / "predictions"

    id_predictions_path = predictions_dir / f"{experiment_id}_id_predictions.csv"
    ood_predictions_path = predictions_dir / f"{experiment_id}_ood_predictions.csv"

    print("\nEvaluating on ID test set...")
    id_metrics = evaluate_model(
        model=model,
        df=test_df,
        output_predictions_path=id_predictions_path,
    )

    print("\nEvaluating on OOD set...")
    ood_metrics = evaluate_model(
        model=model,
        df=ood_df,
        output_predictions_path=ood_predictions_path,
    )

    all_metrics = {
        "experiment_id": experiment_id,
        "model": "TF-IDF + Logistic Regression",
        "id_test": id_metrics,
        "ood": ood_metrics,
    }

    metrics_path = metrics_dir / f"{experiment_id}_tfidf.json"
    save_metrics(all_metrics, metrics_path)

    print("\nDone.")
    print(f"ID accuracy: {id_metrics['accuracy']:.4f}")
    print(f"ID macro F1: {id_metrics['macro_f1']:.4f}")
    print(f"OOD accuracy: {ood_metrics['accuracy']:.4f}")
    print(f"OOD macro F1: {ood_metrics['macro_f1']:.4f}")


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--experiment_id",
        type=str,
        default="E0",
        help="Experiment ID. Default: E0",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    train_and_evaluate(experiment_id=args.experiment_id)


if __name__ == "__main__":
    main()