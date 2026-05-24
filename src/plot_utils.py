import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


RESULTS_DIR = Path("results")
METRICS_DIR = RESULTS_DIR / "metrics"
PLOTS_DIR = RESULTS_DIR / "plots"
SHORTCUT_DIR = RESULTS_DIR / "shortcut_metrics"
SHAP_DIR = RESULTS_DIR / "shap"


def load_json(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def save_plot(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved plot: {path}")


def collect_experiment_metrics():
    rows = []

    for experiment_id in ["E1", "E2", "E3", "E4", "E5"]:
        path = METRICS_DIR / f"{experiment_id}.json"

        if not path.exists():
            continue

        data = load_json(path)

        id_metrics = data.get("id_test", {})
        ood_metrics = data.get("ood", {})

        rows.append(
            {
                "experiment_id": experiment_id,
                "id_macro_f1": id_metrics.get("macro_f1"),
                "ood_macro_f1": ood_metrics.get("macro_f1"),
                "id_accuracy": id_metrics.get("accuracy"),
                "ood_accuracy": ood_metrics.get("accuracy"),
                "id_ood_gap": (
                    id_metrics.get("macro_f1") - ood_metrics.get("macro_f1")
                    if id_metrics.get("macro_f1") is not None and ood_metrics.get("macro_f1") is not None
                    else None
                ),
                "id_confident_wrong": id_metrics.get("confident_wrong_count"),
                "ood_confident_wrong": ood_metrics.get("confident_wrong_count"),
            }
        )

    return pd.DataFrame(rows)


def plot_id_vs_ood_macro_f1(metrics_df):
    if metrics_df.empty:
        return

    plot_df = metrics_df.set_index("experiment_id")[["id_macro_f1", "ood_macro_f1"]]

    plot_df.plot(kind="bar")
    plt.title("ID vs OOD Macro-F1")
    plt.xlabel("Experiment")
    plt.ylabel("Macro-F1")
    plt.ylim(0, 1)

    save_plot(PLOTS_DIR / "id_vs_ood_macro_f1.png")


def plot_id_ood_gap(metrics_df):
    if metrics_df.empty or "id_ood_gap" not in metrics_df.columns:
        return

    plt.figure()
    plt.bar(metrics_df["experiment_id"], metrics_df["id_ood_gap"])
    plt.title("ID-OOD Macro-F1 Gap")
    plt.xlabel("Experiment")
    plt.ylabel("ID Macro-F1 - OOD Macro-F1")

    save_plot(PLOTS_DIR / "id_ood_gap.png")


def plot_confident_wrong(metrics_df):
    if metrics_df.empty:
        return

    if "ood_confident_wrong" not in metrics_df.columns:
        return

    plt.figure()
    plt.bar(metrics_df["experiment_id"], metrics_df["ood_confident_wrong"])
    plt.title("OOD Confident Wrong Count")
    plt.xlabel("Experiment")
    plt.ylabel("Confident Wrong Count")

    save_plot(PLOTS_DIR / "ood_confident_wrong_count.png")


def plot_ood_category_f1():
    rows = []

    for experiment_id in ["E1", "E2", "E3", "E4", "E5"]:
        path = METRICS_DIR / f"{experiment_id}.json"

        if not path.exists():
            continue

        data = load_json(path)
        category_metrics = data.get("ood", {}).get("category_metrics", {})

        for category, values in category_metrics.items():
            rows.append(
                {
                    "experiment_id": experiment_id,
                    "category": category,
                    "macro_f1": values.get("macro_f1"),
                }
            )

    if not rows:
        return

    df = pd.DataFrame(rows)
    pivot_df = df.pivot(index="category", columns="experiment_id", values="macro_f1")

    pivot_df.plot(kind="bar")
    plt.title("OOD Category-wise Macro-F1")
    plt.xlabel("OOD Category")
    plt.ylabel("Macro-F1")
    plt.ylim(0, 1)

    save_plot(PLOTS_DIR / "ood_category_macro_f1.png")


def plot_shortcut_sensitivity():
    path = SHORTCUT_DIR / "shortcut_summary.csv"

    if not path.exists():
        return

    df = pd.read_csv(path)

    plt.figure()
    plt.bar(df["experiment_id"], df["keyword_sensitivity_score"])
    plt.title("Keyword Sensitivity Score")
    plt.xlabel("Experiment")
    plt.ylabel("Average |P(original) - P(masked)|")

    save_plot(PLOTS_DIR / "keyword_sensitivity_score.png")

    plt.figure()
    plt.bar(df["experiment_id"], df["prediction_flip_rate"])
    plt.title("Prediction Flip Rate After Keyword Masking")
    plt.xlabel("Experiment")
    plt.ylabel("Flip Rate")

    save_plot(PLOTS_DIR / "prediction_flip_rate.png")


def plot_shap_keyword_reliance():
    path = SHAP_DIR / "shap_summary.csv"

    if not path.exists():
        return

    df = pd.read_csv(path)

    plt.figure()
    plt.bar(df["experiment_id"], df["average_keyword_shap_ratio"])
    plt.title("SHAP Keyword Reliance")
    plt.xlabel("Experiment")
    plt.ylabel("Average Keyword SHAP Ratio")

    save_plot(PLOTS_DIR / "shap_keyword_reliance.png")


def extract_train_log_rows(experiment_id):
    path = METRICS_DIR / f"{experiment_id}_train_log.json"

    if not path.exists():
        return []

    data = load_json(path)
    log_history = data.get("log_history", [])

    rows = []

    for item in log_history:
        row = {"experiment_id": experiment_id}
        row.update(item)
        rows.append(row)

    return rows


def plot_training_logs():
    rows = []

    for experiment_id in ["E1", "E2", "E3", "E4", "E5"]:
        rows.extend(extract_train_log_rows(experiment_id))

    if not rows:
        return

    df = pd.DataFrame(rows)

    if "loss" in df.columns:
        plt.figure()

        for experiment_id, group in df.dropna(subset=["loss"]).groupby("experiment_id"):
            plt.plot(group["step"], group["loss"], label=experiment_id)

        plt.title("Training Loss")
        plt.xlabel("Step")
        plt.ylabel("Loss")
        plt.legend()

        save_plot(PLOTS_DIR / "training_loss.png")

    if "eval_loss" in df.columns:
        plt.figure()

        for experiment_id, group in df.dropna(subset=["eval_loss"]).groupby("experiment_id"):
            x_axis = group["epoch"] if "epoch" in group.columns else group["step"]
            plt.plot(x_axis, group["eval_loss"], marker="o", label=experiment_id)

        plt.title("Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Validation Loss")
        plt.legend()

        save_plot(PLOTS_DIR / "validation_loss.png")

    if "eval_accuracy" in df.columns:
        plt.figure()

        for experiment_id, group in df.dropna(subset=["eval_accuracy"]).groupby("experiment_id"):
            x_axis = group["epoch"] if "epoch" in group.columns else group["step"]
            plt.plot(x_axis, group["eval_accuracy"], marker="o", label=experiment_id)

        plt.title("Validation Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.ylim(0, 1)
        plt.legend()

        save_plot(PLOTS_DIR / "validation_accuracy.png")

    if "eval_macro_f1" in df.columns:
        plt.figure()

        for experiment_id, group in df.dropna(subset=["eval_macro_f1"]).groupby("experiment_id"):
            x_axis = group["epoch"] if "epoch" in group.columns else group["step"]
            plt.plot(x_axis, group["eval_macro_f1"], marker="o", label=experiment_id)

        plt.title("Validation Macro-F1")
        plt.xlabel("Epoch")
        plt.ylabel("Macro-F1")
        plt.ylim(0, 1)
        plt.legend()

        save_plot(PLOTS_DIR / "validation_macro_f1.png")


def main():
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    metrics_df = collect_experiment_metrics()

    plot_id_vs_ood_macro_f1(metrics_df)
    plot_id_ood_gap(metrics_df)
    plot_confident_wrong(metrics_df)
    plot_ood_category_f1()
    plot_shortcut_sensitivity()
    plot_shap_keyword_reliance()
    plot_training_logs()

    print("\nPlot generation finished.")
    print(f"Plots saved to: {PLOTS_DIR}")


if __name__ == "__main__":
    main()