import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.config import RESULTS_DIR


METRICS_DIR = RESULTS_DIR / "metrics"
PLOTS_DIR = RESULTS_DIR / "plots"
SHORTCUT_DIR = RESULTS_DIR / "shortcut_metrics"
SHAP_DIR = RESULTS_DIR / "shap"

EXPERIMENT_IDS = [
    "E1",
    "E2",
    "E3",
    "E4",
    "E5",
    "E6",
    "E7",
    "E8",
    "E9",
    "E10",
]


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def save_plot(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved plot: {path}")


def collect_experiment_metrics() -> pd.DataFrame:
    """
    Collect ID and OOD metrics for E1–E10.

    Expected metric file:
        results/metrics/E1.json
        results/metrics/E2.json
        ...
        results/metrics/E10.json
    """
    rows = []

    for experiment_id in EXPERIMENT_IDS:
        path = METRICS_DIR / f"{experiment_id}.json"

        if not path.exists():
            print(f"Skipping {experiment_id}: metrics file not found.")
            continue

        data = load_json(path)

        id_metrics = data.get("id_test", {})
        ood_metrics = data.get("ood", {})

        id_macro_f1 = id_metrics.get("macro_f1")
        ood_macro_f1 = ood_metrics.get("macro_f1")

        if id_macro_f1 is not None and ood_macro_f1 is not None:
            id_ood_gap = id_macro_f1 - ood_macro_f1
        else:
            id_ood_gap = None

        rows.append(
            {
                "experiment_id": experiment_id,
                "id_accuracy": id_metrics.get("accuracy"),
                "ood_accuracy": ood_metrics.get("accuracy"),
                "id_macro_f1": id_macro_f1,
                "ood_macro_f1": ood_macro_f1,
                "id_ood_gap": id_ood_gap,
                "id_confident_wrong": id_metrics.get("confident_wrong_count"),
                "ood_confident_wrong": ood_metrics.get("confident_wrong_count"),
            }
        )

    return pd.DataFrame(rows)


def plot_id_vs_ood_macro_f1(metrics_df: pd.DataFrame) -> None:
    if metrics_df.empty:
        print("No metrics available for ID vs OOD Macro-F1 plot.")
        return

    required_columns = ["experiment_id", "id_macro_f1", "ood_macro_f1"]

    if not all(col in metrics_df.columns for col in required_columns):
        print("Missing columns for ID vs OOD Macro-F1 plot.")
        return

    plot_df = metrics_df.set_index("experiment_id")[["id_macro_f1", "ood_macro_f1"]]

    plt.figure()
    plot_df.plot(kind="bar")
    plt.title("ID vs OOD Macro-F1")
    plt.xlabel("Experiment")
    plt.ylabel("Macro-F1")
    plt.ylim(0, 1)
    plt.xticks(rotation=0)

    save_plot(PLOTS_DIR / "id_vs_ood_macro_f1.png")


def plot_id_vs_ood_accuracy(metrics_df: pd.DataFrame) -> None:
    if metrics_df.empty:
        print("No metrics available for ID vs OOD Accuracy plot.")
        return

    required_columns = ["experiment_id", "id_accuracy", "ood_accuracy"]

    if not all(col in metrics_df.columns for col in required_columns):
        print("Missing columns for ID vs OOD Accuracy plot.")
        return

    plot_df = metrics_df.set_index("experiment_id")[["id_accuracy", "ood_accuracy"]]

    plt.figure()
    plot_df.plot(kind="bar")
    plt.title("ID vs OOD Accuracy")
    plt.xlabel("Experiment")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    plt.xticks(rotation=0)

    save_plot(PLOTS_DIR / "id_vs_ood_accuracy.png")


def plot_id_ood_gap(metrics_df: pd.DataFrame) -> None:
    if metrics_df.empty or "id_ood_gap" not in metrics_df.columns:
        print("No ID-OOD gap data available.")
        return

    plot_df = metrics_df.dropna(subset=["id_ood_gap"])

    if plot_df.empty:
        print("ID-OOD gap column is empty.")
        return

    plt.figure()
    plt.bar(plot_df["experiment_id"], plot_df["id_ood_gap"])
    plt.title("ID-OOD Macro-F1 Gap")
    plt.xlabel("Experiment")
    plt.ylabel("ID Macro-F1 - OOD Macro-F1")

    save_plot(PLOTS_DIR / "id_ood_gap.png")


def plot_confident_wrong(metrics_df: pd.DataFrame) -> None:
    if metrics_df.empty or "ood_confident_wrong" not in metrics_df.columns:
        print("No confident-wrong data available.")
        return

    plot_df = metrics_df.dropna(subset=["ood_confident_wrong"])

    if plot_df.empty:
        print("OOD confident-wrong column is empty.")
        return

    plt.figure()
    plt.bar(plot_df["experiment_id"], plot_df["ood_confident_wrong"])
    plt.title("OOD Confident Wrong Count")
    plt.xlabel("Experiment")
    plt.ylabel("Confident Wrong Count")

    save_plot(PLOTS_DIR / "ood_confident_wrong_count.png")


def plot_ood_category_f1() -> None:
    rows = []

    for experiment_id in EXPERIMENT_IDS:
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
        print("No OOD category metrics available.")
        return

    df = pd.DataFrame(rows)

    pivot_df = df.pivot(
        index="category",
        columns="experiment_id",
        values="macro_f1",
    )

    plt.figure()
    pivot_df.plot(kind="bar")
    plt.title("OOD Category-wise Macro-F1")
    plt.xlabel("OOD Category")
    plt.ylabel("Macro-F1")
    plt.ylim(0, 1)
    plt.xticks(rotation=45, ha="right")

    save_plot(PLOTS_DIR / "ood_category_macro_f1.png")


def plot_shortcut_sensitivity() -> None:
    path = SHORTCUT_DIR / "shortcut_summary.csv"

    if not path.exists():
        print("Shortcut summary not found. Skipping shortcut plots.")
        return

    df = pd.read_csv(path)

    if df.empty:
        print("Shortcut summary is empty.")
        return

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



def extract_train_log_rows(experiment_id: str) -> list[dict]:
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


def plot_training_logs() -> None:
    rows = []

    for experiment_id in EXPERIMENT_IDS:
        rows.extend(extract_train_log_rows(experiment_id))

    if not rows:
        print("No training logs found.")
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


def main() -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    metrics_df = collect_experiment_metrics()

    if not metrics_df.empty:
        metrics_table_path = PLOTS_DIR / "experiment_metrics_summary.csv"
        metrics_df.to_csv(metrics_table_path, index=False)
        print(f"Saved metrics summary table: {metrics_table_path}")

    plot_id_vs_ood_macro_f1(metrics_df)
    plot_id_vs_ood_accuracy(metrics_df)
    plot_id_ood_gap(metrics_df)
    plot_confident_wrong(metrics_df)
    plot_ood_category_f1()
    plot_shortcut_sensitivity()
    plot_training_logs()

    print("\nPlot generation finished.")
    print(f"Plots saved to: {PLOTS_DIR}")


if __name__ == "__main__":
    main()