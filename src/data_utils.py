import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import (
    RAW_DATA_PATH,
    RAW_OOD_PATH,
    TRAIN_PATH,
    VAL_PATH,
    TEST_PATH,
    OOD_PATH,
    PROCESSED_DIR,
    SEED,
)


# =========================
# Helper Functions
# =========================

def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean column names by removing extra spaces and converting to lowercase.
    Example: ' Gold_Label ' -> 'gold_label'
    """
    df.columns = df.columns.str.strip().str.lower()
    return df


def validate_columns(df: pd.DataFrame, required_columns: list[str]) -> None:
    """
    Check whether the dataset contains all required columns.
    """
    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}\n"
            f"Available columns: {list(df.columns)}"
        )


def map_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert labels into numbers:
    risky -> 1
    non-risky -> 0
    """
    label_map = {
        "risky": 1,
        "non-risky": 0,
    }

    df["label"] = df["label"].astype(str).str.strip().str.lower()
    df["label"] = df["label"].map(label_map)

    # Remove rows with labels that are not risky or non-risky
    df = df.dropna(subset=["label"])

    df["label"] = df["label"].astype(int)

    return df


def print_label_distribution(df: pd.DataFrame, name: str) -> None:
    """
    Print the number of risky and non-risky samples.
    """
    print(f"\n{name} label distribution:")
    print(df["label"].value_counts().sort_index())
    print("0 = non-risky, 1 = risky")


def save_csv(df: pd.DataFrame, path) -> None:
    """
    Save dataframe to CSV and print the saved path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Saved: {path}")


# =========================
# Main Dataset Processing
# =========================

def prepare_main_dataset() -> None:
    """
    Load main dataset, clean it, split into train/val/test, and save CSV files.
    """
    print("\nLoading main dataset...")
    print(f"Path: {RAW_DATA_PATH}")

    df = pd.read_excel(RAW_DATA_PATH)
    df = clean_column_names(df)

    required_columns = ["id", "text", "keyword", "label"]
    validate_columns(df, required_columns)

    # Keep only useful columns
    df = df[required_columns].copy()

    # Remove missing text or label
    df = df.dropna(subset=["text", "label"])

    # Remove duplicate text rows
    df = df.drop_duplicates(subset=["text"])

    # Map labels to 0 and 1
    df = map_labels(df)

    print_label_distribution(df, "Full main dataset")

    # First split:
    # train = 70%
    # temporary = 30%
    train_df, temp_df = train_test_split(
        df,
        test_size=0.30,
        random_state=SEED,
        stratify=df["label"],
    )

    # Second split:
    # validation = 15%
    # test = 15%
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=SEED,
        stratify=temp_df["label"],
    )

    print_label_distribution(train_df, "Train set")
    print_label_distribution(val_df, "Validation set")
    print_label_distribution(test_df, "Test set")

    save_csv(train_df, TRAIN_PATH)
    save_csv(val_df, VAL_PATH)
    save_csv(test_df, TEST_PATH)


# =========================
# OOD Dataset Processing
# =========================

def prepare_ood_dataset() -> None:
    """
    Load OOD dataset, clean it, map labels, and save as data/processed/ood.csv.
    """
    print("\nLoading OOD dataset...")
    print(f"Path: {RAW_OOD_PATH}")

    ood_df = pd.read_csv(RAW_OOD_PATH)
    ood_df = clean_column_names(ood_df)

    # If gold_label exists, use it as the main label column
    if "gold_label" in ood_df.columns:
        ood_df["label"] = ood_df["gold_label"]

    required_columns = ["text", "label"]
    validate_columns(ood_df, required_columns)

    # Keep common useful columns if they exist
    useful_columns = []

    for col in ["text", "keyword", "label", "category"]:
        if col in ood_df.columns:
            useful_columns.append(col)

    ood_df = ood_df[useful_columns].copy()

    # Remove missing text or label
    ood_df = ood_df.dropna(subset=["text", "label"])

    # Remove duplicate text rows
    ood_df = ood_df.drop_duplicates(subset=["text"])

    # Map labels to 0 and 1
    ood_df = map_labels(ood_df)

    print_label_distribution(ood_df, "OOD dataset")

    save_csv(ood_df, OOD_PATH)


# =========================
# Main Function
# =========================

def main() -> None:
    """
    Run all data preparation steps.

    Usage:
        python -m src.data_utils
    """
    print("Starting data preparation...")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    prepare_main_dataset()
    prepare_ood_dataset()

    print("\nData preparation finished successfully.")


if __name__ == "__main__":
    main()