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
    df = df.copy()
    df.columns = df.columns.str.strip().str.lower()
    return df


def validate_columns(df: pd.DataFrame, required_columns: list[str], dataset_name: str) -> None:
    """
    Check whether the dataset contains all required columns.
    """
    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        raise ValueError(
            f"{dataset_name} is missing required columns: {missing_columns}\n"
            f"Available columns: {list(df.columns)}"
        )


def map_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert labels into numbers:
        non-risky -> 0
        risky     -> 1

    If labels are already 0/1, keep them.
    """
    df = df.copy()

    label_map = {
        "non-risky": 0,
        "risky": 1,
        "0": 0,
        "1": 1,
    }

    df["label"] = df["label"].astype(str).str.strip().str.lower()
    df["label"] = df["label"].map(label_map)

    # Remove rows with invalid labels.
    df = df.dropna(subset=["label"]).copy()
    df["label"] = df["label"].astype(int)

    return df


def clean_text_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean text-like columns.
    """
    df = df.copy()

    for col in ["id", "text", "keyword", "category"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    return df


def print_label_distribution(df: pd.DataFrame, name: str) -> None:
    """
    Print risky/non-risky label distribution.
    """
    print(f"\n{name} label distribution:")
    print(df["label"].value_counts().sort_index())
    print("0 = non-risky, 1 = risky")


def print_category_distribution(df: pd.DataFrame, name: str) -> None:
    """
    Print category distribution.
    """
    if "category" not in df.columns:
        return

    print(f"\n{name} category distribution:")
    print(df["category"].value_counts().sort_index())


def print_keyword_distribution_summary(df: pd.DataFrame, name: str) -> None:
    """
    Print keyword summary.
    """
    if "keyword" not in df.columns:
        return

    keyword_counts = df["keyword"].value_counts()

    print(f"\n{name} keyword summary:")
    print(f"Number of unique keywords: {keyword_counts.shape[0]}")
    print("Top keywords:")
    print(keyword_counts.head(10))


def audit_balanced_dataset(df: pd.DataFrame) -> None:
    """
    Audit keyword-label-category balance.

    For the new dataset, target design is:
        20 keywords
        2 labels
        5 categories
        13 rows per keyword-label-category group
        Total = 2600 rows

    For the old dataset without category, category will be 'unknown',
    so this audit is only informational.
    """
    if not all(col in df.columns for col in ["keyword", "label", "category"]):
        print("\nSkipping balance audit because keyword/label/category columns are missing.")
        return

    group_counts = (
        df.groupby(["keyword", "label", "category"])
        .size()
        .reset_index(name="count")
    )

    print("\nDataset balance audit:")
    print(f"Total rows: {len(df)}")
    print(f"Unique keywords: {df['keyword'].nunique()}")
    print(f"Unique labels: {df['label'].nunique()}")
    print(f"Unique categories: {df['category'].nunique()}")
    print(f"Number of keyword-label-category groups: {len(group_counts)}")

    print("\nGroup count summary:")
    print(group_counts["count"].describe())

    # Only check exact 13 count if this looks like the new 2600-row dataset.
    if len(df) == 2600 and df["category"].nunique() > 1:
        bad_groups = group_counts[group_counts["count"] != 13]

        if bad_groups.empty:
            print("All keyword-label-category groups have exactly 13 rows.")
        else:
            print("\nWarning: Some keyword-label-category groups do not have exactly 13 rows.")
            print(bad_groups)
    else:
        print("Skipping exact 13-per-group check because this does not look like the final 2600-row dataset.")


def create_stratify_key(df: pd.DataFrame) -> pd.Series:
    """
    Create stratification key.

    This is used for splitting only.
    The model does NOT train on category directly.
    """
    return (
        df["keyword"].astype(str).str.strip().str.lower()
        + "_"
        + df["label"].astype(str)
        + "_"
        + df["category"].astype(str).str.strip().str.lower()
    )


def choose_stratify_column(df: pd.DataFrame, preferred_column: str):
    """
    Safely choose stratification column.

    If some keyword-label-category groups are too small,
    fall back to label-only stratification.
    """
    counts = df[preferred_column].value_counts()

    if counts.min() >= 2:
        print(f"Using stratified split by {preferred_column}.")
        return df[preferred_column]

    print(f"Warning: Some {preferred_column} groups have fewer than 2 rows.")
    print("Falling back to stratified split by label only.")
    return df["label"]


def save_csv(df: pd.DataFrame, path) -> None:
    """
    Save dataframe to CSV.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Saved: {path}")


def load_raw_main_dataset() -> pd.DataFrame:
    """
    Load the main raw dataset.

    Supports:
        .xlsx
        .xls
        .csv
    """
    print("\nLoading main dataset...")
    print(f"Path: {RAW_DATA_PATH}")

    suffix = RAW_DATA_PATH.suffix.lower()

    if suffix in [".xlsx", ".xls"]:
        return pd.read_excel(RAW_DATA_PATH)

    if suffix == ".csv":
        return pd.read_csv(RAW_DATA_PATH)

    raise ValueError(
        f"Unsupported main dataset file type: {suffix}\n"
        "Please use .xlsx, .xls, or .csv"
    )


# =========================
# Main Dataset Processing
# =========================

def prepare_main_dataset() -> None:
    """
    Load main dataset, clean it, split into train/val/test, and save CSV files.

    Required columns for old dataset:
        id, text, keyword, label

    Required columns for new dataset:
        id, text, keyword, label, category

    If category is missing, it creates:
        category = "unknown"
    """
    df = load_raw_main_dataset()
    df = clean_column_names(df)

    # Old dataset only has id/text/keyword/label.
    required_columns = ["id", "text", "keyword", "label"]
    validate_columns(df, required_columns, dataset_name="Main dataset")

    # If the dataset does not have category, create placeholder category.
    # This does not affect model training because training uses text + label only.
    if "category" not in df.columns:
        print("Warning: 'category' column not found. Creating category='unknown'.")
        df["category"] = "unknown"

    useful_columns = ["id", "text", "keyword", "label", "category"]
    df = df[useful_columns].copy()

    # Basic cleaning.
    df = clean_text_columns(df)

    # Remove missing important values.
    df = df.dropna(subset=["text", "label", "keyword", "category"]).copy()

    # Remove exact duplicate texts.
    before_dedup = len(df)
    df = df.drop_duplicates(subset=["text"]).copy()
    after_dedup = len(df)

    print(f"\nRemoved duplicate text rows: {before_dedup - after_dedup}")

    # Map labels to 0/1.
    df = map_labels(df)

    # Print dataset checks.
    print_label_distribution(df, "Full main dataset")
    print_category_distribution(df, "Full main dataset")
    print_keyword_distribution_summary(df, "Full main dataset")
    audit_balanced_dataset(df)

    # Stratification is for split only, not model training.
    df["stratify_key"] = create_stratify_key(df)

    first_stratify = choose_stratify_column(df, "stratify_key")

    # First split:
    # train = 70%
    # temp = 30%
    train_df, temp_df = train_test_split(
        df,
        test_size=0.30,
        random_state=SEED,
        stratify=first_stratify,
    )

    # For val/test, try keyword-label-category stratification again.
    temp_stratify = choose_stratify_column(temp_df, "stratify_key")

    # Second split:
    # validation = 15%
    # test = 15%
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=SEED,
        stratify=temp_stratify,
    )

    # Remove helper column before saving.
    train_df = train_df.drop(columns=["stratify_key"])
    val_df = val_df.drop(columns=["stratify_key"])
    test_df = test_df.drop(columns=["stratify_key"])

    print_label_distribution(train_df, "Train set")
    print_label_distribution(val_df, "Validation set")
    print_label_distribution(test_df, "Test set")

    print_category_distribution(train_df, "Train set")
    print_category_distribution(val_df, "Validation set")
    print_category_distribution(test_df, "Test set")

    save_csv(train_df, TRAIN_PATH)
    save_csv(val_df, VAL_PATH)
    save_csv(test_df, TEST_PATH)


# =========================
# OOD Dataset Processing
# =========================

def prepare_ood_dataset() -> None:
    """
    Load OOD dataset, clean it, map labels, and save as data/processed/ood.csv.

    OOD required columns:
        text, label

    Optional useful columns:
        id, keyword, category

    If gold_label exists, it is used as label.
    If category is missing, it creates category='unknown'.
    """
    print("\nLoading OOD dataset...")
    print(f"Path: {RAW_OOD_PATH}")

    ood_df = pd.read_csv(RAW_OOD_PATH)
    ood_df = clean_column_names(ood_df)

    # If gold_label exists, use it as the main label column.
    if "gold_label" in ood_df.columns:
        ood_df["label"] = ood_df["gold_label"]

    required_columns = ["text", "label"]
    validate_columns(ood_df, required_columns, dataset_name="OOD dataset")

    if "category" not in ood_df.columns:
        print("Warning: OOD 'category' column not found. Creating category='unknown'.")
        ood_df["category"] = "unknown"

    useful_columns = []

    for col in ["id", "text", "keyword", "label", "category"]:
        if col in ood_df.columns:
            useful_columns.append(col)

    ood_df = ood_df[useful_columns].copy()
    ood_df = clean_text_columns(ood_df)

    # Remove missing text or label.
    ood_df = ood_df.dropna(subset=["text", "label"]).copy()

    # Remove exact duplicate texts.
    before_dedup = len(ood_df)
    ood_df = ood_df.drop_duplicates(subset=["text"]).copy()
    after_dedup = len(ood_df)

    print(f"\nRemoved duplicate OOD text rows: {before_dedup - after_dedup}")

    # Map labels to 0/1.
    ood_df = map_labels(ood_df)

    print_label_distribution(ood_df, "OOD dataset")
    print_category_distribution(ood_df, "OOD dataset")
    print_keyword_distribution_summary(ood_df, "OOD dataset")

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