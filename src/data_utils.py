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
    LABEL2ID,
)


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.str.strip().str.lower()
    return df


def validate_columns(df: pd.DataFrame, required_columns: list[str], dataset_name: str) -> None:
    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        raise ValueError(
            f"{dataset_name} is missing required columns: {missing_columns}\n"
            f"Available columns: {list(df.columns)}"
        )


def map_labels(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["label"] = df["label"].astype(str).str.strip().str.lower()

    label_map = {
        "non-risky": 0,
        "risky": 1,
        "0": 0,
        "1": 1,
    }

    df["label"] = df["label"].map(label_map)
    df = df.dropna(subset=["label"]).copy()
    df["label"] = df["label"].astype(int)

    return df


def clean_text_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for col in ["id", "text", "keyword", "category"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    return df


def print_label_distribution(df: pd.DataFrame, name: str) -> None:
    print(f"\n{name} label distribution:")
    print(df["label"].value_counts().sort_index())
    print("0 = non-risky, 1 = risky")


def print_category_distribution(df: pd.DataFrame, name: str) -> None:
    if "category" in df.columns:
        print(f"\n{name} category distribution:")
        print(df["category"].value_counts().sort_index())


def print_keyword_distribution(df: pd.DataFrame, name: str) -> None:
    if "keyword" in df.columns:
        print(f"\n{name} keyword distribution:")
        print(df["keyword"].value_counts().sort_index())


def audit_balanced_dataset(df: pd.DataFrame) -> None:
    """
    For your new dataset:
    20 keywords x 2 labels x 5 categories x 13 examples = 2600 rows.
    """
    required_columns = ["keyword", "label", "category"]

    if not all(col in df.columns for col in required_columns):
        print("\nSkipping balance audit because keyword/label/category is missing.")
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
    print(f"Keyword-label-category groups: {len(group_counts)}")

    bad_groups = group_counts[group_counts["count"] != 13]

    if bad_groups.empty:
        print("All keyword-label-category groups have exactly 13 rows.")
    else:
        print("\nWarning: Some groups do not have exactly 13 rows:")
        print(bad_groups)

    print("\nGroup count summary:")
    print(group_counts["count"].describe())


def create_stratify_key(df: pd.DataFrame) -> pd.Series:
    """
    Preserve keyword + label + category balance during train/val/test split.
    """
    return (
        df["keyword"].astype(str).str.strip().str.lower()
        + "_"
        + df["label"].astype(str)
        + "_"
        + df["category"].astype(str).str.strip().str.lower()
    )


def save_csv(df: pd.DataFrame, path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Saved: {path}")


def load_raw_main_dataset() -> pd.DataFrame:
    print("\nLoading main dataset...")
    print(f"Path: {RAW_DATA_PATH}")

    suffix = RAW_DATA_PATH.suffix.lower()

    if suffix in [".xlsx", ".xls"]:
        return pd.read_excel(RAW_DATA_PATH)

    if suffix == ".csv":
        return pd.read_csv(RAW_DATA_PATH)

    raise ValueError(f"Unsupported file type: {suffix}. Please use .xlsx, .xls, or .csv.")


def prepare_main_dataset() -> None:
    df = load_raw_main_dataset()
    df = clean_column_names(df)

    required_columns = ["id", "text", "keyword", "label", "category"]
    validate_columns(df, required_columns, dataset_name="Main dataset")

    df = df[required_columns].copy()
    df = clean_text_columns(df)

    df = df.dropna(subset=["text", "label", "keyword", "category"]).copy()

    before = len(df)
    df = df.drop_duplicates(subset=["text"]).copy()
    print(f"\nRemoved duplicate text rows: {before - len(df)}")

    df = map_labels(df)

    print_label_distribution(df, "Full main dataset")
    print_category_distribution(df, "Full main dataset")
    print_keyword_distribution(df, "Full main dataset")
    audit_balanced_dataset(df)

    df["stratify_key"] = create_stratify_key(df)

    train_df, temp_df = train_test_split(
        df,
        test_size=0.30,
        random_state=SEED,
        stratify=df["stratify_key"],
    )

    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=SEED,
        stratify=temp_df["stratify_key"],
    )

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


def prepare_ood_dataset() -> None:
    print("\nLoading OOD dataset...")
    print(f"Path: {RAW_OOD_PATH}")

    ood_df = pd.read_csv(RAW_OOD_PATH)
    ood_df = clean_column_names(ood_df)

    if "gold_label" in ood_df.columns:
        ood_df["label"] = ood_df["gold_label"]

    required_columns = ["text", "label"]
    validate_columns(ood_df, required_columns, dataset_name="OOD dataset")

    useful_columns = [
        col for col in ["id", "text", "keyword", "label", "category"]
        if col in ood_df.columns
    ]

    ood_df = ood_df[useful_columns].copy()
    ood_df = clean_text_columns(ood_df)

    ood_df = ood_df.dropna(subset=["text", "label"]).copy()

    before = len(ood_df)
    ood_df = ood_df.drop_duplicates(subset=["text"]).copy()
    print(f"\nRemoved duplicate OOD text rows: {before - len(ood_df)}")

    ood_df = map_labels(ood_df)

    print_label_distribution(ood_df, "OOD dataset")
    print_category_distribution(ood_df, "OOD dataset")
    print_keyword_distribution(ood_df, "OOD dataset")

    save_csv(ood_df, OOD_PATH)


def main() -> None:
    print("Starting data preparation...")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    prepare_main_dataset()
    prepare_ood_dataset()

    print("\nData preparation finished successfully.")


if __name__ == "__main__":
    main()