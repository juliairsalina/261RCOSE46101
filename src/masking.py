import random
import re

import pandas as pd

from src.config import TRAIN_PATH, MASKED_TRAIN_PATH, SEED


def mask_keyword_in_text(text, keyword, mask_token="[MASK]", p=0.5):
    if pd.isna(text) or pd.isna(keyword):
        return text

    text = str(text)
    keyword = str(keyword).strip()

    if keyword == "":
        return text

    # Match the keyword as a full word, not inside another word.
    pattern = re.compile(r"\b" + re.escape(keyword) + r"\b", re.IGNORECASE)

    if not pattern.search(text):
        return text

    if random.random() < p:
        return pattern.sub(mask_token, text)

    return text


def create_masked_train_data(
    input_path=TRAIN_PATH,
    output_path=MASKED_TRAIN_PATH,
    p=0.5,
):
    random.seed(SEED)

    print("Loading training data...")
    print(f"Path: {input_path}")

    df = pd.read_csv(input_path)

    required_columns = ["id", "text", "keyword", "label"]
    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        raise ValueError(
            f"Missing columns in {input_path}: {missing_columns}\n"
            f"Available columns: {list(df.columns)}"
        )

    df = df.copy()

    original_texts = df["text"].astype(str).copy()

    df["text"] = df.apply(
        lambda row: mask_keyword_in_text(
            text=row["text"],
            keyword=row["keyword"],
            mask_token="[MASK]",
            p=p,
        ),
        axis=1,
    )

    masked_count = int((original_texts != df["text"].astype(str)).sum())

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"Saved masked training data to: {output_path}")
    print(f"Total rows: {len(df)}")
    print(f"Masked rows: {masked_count}")
    print(f"Mask probability: {p}")

    print("\nLabel distribution:")
    print(df["label"].value_counts().sort_index())

    if "category" in df.columns:
        print("\nCategory distribution:")
        print(df["category"].value_counts().sort_index())


def main():
    create_masked_train_data()


if __name__ == "__main__":
    main()