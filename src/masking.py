import random
import re

import pandas as pd

from src.config import TRAIN_PATH, MASKED_TRAIN_PATH, SEED


def mask_keyword_in_text(text, keyword, mask_token="[MASK]", p=0.5):
    """
    Replace keyword with [MASK] with probability p.

    If keyword is missing or not found in the text,
    the original text is returned.
    """
    if pd.isna(text) or pd.isna(keyword):
        return text

    text = str(text)
    keyword = str(keyword).strip()

    if keyword == "":
        return text

    # Check whether keyword exists in the text, ignoring uppercase/lowercase
    pattern = re.compile(re.escape(keyword), re.IGNORECASE)

    if not pattern.search(text):
        return text

    # Randomly decide whether to mask
    if random.random() < p:
        return pattern.sub(mask_token, text)

    return text


def create_masked_train_data(input_path=TRAIN_PATH, output_path=MASKED_TRAIN_PATH, p=0.5):
    """
    Load train.csv, mask shortcut keywords, and save train_masked.csv.
    """
    random.seed(SEED)

    print("Loading training data...")
    print(f"Path: {input_path}")

    df = pd.read_csv(input_path)

    required_columns = ["id", "text", "keyword", "label"]
    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        raise ValueError(f"Missing columns: {missing_columns}")

    df["text"] = df.apply(
        lambda row: mask_keyword_in_text(
            text=row["text"],
            keyword=row["keyword"],
            mask_token="[MASK]",
            p=p,
        ),
        axis=1,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"Saved masked training data to: {output_path}")
    print(f"Number of rows: {len(df)}")


def main():
    create_masked_train_data()


if __name__ == "__main__":
    main()