import random

import pandas as pd

from src.config import (
    TRAIN_PATH,
    COUNTERFACTUAL_TRAIN_PATH,
    FULL_TRAIN_PATH,
    SEED,
)

from src.masking import mask_keyword_in_text


def get_keywords_from_train(df):
    """
    Get unique keywords from the training data.
    Also add common shortcut keywords to make sure we have enough examples.
    """
    keywords = df["keyword"].dropna().astype(str).str.strip().unique().tolist()

    default_keywords = [
        "die",
        "kill",
        "cut",
        "jump",
        "hurt",
    ]

    all_keywords = list(set(keywords + default_keywords))

    return all_keywords


def create_counterfactual_rows(keywords, max_rows=300):
    """
    Create small counterfactual examples.

    Purpose:
    Teach the model that the same risky keyword can appear in
    risky and non-risky contexts.
    """
    rows = []
    row_id = 0

    figurative_examples = [
        {
            "text": "I am dying of laughter",
            "keyword": "dying",
            "label": 0,
        },
        {
            "text": "This homework is killing me",
            "keyword": "killing",
            "label": 0,
        },
        {
            "text": "My phone battery died",
            "keyword": "died",
            "label": 0,
        },
        {
            "text": "The joke made me explode with laughter",
            "keyword": "explode",
            "label": 0,
        },
    ]

    for example in figurative_examples:
        rows.append(
            {
                "id": f"cf_{row_id}",
                "text": example["text"],
                "keyword": example["keyword"],
                "label": example["label"],
                "source": "counterfactual",
            }
        )
        row_id += 1

    for keyword in keywords:
        keyword = str(keyword).strip()

        if keyword == "":
            continue

        examples = [
            {
                "text": f"I want to {keyword}",
                "keyword": keyword,
                "label": 1,
            },
            {
                "text": f"I do not want to {keyword}",
                "keyword": keyword,
                "label": 0,
            },
            {
                "text": f"I used to feel like I wanted to {keyword}, but I do not feel that way now",
                "keyword": keyword,
                "label": 0,
            },
        ]

        for example in examples:
            if len(rows) >= max_rows:
                return rows

            rows.append(
                {
                    "id": f"cf_{row_id}",
                    "text": example["text"],
                    "keyword": example["keyword"],
                    "label": example["label"],
                    "source": "counterfactual",
                }
            )
            row_id += 1

    return rows


def add_source_column(df):
    """
    Add source column to original training data.
    """
    df = df.copy()

    if "source" not in df.columns:
        df["source"] = "original"
    else:
        df["source"] = df["source"].fillna("original")

    return df


def create_counterfactual_train_data(
    input_path=TRAIN_PATH,
    output_path=COUNTERFACTUAL_TRAIN_PATH,
    max_counterfactual_rows=300,
):
    """
    Create train_counterfactual.csv.

    This file contains:
    - original training data
    - generated counterfactual rows
    """
    random.seed(SEED)

    print("Loading training data...")
    print(f"Path: {input_path}")

    train_df = pd.read_csv(input_path)

    required_columns = ["id", "text", "keyword", "label"]
    missing_columns = [col for col in required_columns if col not in train_df.columns]

    if missing_columns:
        raise ValueError(f"Missing columns: {missing_columns}")

    train_df = add_source_column(train_df)

    keywords = get_keywords_from_train(train_df)
    counterfactual_rows = create_counterfactual_rows(
        keywords=keywords,
        max_rows=max_counterfactual_rows,
    )

    counterfactual_df = pd.DataFrame(counterfactual_rows)

    combined_df = pd.concat(
        [train_df, counterfactual_df],
        ignore_index=True,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined_df.to_csv(output_path, index=False)

    print(f"Saved counterfactual training data to: {output_path}")
    print(f"Original rows: {len(train_df)}")
    print(f"Counterfactual rows: {len(counterfactual_df)}")
    print(f"Total rows: {len(combined_df)}")

    return combined_df


def create_full_train_data(
    counterfactual_df,
    output_path=FULL_TRAIN_PATH,
    p=0.5,
):
    """
    Create train_full.csv.

    This applies keyword masking to the counterfactual training data.
    """
    random.seed(SEED)

    full_df = counterfactual_df.copy()

    full_df["text"] = full_df.apply(
        lambda row: mask_keyword_in_text(
            text=row["text"],
            keyword=row["keyword"],
            mask_token="[MASK]",
            p=p,
        ),
        axis=1,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    full_df.to_csv(output_path, index=False)

    print(f"Saved full training data to: {output_path}")
    print(f"Total rows: {len(full_df)}")


def main():
    counterfactual_df = create_counterfactual_train_data()
    create_full_train_data(counterfactual_df)


if __name__ == "__main__":
    main()