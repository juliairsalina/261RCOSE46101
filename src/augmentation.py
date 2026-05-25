import random

import pandas as pd

from src.config import (
    TRAIN_PATH,
    COUNTERFACTUAL_TRAIN_PATH,
    FULL_TRAIN_PATH,
    SEED,
)

from src.masking import mask_keyword_in_text


def get_keywords_from_train(df: pd.DataFrame) -> list[str]:
    """
    Get unique keywords from the training data.
    """
    keywords = (
        df["keyword"]
        .dropna()
        .astype(str)
        .str.strip()
        .replace("", pd.NA)
        .dropna()
        .unique()
        .tolist()
    )

    return sorted(set(keywords))


def make_counterfactual_row(
    row_id: int,
    text: str,
    keyword: str,
    label: int,
    category: str,
) -> dict:
    """
    Create one counterfactual row with consistent columns.
    """
    return {
        "id": f"cf_{row_id}",
        "text": text,
        "keyword": keyword,
        "label": label,
        "category": category,
        "source": "counterfactual",
    }


def create_counterfactual_rows(
    keywords: list[str],
    max_rows: int = 300,
) -> list[dict]:
    """
    Create counterfactual examples.

    Purpose:
    Teach the model that shortcut keywords can appear in both
    risky and non-risky contexts.

    Categories:
    - ambiguous
    - negation
    - temporal
    - figurative
    - negation_temporal
    """
    rows = []
    row_id = 0

    # General real-world examples that do not depend on the keyword list.
    fixed_examples = [
        # Figurative non-risky
        {
            "text": "I am dying of laughter.",
            "keyword": "dying",
            "label": 0,
            "category": "figurative",
        },
        {
            "text": "This homework is killing me.",
            "keyword": "killing",
            "label": 0,
            "category": "figurative",
        },
        {
            "text": "My phone battery died again.",
            "keyword": "died",
            "label": 0,
            "category": "figurative",
        },
        {
            "text": "The joke made me explode with laughter.",
            "keyword": "explode",
            "label": 0,
            "category": "figurative",
        },
        {
            "text": "I could kill for a good cup of coffee right now.",
            "keyword": "kill",
            "label": 0,
            "category": "figurative",
        },
        {
            "text": "I am dead tired after class.",
            "keyword": "dead",
            "label": 0,
            "category": "figurative",
        },

        # Negation non-risky
        {
            "text": "I don't want to die.",
            "keyword": "die",
            "label": 0,
            "category": "negation",
        },
        {
            "text": "I do not want to hurt myself.",
            "keyword": "hurt",
            "label": 0,
            "category": "negation",
        },
        {
            "text": "I won't hurt anyone.",
            "keyword": "hurt",
            "label": 0,
            "category": "negation",
        },
        {
            "text": "I can't imagine killing myself.",
            "keyword": "killing",
            "label": 0,
            "category": "negation",
        },

        # Temporal / recovery non-risky
        {
            "text": "I used to want to disappear, but I feel better now.",
            "keyword": "disappear",
            "label": 0,
            "category": "temporal",
        },
        {
            "text": "Last year I wanted to hurt myself, but now I am getting help.",
            "keyword": "hurt",
            "label": 0,
            "category": "temporal",
        },
        {
            "text": "I felt like dying before, but I do not feel that way anymore.",
            "keyword": "dying",
            "label": 0,
            "category": "negation_temporal",
        },
        {
            "text": "I used to think about killing myself, but I don't think that now.",
            "keyword": "killing",
            "label": 0,
            "category": "negation_temporal",
        },

        # Risky direct intent
        {
            "text": "I want to die tonight.",
            "keyword": "die",
            "label": 1,
            "category": "ambiguous",
        },
        {
            "text": "I might hurt myself if I am alone tonight.",
            "keyword": "hurt",
            "label": 1,
            "category": "ambiguous",
        },
        {
            "text": "I want to kill myself.",
            "keyword": "kill",
            "label": 1,
            "category": "ambiguous",
        },
    ]

    for example in fixed_examples:
        if len(rows) >= max_rows:
            return rows

        rows.append(
            make_counterfactual_row(
                row_id=row_id,
                text=example["text"],
                keyword=example["keyword"],
                label=example["label"],
                category=example["category"],
            )
        )
        row_id += 1

    # Keyword-based examples.
    # These generate matched risky/non-risky examples for every keyword.
    for keyword in keywords:
        keyword = str(keyword).strip()

        if keyword == "":
            continue

        examples = [
            # Risky / direct
            {
                "text": f"I want to {keyword}.",
                "keyword": keyword,
                "label": 1,
                "category": "ambiguous",
            },
            {
                "text": f"I might {keyword} myself tonight.",
                "keyword": keyword,
                "label": 1,
                "category": "ambiguous",
            },

            # Negation non-risky
            {
                "text": f"I do not want to {keyword}.",
                "keyword": keyword,
                "label": 0,
                "category": "negation",
            },
            {
                "text": f"I don't want to {keyword}.",
                "keyword": keyword,
                "label": 0,
                "category": "negation",
            },
            {
                "text": f"I won't {keyword}.",
                "keyword": keyword,
                "label": 0,
                "category": "negation",
            },
            {
                "text": f"I can't imagine wanting to {keyword}.",
                "keyword": keyword,
                "label": 0,
                "category": "negation",
            },

            # Temporal non-risky
            {
                "text": f"I used to feel like I wanted to {keyword}, but I feel safer now.",
                "keyword": keyword,
                "label": 0,
                "category": "temporal",
            },
            {
                "text": f"Last year I thought about wanting to {keyword}, but now I am doing better.",
                "keyword": keyword,
                "label": 0,
                "category": "temporal",
            },

            # Negation + temporal non-risky
            {
                "text": f"I used to feel like I wanted to {keyword}, but I do not feel that way now.",
                "keyword": keyword,
                "label": 0,
                "category": "negation_temporal",
            },
            {
                "text": f"I wanted to {keyword} before, but I don't want that anymore.",
                "keyword": keyword,
                "label": 0,
                "category": "negation_temporal",
            },
            {
                "text": f"I thought about wanting to {keyword} in the past, but I won't do that now.",
                "keyword": keyword,
                "label": 0,
                "category": "negation_temporal",
            },

            # Figurative non-risky
            {
                "text": f"This assignment makes me feel like I could {keyword}, but I am only joking.",
                "keyword": keyword,
                "label": 0,
                "category": "figurative",
            },
            {
                "text": f"I said I wanted to {keyword} because I was being dramatic.",
                "keyword": keyword,
                "label": 0,
                "category": "figurative",
            },
        ]

        for example in examples:
            if len(rows) >= max_rows:
                return rows

            rows.append(
                make_counterfactual_row(
                    row_id=row_id,
                    text=example["text"],
                    keyword=example["keyword"],
                    label=example["label"],
                    category=example["category"],
                )
            )
            row_id += 1

    return rows


def add_source_column(df: pd.DataFrame) -> pd.DataFrame:
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
        raise ValueError(
            f"Missing columns in {input_path}: {missing_columns}\n"
            f"Available columns: {list(train_df.columns)}"
        )

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

    combined_df = combined_df.drop_duplicates(subset=["text"]).reset_index(drop=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined_df.to_csv(output_path, index=False)

    print(f"Saved counterfactual training data to: {output_path}")
    print(f"Original rows: {len(train_df)}")
    print(f"Counterfactual rows before dedup: {len(counterfactual_df)}")
    print(f"Total rows after dedup: {len(combined_df)}")

    print("\nLabel distribution:")
    print(combined_df["label"].value_counts().sort_index())

    if "category" in combined_df.columns:
        print("\nCategory distribution:")
        print(combined_df["category"].value_counts().sort_index())

    if "source" in combined_df.columns:
        print("\nSource distribution:")
        print(combined_df["source"].value_counts())

    return combined_df


def create_full_train_data(
    counterfactual_df: pd.DataFrame,
    output_path=FULL_TRAIN_PATH,
    p=0.5,
):
    """
    Create train_full.csv.

    This applies keyword masking to train_counterfactual.csv.
    """
    random.seed(SEED)

    full_df = counterfactual_df.copy()
    original_texts = full_df["text"].astype(str).copy()

    full_df["text"] = full_df.apply(
        lambda row: mask_keyword_in_text(
            text=row["text"],
            keyword=row["keyword"],
            mask_token="[MASK]",
            p=p,
        ),
        axis=1,
    )

    masked_count = int((original_texts != full_df["text"].astype(str)).sum())

    output_path.parent.mkdir(parents=True, exist_ok=True)
    full_df.to_csv(output_path, index=False)

    print(f"Saved full training data to: {output_path}")
    print(f"Total rows: {len(full_df)}")
    print(f"Masked rows: {masked_count}")
    print(f"Mask probability: {p}")

    print("\nLabel distribution:")
    print(full_df["label"].value_counts().sort_index())

    if "category" in full_df.columns:
        print("\nCategory distribution:")
        print(full_df["category"].value_counts().sort_index())

    if "source" in full_df.columns:
        print("\nSource distribution:")
        print(full_df["source"].value_counts())


def main():
    counterfactual_df = create_counterfactual_train_data()
    create_full_train_data(counterfactual_df)


if __name__ == "__main__":
    main()