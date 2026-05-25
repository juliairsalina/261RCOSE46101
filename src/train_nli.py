import argparse
from pathlib import Path

from src.train_transformer import train_transformer


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--experiment_id", type=str, required=True)
    parser.add_argument("--train_file", type=str, required=True)

    parser.add_argument(
        "--model_name",
        type=str,
        default=None,
        help="Optional custom NLI checkpoint.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    train_transformer(
        experiment_id=args.experiment_id,
        train_file=Path(args.train_file),
        model_family="nli",
        model_name=args.model_name,
    )


if __name__ == "__main__":
    main()