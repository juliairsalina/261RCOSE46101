import argparse
from pathlib import Path

from src.train_transformer import train_transformer


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--experiment_id", type=str, required=True)
    parser.add_argument("--train_file", type=str, required=True)

    parser.add_argument(
        "--replay_file",
        type=str,
        default=None,
        help="Optional replay examples CSV for RoBERTa + Experience Replay.",
    )

    parser.add_argument(
        "--replay_repeat",
        type=int,
        default=0,
        help="How many times to repeat replay examples.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    replay_file = Path(args.replay_file) if args.replay_file else None

    train_transformer(
        experiment_id=args.experiment_id,
        train_file=Path(args.train_file),
        model_family="roberta",
        replay_file=replay_file,
        replay_repeat=args.replay_repeat,
    )


if __name__ == "__main__":
    main()