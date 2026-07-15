#!/usr/bin/env python3
"""3PLM training script.

Trains a new model by selecting exactly three random words from input text.
"""

from __future__ import annotations

import argparse
import random
import re
from pathlib import Path

PARAMETER_COUNT = 3
WORD_PATTERN = re.compile(r"[^\W_]+(?:['’-][^\W_]+)*", re.UNICODE)


def tokenize(training_data: str) -> list[str]:
    """Extract words from training data while ignoring punctuation."""
    return WORD_PATTERN.findall(training_data)


def train(training_data: str) -> list[str]:
    """Select exactly three random words to serve as model parameters."""
    words = tokenize(training_data)

    if len(words) < PARAMETER_COUNT:
        raise ValueError(
            f"Training data must contain at least {PARAMETER_COUNT} words; "
            f"found {len(words)}."
        )

    return random.sample(words, k=PARAMETER_COUNT)


def save_model(parameters: list[str], output_path: Path) -> None:
    """Write one model parameter per line."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(parameters) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train an enterprise-grade 3-parameter language model."
    )
    parser.add_argument("trainingdata", help="Text used to train the model.")
    parser.add_argument("output", type=Path, help="Output path for the .mlm model.")
    parser.add_argument(
        "--seed",
        type=int,
        help="Optional random seed for reproducible model training.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    try:
        parameters = train(args.trainingdata)
        save_model(parameters, args.output)
    except (ValueError, OSError) as error:
        raise SystemExit(f"Error: {error}") from error

    print(f"Successfully trained a {PARAMETER_COUNT}-parameter model: {args.output}")
    print(f"Parameters: {', '.join(parameters)}")


if __name__ == "__main__":
    main()
