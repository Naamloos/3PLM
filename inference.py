#!/usr/bin/env python3
"""3PLM inference engine.

Completes a prompt by sampling random words from a trained `.mlm` model file.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path


def load_model(model_path: Path) -> list[str]:
    """Load model parameters from a whitespace-separated text file."""
    if not model_path.is_file():
        raise FileNotFoundError(f"Model file does not exist: {model_path}")

    parameters = model_path.read_text(encoding="utf-8").split()
    if not parameters:
        raise ValueError(f"Model file is empty: {model_path}")

    return parameters


def complete(query: str, parameters: list[str], word_count: int = 12) -> str:
    """Complete a query by randomly sampling words from the model."""
    generated = " ".join(random.choices(parameters, k=word_count))
    separator = "" if not query or query.endswith((" ", "\n", "\t")) else " "
    return f"{query}{separator}{generated}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run inference using a state-of-the-art 3-parameter language model."
    )
    parser.add_argument("query", help="Input text to complete.")
    parser.add_argument("model", type=Path, help="Path to the .mlm model file.")
    parser.add_argument(
        "--words",
        type=int,
        default=12,
        help="Number of words to generate (default: 12).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Optional random seed for reproducible enterprise inference.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.words < 1:
        raise SystemExit("Error: --words must be at least 1.")

    if args.seed is not None:
        random.seed(args.seed)

    try:
        parameters = load_model(args.model)
        print(complete(args.query, parameters, args.words))
    except (FileNotFoundError, ValueError, OSError) as error:
        raise SystemExit(f"Error: {error}") from error


if __name__ == "__main__":
    main()
