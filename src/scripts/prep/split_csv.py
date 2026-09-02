#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from sfs_core.paths import EXPERIMENTS_ROOT

DEFAULT_ROOT = EXPERIMENTS_ROOT / "batch_stats" / "batch_stats_2048_cs"
DEFAULT_RANDOM_SEED = 69


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Shuffle each batch-stats CSV and emit 80/20 train/test splits."
    )
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--seed", type=int, default=DEFAULT_RANDOM_SEED)
    parser.add_argument(
        "--glob",
        default="batch_stats_qwen3*.csv",
        help="Input filename glob under --root (default: %(default)s).",
    )
    parser.add_argument(
        "--train-fraction",
        type=float,
        default=0.8,
        help="Fraction of shuffled rows written to each training split.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for splits; defaults to --root.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Input directory not found: {root}")
    if not 0.0 < args.train_fraction < 1.0:
        raise ValueError("--train-fraction must be between 0 and 1")
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else root
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = [
        path
        for path in sorted(root.glob(args.glob))
        if not path.stem.endswith(("_train", "_test"))
    ]
    if not paths:
        raise FileNotFoundError(
            f"No input CSVs matched {args.glob!r} under {root}"
        )
    for path in paths:
        print(f"Processing {path.name}")
        df = pd.read_csv(path).sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
        cut = int(args.train_fraction * len(df))
        train = df.iloc[:cut]
        test = df.iloc[cut:]
        train_path = output_dir / f"{path.stem}_train.csv"
        test_path = output_dir / f"{path.stem}_test.csv"
        train.to_csv(train_path, index=False)
        test.to_csv(test_path, index=False)
        print(
            f"{path.name} -> {train_path.name} {len(train)} | "
            f"{test_path.name} {len(test)} (random_seed={args.seed})"
        )


if __name__ == "__main__":
    main()
