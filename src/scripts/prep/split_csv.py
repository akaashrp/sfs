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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Input directory not found: {root}")

    for path in sorted(root.glob("batch_stats_qwen3*.csv")):
        print(f"Processing {path.name}")
        df = pd.read_csv(path).sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
        cut = int(0.8 * len(df))
        train = df.iloc[:cut]
        test = df.iloc[cut:]
        train_path = root / f"{path.stem}_train.csv"
        test_path = root / f"{path.stem}_test.csv"
        train.to_csv(train_path, index=False)
        test.to_csv(test_path, index=False)
        print(
            f"{path.name} -> {train_path.name} {len(train)} | "
            f"{test_path.name} {len(test)} (random_seed={args.seed})"
        )


if __name__ == "__main__":
    main()
