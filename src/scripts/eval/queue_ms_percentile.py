"""Compute queue_ms percentiles from a per-request wait log."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from sfs_core.shared.shared_experiment_helpers import _compute_percentile

QUEUE_PATTERN = re.compile(r"queue_ms=([0-9.+-eE]+)")


def _read_queue_values(path: Path) -> tuple[list[float], int]:
    values: list[float] = []
    bad_value_count = 0
    with path.open("r", encoding="utf-8", errors="ignore") as src:
        for line in src:
            match = QUEUE_PATTERN.search(line)
            if not match:
                continue
            try:
                values.append(float(match.group(1)))
            except ValueError:
                bad_value_count += 1
    return values, bad_value_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute a queue_ms percentile from a wait log file."
    )
    parser.add_argument(
        "wait_log",
        type=Path,
        help="Path to a per-request wait log containing queue_ms=<value> entries.",
    )
    parser.add_argument(
        "--percentile",
        type=float,
        default=95,
        help="Percentile to compute in [0, 100] (default: %(default)s).",
    )
    parser.add_argument(
        "--precision",
        type=int,
        default=6,
        help="Decimal precision for printed percentile values (default: %(default)s).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.percentile < 0 or args.percentile > 100:
        raise SystemExit("--percentile must be between 0 and 100.")
    if args.precision < 0:
        raise SystemExit("--precision must be >= 0.")

    wait_log = args.wait_log.expanduser()
    if not wait_log.exists():
        raise SystemExit(f"Wait log not found: {wait_log}")
    if not wait_log.is_file():
        raise SystemExit(f"Expected a file path: {wait_log}")

    values, bad_value_count = _read_queue_values(wait_log)
    if not values:
        raise SystemExit(f"No queue_ms values found in {wait_log}")

    q = args.percentile / 100.0
    percentile_value = _compute_percentile(values, q)
    if percentile_value is None:
        raise SystemExit(f"No queue_ms values found in {wait_log}")

    print(f"path={wait_log}")
    print(f"count={len(values)}")
    if bad_value_count:
        print(f"skipped_non_numeric_queue_ms={bad_value_count}")
    print(
        f"p{args.percentile:g}_queue_ms={percentile_value:.{args.precision}f}"
    )


if __name__ == "__main__":
    main()
