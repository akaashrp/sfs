from __future__ import annotations

import sys

import pandas as pd

from scripts.prep import split_csv


def test_custom_glob_and_output_directory(monkeypatch, tmp_path):
    source_root = tmp_path / "source"
    output_root = tmp_path / "splits"
    source_root.mkdir()
    source = source_root / "batch_stats_ministral3-3b.csv"
    pd.DataFrame({"row_id": list(range(10))}).to_csv(source, index=False)
    pd.DataFrame({"row_id": [999]}).to_csv(
        source_root / "batch_stats_ministral3-3b_train.csv", index=False
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "split_csv.py",
            "--root",
            str(source_root),
            "--glob",
            "batch_stats_ministral3-*.csv",
            "--seed",
            "69",
            "--train-fraction",
            "0.8",
            "--output-dir",
            str(output_root),
        ],
    )
    split_csv.main()

    train = pd.read_csv(output_root / "batch_stats_ministral3-3b_train.csv")
    test = pd.read_csv(output_root / "batch_stats_ministral3-3b_test.csv")
    assert len(train) == 8
    assert len(test) == 2
    assert set(train["row_id"]).isdisjoint(set(test["row_id"]))
    assert set(train["row_id"]) | set(test["row_id"]) == set(range(10))
