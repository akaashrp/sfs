import json

from scripts.eval.augment_router_actual_accuracy import load_quality_index


def test_quality_index_accepts_direct_model_scoring_layout(tmp_path):
    model_dir = tmp_path / "ministral3-3b"
    model_dir.mkdir()
    record = {
        "model_label": "ministral3-3b",
        "bucket": "alpaca",
        "prompt_metadata": {"example_id": "dataset:train:7"},
        "quality": 0.8,
    }
    (model_dir / "alpaca_scored.jsonl").write_text(
        json.dumps(record) + "\n",
        encoding="utf-8",
    )

    assert load_quality_index(tmp_path) == {
        ("ministral3-3b", "alpaca", "dataset:train:7"): 0.8
    }


def test_quality_index_keeps_legacy_nested_scoring_layout(tmp_path):
    scored_dir = tmp_path / "qwen3-8b" / "scored"
    scored_dir.mkdir(parents=True)
    record = {
        "bucket": "alpaca",
        "prompt_metadata": {"example_id": "dataset:train:9"},
        "quality": 0.9,
    }
    (scored_dir / "alpaca_scored.jsonl").write_text(
        json.dumps(record) + "\n",
        encoding="utf-8",
    )

    assert load_quality_index(tmp_path) == {
        ("qwen3-8b", "alpaca", "dataset:train:9"): 0.9
    }
