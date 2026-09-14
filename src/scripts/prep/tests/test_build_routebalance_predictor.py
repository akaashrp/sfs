from __future__ import annotations

import json

import pytest

from scripts.prep.build_routebalance_predictor import (
    CalibrationPrompt,
    load_calibration,
    split_calibration,
    validation_metrics,
)


def archive(tmp_path, *, mutation=None):
    for model in ("small", "large"):
        folder = tmp_path / model
        folder.mkdir()
        records = []
        for index in range(2):
            record = {
                "bucket": "alpaca", "model_label": model, "prompt_index": index,
                "prompt_metadata": {"example_id": f"some/dataset:train:{index}"},
                "prompt": f"prompt {index}", "quality": .2 if model == "small" else .8,
                "quality_metric": "judge", "max_completion_tokens": 8192,
                "response": {"completion_tokens": index + 1}, "error": None,
            }
            if mutation:
                mutation(record, model, index)
            records.append(record)
        (folder / "alpaca_scored.jsonl").write_text("".join(json.dumps(record) + "\n" for record in records))
    return tmp_path


def load(root):
    return load_calibration(root, model_labels=["small", "large"], buckets=["alpaca"], expected_per_bucket=2)


def test_calibration_preserves_canonical_ids_and_model_order(tmp_path):
    rows, metadata = load(archive(tmp_path))
    assert rows[0].example_id == "some/dataset:train:0"
    assert rows[0].qualities == [.2, .8]
    assert rows[1].output_lengths == [2, 2]
    assert len(metadata["source_files"]) == 2
    assert metadata["calibration_boundary"]["prompt_index_max_exclusive"] == 2


@pytest.mark.parametrize("field,value,pattern", [
    ("prompt_index", 2500, "outside calibration"),
    ("quality", 1.1, "quality"),
    ("quality", float("nan"), "quality"),
    ("error", "failed", "failed generation"),
    ("prompt", "different model prompt", "text/index mismatch"),
    ("max_completion_tokens", 9000, "completion cap"),
    ("response", {"completion_tokens": 8193}, "completion_tokens"),
    ("prompt_metadata", {"example_id": "unknown:train:0"}, "misaligned canonical"),
])
def test_calibration_refuses_leakage_corruption_and_misalignment(tmp_path, field, value, pattern):
    def mutate(record, model, index):
        if model == "large" and index == 0:
            record[field] = value
    with pytest.raises(ValueError, match=pattern):
        load(archive(tmp_path, mutation=mutate))


def test_calibration_rejects_duplicate_indices(tmp_path):
    def mutate(record, model, index):
        if model == "small":
            record["prompt_index"] = 0
            record["prompt_metadata"]["example_id"] = "arbitrary-id-" + str(index)
    with pytest.raises(ValueError, match="duplicate prompt_index"):
        load(archive(tmp_path, mutation=mutate))


def test_holdout_canonical_ids_cannot_hide_behind_local_row_numbers(tmp_path):
    def mutate(record, model, index):
        record["prompt_metadata"]["example_id"] = f"some/dataset:train:{2500 + index}"
    with pytest.raises(ValueError, match="canonical ID index"):
        load(archive(tmp_path, mutation=mutate))
    with pytest.raises(ValueError, match="2500"):
        load_calibration(tmp_path, expected_per_bucket=2501)


def row(index, prompt=None):
    return CalibrationPrompt("alpaca", f"data:train:{index}", index, prompt or f"prompt {index}", [.2], [4], [8192])


def test_validation_manifest_intersection_and_duplicate_prompt_isolation(tmp_path):
    paths = [tmp_path / "quality.json", tmp_path / "length.json"]
    paths[0].write_text(json.dumps({"example_ids": ["data:train:1", "data:train:3"]}))
    paths[1].write_text(json.dumps({"example_ids": ["data:train:1", "data:train:2"]}))
    rows = [row(0), row(1), row(2, "prompt 1"), row(3)]
    train, validation, metadata = split_calibration(rows, validation_id_files=paths)
    assert [item.prompt_index for item in train] == [0, 3]
    assert [item.prompt_index for item in validation] == [1, 2]
    assert metadata["validation_duplicate_text_expansion"] == 1
    assert metadata["validation_selection_ids"] == ["data:train:1"]
    assert len(metadata["validation_manifest_sources"]) == 2


def test_validation_deterministic_by_prompt_not_record_order():
    rows = [row(index) for index in range(100)]
    train, validation, _ = split_calibration(rows)
    reverse_train, reverse_validation, _ = split_calibration(list(reversed(rows)))
    assert {item.example_id for item in train} == {item.example_id for item in reverse_train}
    assert {item.example_id for item in validation} == {item.example_id for item in reverse_validation}
    assert not {item.prompt for item in train} & {item.prompt for item in validation}


def test_invalid_or_unmatched_validation_manifest_rejected(tmp_path):
    path = tmp_path / "ids.json"
    path.write_text(json.dumps({"example_ids": ["data:holdout:2500"]}))
    with pytest.raises(ValueError, match="absent from calibration"):
        split_calibration([row(0), row(1)], validation_id_files=[path])
    path.write_text(json.dumps({"example_ids": [1]}))
    with pytest.raises(ValueError, match="canonical string"):
        split_calibration([row(0), row(1)], validation_id_files=[path])


def test_validation_metrics_report_quality_and_length_errors():
    result = validation_metrics([row(0), row(1)], [
        {"small": {"quality": .4, "output_tokens": 6}},
        {"small": {"quality": .0, "output_tokens": 0}},
    ], ["small"])
    assert result["all"]["small"]["quality_mae"] == pytest.approx(.2)
    assert result["alpaca"]["small"]["length_mae_tokens"] == 3
    assert result["all"]["small"]["num_prompts"] == 2
