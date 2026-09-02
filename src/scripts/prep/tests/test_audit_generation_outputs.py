import json

from scripts.prep.audit_generation_outputs import audit_generation_outputs


def _record(example_id: str):
    return {
        "prompt_metadata": {"example_id": example_id},
        "error": None,
        "response": {
            "output_text": "answer",
            "completion_tokens": 1,
            "finish_reason": "stop",
        },
    }


def _write_jsonl(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )


def test_generation_audit_requires_complete_aligned_success(tmp_path):
    for model in ("small", "large"):
        _write_jsonl(
            tmp_path / model / "bucket.jsonl",
            [_record("dataset:0"), _record("dataset:1")],
        )

    result = audit_generation_outputs(
        tmp_path,
        model_names=("small", "large"),
        bucket_names=("bucket",),
        expected_per_bucket=2,
    )

    assert result["status"] == "PASS"
    assert result["errors"] == []


def test_generation_audit_rejects_request_errors_and_misalignment(tmp_path):
    bad = _record("dataset:2")
    bad["error"] = {"type": "RuntimeError"}
    bad["response"]["output_text"] = ""
    _write_jsonl(tmp_path / "small" / "bucket.jsonl", [_record("dataset:0")])
    _write_jsonl(tmp_path / "large" / "bucket.jsonl", [bad])

    result = audit_generation_outputs(
        tmp_path,
        model_names=("small", "large"),
        bucket_names=("bucket",),
        expected_per_bucket=1,
    )

    assert result["status"] == "FAIL"
    assert any("request_errors=1" in error for error in result["errors"])
    assert any("not aligned" in error for error in result["errors"])
