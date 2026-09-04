from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

from scripts.prep import quality_metrics


MODELS = ("ministral3-3b", "ministral3-8b", "ministral3-14b")


def _write_generation_file(path: Path, model_name: str, count: int = 4) -> None:
    with path.open("w", encoding="utf-8") as dst:
        for index in range(count):
            record = {
                "bucket": "alpaca",
                "prompt_index": index,
                "prompt": f"prompt {index}",
                "prompt_metadata": {
                    "dataset_id": "tatsu-lab/alpaca",
                    "example_id": f"tatsu-lab/alpaca:train:{index}",
                    "ref_output": f"reference {index}",
                    "bucket": "alpaca",
                },
                "response": {"output_text": f"{model_name} response {index}"},
            }
            dst.write(json.dumps(record))
            dst.write("\n")


def _read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as src:
        return [json.loads(line) for line in src if line.strip()]


def test_grouped_judge_returns_and_logs_realized_alias_mapping(
    monkeypatch,
    caplog,
) -> None:
    response = SimpleNamespace(
        candidates=[
            SimpleNamespace(
                content=SimpleNamespace(
                    parts=[SimpleNamespace(text='{"scores":{"A":1,"B":2,"C":3}}')]
                )
            )
        ]
    )
    client = SimpleNamespace(
        models=SimpleNamespace(generate_content=lambda **_: response)
    )
    monkeypatch.setattr(quality_metrics, "_get_gemini_client", lambda: client)
    monkeypatch.setattr(quality_metrics.random, "shuffle", lambda items: items.reverse())

    result = quality_metrics.judge_scores_for_prompt_group(
        prompt="prompt",
        gold="reference",
        example_id="example:1",
        candidates_by_model={
            "small": "small response",
            "medium": "medium response",
            "large": "large response",
        },
    )

    assert result.alias_to_model == {
        "A": "large",
        "B": "medium",
        "C": "small",
    }
    assert result.scores_by_model == {
        "large": 1.0,
        "medium": 2.0,
        "small": 3.0,
    }
    assert (
        'Group-judging example_id=example:1 alias_to_model={"A": "large", '
        '"B": "medium", "C": "small"}'
        in caplog.text
    )


def test_grouped_judge_retries_failures_individually_then_imputes(
    tmp_path: Path,
    monkeypatch,
) -> None:
    inputs = {}
    outputs = {}
    for model_name in MODELS:
        inputs[model_name] = tmp_path / f"{model_name}.jsonl"
        outputs[model_name] = tmp_path / f"{model_name}_scored.jsonl"
        _write_generation_file(inputs[model_name], model_name)

    attempts: Counter[str] = Counter()

    alias_to_model = {
        chr(ord("A") + index): model_name
        for index, model_name in enumerate(sorted(MODELS))
    }

    def fake_grouped_judge(
        *, example_id: str, candidates_by_model: dict, **_
    ) -> quality_metrics.GroupedJudgeResult:
        attempts[example_id] += 1
        index = int(example_id.rsplit(":", 1)[-1])
        attempt = attempts[example_id]

        if index == 1 and attempt < 3:
            raise RuntimeError("transient grouped failure")
        if index == 2 and attempt <= 3:
            raise RuntimeError("requires individual fallback")
        if index == 3:
            raise RuntimeError("persistent judge refusal")

        return quality_metrics.GroupedJudgeResult(
            scores_by_model={
                model_name: float(2 + model_index * 2 + index)
                for model_index, model_name in enumerate(sorted(candidates_by_model))
            },
            alias_to_model=alias_to_model,
        )

    monkeypatch.setattr(
        quality_metrics,
        "judge_scores_for_prompt_group",
        fake_grouped_judge,
    )

    summary = quality_metrics.annotate_bucket_group_with_quality(
        inputs,
        outputs,
        judge_concurrency=2,
        judge_retries=3,
        individual_retries=3,
        retry_sleep_s=0,
    )

    assert attempts["tatsu-lab/alpaca:train:0"] == 1
    assert attempts["tatsu-lab/alpaca:train:1"] == 3
    assert attempts["tatsu-lab/alpaca:train:2"] == 4
    assert attempts["tatsu-lab/alpaca:train:3"] == 6

    assert summary == {
        "total_groups": 4,
        "pending_groups": 4,
        "batch_scored_groups": 2,
        "individual_fallback_groups": 2,
        "individual_recovered_groups": 1,
        "imputed_groups": 1,
        "judged_records_written": 9,
        "imputed_records_written": 3,
        "reused_records": 0,
    }

    for model_index, model_name in enumerate(sorted(MODELS)):
        records = _read_jsonl(outputs[model_name])
        assert [row["prompt_index"] for row in records] == [0, 1, 2, 3]

        expected_mean = sum(
            (2 + model_index * 2 + index) / 10.0 for index in range(3)
        ) / 3
        imputed = records[-1]
        for judged in records[:-1]:
            assert judged["judge_alias_to_model"] == alias_to_model
        assert imputed["quality"] == expected_mean
        assert imputed["quality_metric"] == "judge_default_bucket_mean"
        assert imputed["quality_imputed"] is True
        assert imputed["quality_imputed_reason"] == "judge_refusal_or_unavailable"
        assert "judge_alias_to_model" not in imputed


def test_grouped_judge_rejects_invalid_retry_configuration() -> None:
    for kwargs in (
        {"judge_concurrency": 0},
        {"judge_retries": 0},
        {"individual_retries": 0},
        {"retry_sleep_s": -1},
    ):
        try:
            quality_metrics.annotate_bucket_group_with_quality({}, {}, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Expected ValueError for {kwargs}")
