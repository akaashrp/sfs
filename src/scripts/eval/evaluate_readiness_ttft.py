"""Evaluate router-time TTFT predictions from readiness validation runs."""

from __future__ import annotations

import argparse
import ast
import json
import math
import re
from pathlib import Path
from typing import Any, Mapping, Sequence


ENGINE_LINE_PATTERN = re.compile(
    r"\brequest_id=(?P<request_id>\S+).*?"
    r"\bready_ts_s=(?P<ready>[0-9.eE+-]+).*?"
    r"\bqueued_ts_s=(?P<queued>[0-9.eE+-]+).*?"
    r"\bfirst_token_ts_s=(?P<first>[0-9.eE+-]+)"
)


def _read_engine_times(path: Path) -> dict[str, dict[str, float]]:
    values: dict[str, dict[str, float]] = {}
    with path.open("r", encoding="utf-8", errors="ignore") as source:
        for line in source:
            match = ENGINE_LINE_PATTERN.search(line)
            if match is None:
                continue
            values[match.group("request_id")] = {
                name: float(match.group(name))
                for name in ("ready", "queued", "first")
            }
    return values


def _read_router_records(path: Path) -> dict[str, Mapping[str, Any]]:
    values: dict[str, Mapping[str, Any]] = {}
    with path.open("r", encoding="utf-8", errors="ignore") as source:
        for line in source:
            try:
                record = ast.literal_eval(line)
            except (SyntaxError, ValueError):
                continue
            if not isinstance(record, Mapping):
                continue
            request_id = record.get("request_id")
            if isinstance(request_id, str):
                values[request_id] = record
    return values


def _metrics(actual: Sequence[float], predicted: Sequence[float]) -> dict[str, Any]:
    errors = [
        predicted_value - actual_value
        for actual_value, predicted_value in zip(actual, predicted)
    ]
    return {
        "count": len(errors),
        "mae_ms": sum(abs(error) for error in errors) / len(errors),
        "rmse_ms": math.sqrt(
            sum(error * error for error in errors) / len(errors)
        ),
        "bias_ms": sum(errors) / len(errors),
        "actual_mean_ms": sum(actual) / len(actual),
        "predicted_mean_ms": sum(predicted) / len(predicted),
    }


def load_case(result_path: Path) -> dict[str, Any]:
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    run = payload["wait_gof"]["runs"][0]
    router_records = _read_router_records(Path(run["request_log_path"]))
    engine_times = _read_engine_times(
        result_path.parent / "actual_wait_times.log"
    )
    actual: list[float] = []
    predicted: list[float] = []

    for item in run["per_request"]:
        scheduler_request_id = item.get("scheduler_request_id")
        response_id = item.get("response_id")
        if not isinstance(scheduler_request_id, str) or not isinstance(
            response_id,
            str,
        ):
            continue
        record = router_records.get(scheduler_request_id)
        observed = engine_times.get(response_id)
        if record is None or observed is None:
            continue
        record_payload = record.get("payload")
        predicted_wait_ms = record.get("wait_time_ms")
        if not isinstance(record_payload, Mapping) or not isinstance(
            predicted_wait_ms,
            (int, float),
        ):
            continue
        reports = record_payload.get("reports")
        if (
            not isinstance(reports, list)
            or not reports
            or not isinstance(reports[0], Mapping)
        ):
            continue
        simulation_timestamp = reports[0].get("simulation_timestamp")
        if not isinstance(simulation_timestamp, (int, float)):
            continue
        actual.append(
            (observed["first"] - float(simulation_timestamp)) * 1000.0
        )
        predicted.append(float(predicted_wait_ms))

    expected = len(run["per_request"])
    if len(actual) < expected * 0.9:
        raise RuntimeError(
            f"Matched only {len(actual)} of {expected} requests for "
            f"{result_path}"
        )
    return {
        "result_path": str(result_path),
        "target": "(first_token_ts_s - simulation_timestamp) * 1000",
        "metrics": _metrics(actual, predicted),
    }


def _parse_case(value: str) -> tuple[str, Path]:
    label, separator, path = value.partition("=")
    if not separator or not label or not path:
        raise argparse.ArgumentTypeError("Expected --case LABEL=RESULT_JSON")
    return label, Path(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--case",
        action="append",
        type=_parse_case,
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cases = {label: load_case(path) for label, path in args.case}
    output: dict[str, Any] = {"cases": cases}
    if len(cases) == 2:
        labels = list(cases)
        baseline = cases[labels[0]]["metrics"]
        candidate = cases[labels[1]]["metrics"]
        output["comparison"] = {
            "baseline": labels[0],
            "candidate": labels[1],
            "candidate_minus_baseline": {
                name: candidate[name] - baseline[name]
                for name in ("mae_ms", "rmse_ms", "bias_ms")
            },
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
