#!/usr/bin/env python3
"""Thin sweep wrapper for router experiments.

This wrapper runs multiple router points (delta, QPS, and/or lambda sweeps) in
a single Python process while reusing loaded InstanceClient objects across
points.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Optional, Sequence

THIS_DIR = Path(__file__).resolve().parent
SRC_ROOT = THIS_DIR.parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.append(str(SRC_ROOT))

_EXP_MODULE = None


def _exp():
    global _EXP_MODULE
    if _EXP_MODULE is None:
        from scripts.runs import experiments as exp_module

        _EXP_MODULE = exp_module
    return _EXP_MODULE


def _parse_wrapper_args(argv: Sequence[str]) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description=(
            "Sweep wrapper around experiments.py for router experiments. "
            "Unknown arguments are forwarded to experiments.py."
        )
    )
    parser.add_argument(
        "--sweep",
        type=str,
        choices=("delta", "qps", "lambda", "both", "qps_arrival"),
        default="both",
        help="Which sweep(s) to run.",
    )
    parser.add_argument(
        "--delta-values",
        nargs="+",
        type=float,
        default=[],
        help="Delta values for soft-utility sweep.",
    )
    parser.add_argument(
        "--qps-values",
        nargs="+",
        type=float,
        default=[],
        help="QPS values for fixed-utility sweep.",
    )
    parser.add_argument(
        "--lambda-values",
        nargs="+",
        type=float,
        default=[],
        help="Lambda values for fixed-QPS sweep.",
    )
    parser.add_argument(
        "--delta-zero-utilities",
        nargs="+",
        default=None,
        help=(
            "Utilities used for delta points where delta == 0. "
            "Defaults to all built-in utilities when omitted."
        ),
    )
    parser.add_argument(
        "--delta-nonzero-utilities",
        nargs="+",
        default=["soft"],
        help="Utilities used for delta points where delta != 0.",
    )
    parser.add_argument(
        "--qps-utilities",
        nargs="+",
        default=["hard", "hard_prefill_tps", "shortest_queue"],
        help="Utilities used for every QPS point.",
    )
    parser.add_argument(
        "--lambda-utilities",
        nargs="+",
        default=["hard_prefill_tps"],
        help="Utilities used for every lambda point.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for per-point JSON outputs and sweep manifest.",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="router_sweep",
        help="Filename prefix for generated outputs.",
    )
    parser.add_argument(
        "--keep-instances-open",
        action="store_true",
        help=(
            "Do not call InstanceClient.close() when the sweep wrapper exits. "
            "Useful when surrounding orchestration controls lifecycle."
        ),
    )
    parser.add_argument(
        "--arrival-sweep-ids",
        nargs="+",
        default=[],
        help="Arrival configuration ids for --sweep qps_arrival.",
    )
    parser.add_argument(
        "--arrival-sweep-processes",
        nargs="+",
        default=[],
        help="Arrival processes for --sweep qps_arrival.",
    )
    parser.add_argument(
        "--arrival-sweep-rate-ratios",
        nargs="+",
        type=float,
        default=[],
        help=(
            "Per-arrival MMPP-2 rate ratios for --sweep qps_arrival. "
            "Required only for entries where process=mmpp2."
        ),
    )
    parser.add_argument(
        "--arrival-sweep-mmpp2-high-fraction",
        type=float,
        default=None,
        help=(
            "Shared mmpp2 high-fraction used for all mmpp2 entries in "
            "--sweep qps_arrival. Defaults to experiments.py base argument."
        ),
    )
    parser.add_argument(
        "--arrival-sweep-mmpp2-correlation-time-s",
        type=float,
        default=None,
        help=(
            "Shared mmpp2 correlation time used for all mmpp2 entries in "
            "--sweep qps_arrival. Defaults to experiments.py base argument."
        ),
    )
    return parser.parse_known_args(argv)


def _parse_experiment_args(forwarded_argv: Sequence[str]) -> argparse.Namespace:
    exp_mod = _exp()
    original_argv = list(sys.argv)
    try:
        sys.argv = ["experiments.py", *list(forwarded_argv)]
        return exp_mod.parse_args()
    finally:
        sys.argv = original_argv


def _sanitize_tag(value: float) -> str:
    raw = f"{float(value):.12g}"
    raw = raw.replace("+", "")
    raw = raw.replace("-", "m")
    raw = raw.replace(".", "p")
    return raw


def _sanitize_label(label: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_-]+", "_", str(label)).strip("_")
    return cleaned or "label"


def _point_output_path(
    *,
    output_dir: Path,
    output_prefix: str,
    session_stamp: str,
    point_kind: str,
    point_value: float,
    point_index: int,
    point_label: Optional[str] = None,
) -> Path:
    tag = _sanitize_tag(point_value)
    label = f"_{_sanitize_label(point_label)}" if point_label else ""
    name = (
        f"{output_prefix}_{session_stamp}_"
        f"{point_kind}{tag}{label}_point{point_index:02d}.json"
    )
    return output_dir / name


def _request_set_payload(
    requests: list[Any],
    request_manifest: list[dict[str, Any]],
) -> dict[str, Any]:
    exp_mod = _exp()
    return {
        "num_requests": len(request_manifest),
        "slo_ms": exp_mod._metric_summary([float(req.latency_slo_ms) for req in requests]),
        "latency_slo_ms": exp_mod._metric_summary(
            [float(req.latency_slo_ms) for req in requests]
        ),
        "queue_slo_ms": exp_mod._metric_summary(
            [float(req.queue_slo_ms) for req in requests]
        ),
        "ttft_slo_ms": exp_mod._metric_summary([float(req.ttft_slo_ms) for req in requests]),
        "prompt_tokens": exp_mod._metric_summary(
            [float(req.prompt_tokens) for req in requests]
        ),
        "requests": request_manifest,
    }


def _build_run_points(
    wrapper_args: argparse.Namespace,
    base_args: argparse.Namespace,
) -> list[dict[str, Any]]:
    points: list[dict[str, Any]] = []

    if wrapper_args.sweep in {"delta", "both"}:
        if not wrapper_args.delta_values:
            raise ValueError("--delta-values is required when --sweep is delta/both.")
        all_utilities = list(_exp().BUILTIN_UTILITIES)
        delta_zero_utilities = (
            list(wrapper_args.delta_zero_utilities)
            if wrapper_args.delta_zero_utilities is not None
            else list(all_utilities)
        )
        for delta in wrapper_args.delta_values:
            delta_value = float(delta)
            if delta_value == 0.0:
                utilities = list(delta_zero_utilities)
                run_mode = "delta_zero"
            else:
                utilities = list(wrapper_args.delta_nonzero_utilities)
                run_mode = "nonzero_delta"
            points.append(
                {
                    "kind": "delta",
                    "value": delta_value,
                    "utilities": utilities,
                    "lambda_weight": float(base_args.lambda_weight),
                    "delta_weight": delta_value,
                    "request_rate_qps": float(base_args.request_rate_qps),
                    "run_mode": run_mode,
                }
            )

    if wrapper_args.sweep in {"qps", "both"}:
        if not wrapper_args.qps_values:
            raise ValueError("--qps-values is required when --sweep is qps/both.")
        for qps in wrapper_args.qps_values:
            points.append(
                {
                    "kind": "qps",
                    "value": float(qps),
                    "utilities": list(wrapper_args.qps_utilities),
                    "lambda_weight": float(base_args.lambda_weight),
                    "delta_weight": float(base_args.delta_weight),
                    "request_rate_qps": float(qps),
                    "run_mode": "qps_sweep",
                }
            )

    if wrapper_args.sweep == "qps_arrival":
        if not wrapper_args.qps_values:
            raise ValueError("--qps-values is required when --sweep is qps_arrival.")
        if not wrapper_args.arrival_sweep_ids:
            raise ValueError(
                "--arrival-sweep-ids is required when --sweep is qps_arrival."
            )
        if not wrapper_args.arrival_sweep_processes:
            raise ValueError(
                "--arrival-sweep-processes is required when --sweep is qps_arrival."
            )

        arrival_ids = list(wrapper_args.arrival_sweep_ids)
        arrival_processes = [str(proc).strip().lower() for proc in wrapper_args.arrival_sweep_processes]
        rate_ratios = list(wrapper_args.arrival_sweep_rate_ratios)

        if len(arrival_ids) != len(arrival_processes):
            raise ValueError(
                "--arrival-sweep-ids and --arrival-sweep-processes must have the same length."
            )
        if len(set(arrival_ids)) != len(arrival_ids):
            raise ValueError("--arrival-sweep-ids values must be unique.")
        if rate_ratios and len(rate_ratios) != len(arrival_ids):
            raise ValueError(
                "--arrival-sweep-rate-ratios must be omitted or match --arrival-sweep-ids length."
            )

        valid_arrivals = {str(name).strip().lower() for name in _exp().ARRIVAL_PROCESSES}
        mmpp2_high_fraction = (
            float(wrapper_args.arrival_sweep_mmpp2_high_fraction)
            if wrapper_args.arrival_sweep_mmpp2_high_fraction is not None
            else float(base_args.mmpp2_high_fraction)
        )
        mmpp2_correlation_time_s = (
            float(wrapper_args.arrival_sweep_mmpp2_correlation_time_s)
            if wrapper_args.arrival_sweep_mmpp2_correlation_time_s is not None
            else float(base_args.mmpp2_correlation_time_s)
        )

        arrival_configs: list[dict[str, Any]] = []
        for idx, (arrival_id, arrival_process) in enumerate(zip(arrival_ids, arrival_processes)):
            if arrival_process not in valid_arrivals:
                raise ValueError(
                    f"Unsupported arrival process '{arrival_process}' for arrival id '{arrival_id}'."
                )
            ratio_value = (
                float(rate_ratios[idx])
                if idx < len(rate_ratios)
                else None
            )
            if arrival_process == "mmpp2" and ratio_value is None:
                raise ValueError(
                    "MMPP-2 arrival entries in --arrival-sweep-processes require "
                    "--arrival-sweep-rate-ratios."
                )
            arrival_configs.append(
                {
                    "arrival_id": str(arrival_id),
                    "arrival_process": arrival_process,
                    "mmpp2_rate_ratio": ratio_value,
                }
            )

        for qps in wrapper_args.qps_values:
            qps_value = float(qps)
            for arrival_cfg in arrival_configs:
                mmpp2_rate_ratio = arrival_cfg["mmpp2_rate_ratio"]
                is_mmpp2 = str(arrival_cfg["arrival_process"]) == "mmpp2"
                points.append(
                    {
                        "kind": "qps_arrival",
                        "value": qps_value,
                        "qps_value": qps_value,
                        "output_label": str(arrival_cfg["arrival_id"]),
                        "arrival_id": str(arrival_cfg["arrival_id"]),
                        "arrival_process": str(arrival_cfg["arrival_process"]),
                        "mmpp2_rate_ratio": (
                            float(mmpp2_rate_ratio)
                            if is_mmpp2 and mmpp2_rate_ratio is not None
                            else None
                        ),
                        "mmpp2_high_fraction": (
                            mmpp2_high_fraction if is_mmpp2 else None
                        ),
                        "mmpp2_correlation_time_s": (
                            mmpp2_correlation_time_s if is_mmpp2 else None
                        ),
                        "utilities": list(wrapper_args.qps_utilities),
                        "lambda_weight": float(base_args.lambda_weight),
                        "delta_weight": float(base_args.delta_weight),
                        "request_rate_qps": qps_value,
                        "run_mode": "qps_arrival_sweep",
                    }
                )

    if wrapper_args.sweep == "lambda":
        if not wrapper_args.lambda_values:
            raise ValueError("--lambda-values is required when --sweep is lambda.")
        for lambda_weight in wrapper_args.lambda_values:
            points.append(
                {
                    "kind": "lambda",
                    "value": float(lambda_weight),
                    "utilities": list(wrapper_args.lambda_utilities),
                    "lambda_weight": float(lambda_weight),
                    "delta_weight": float(base_args.delta_weight),
                    "request_rate_qps": float(base_args.request_rate_qps),
                    "run_mode": "lambda_sweep",
                }
            )

    return points


def _as_run_args(
    *,
    base_args: argparse.Namespace,
    utilities: list[str],
    lambda_weight: float,
    delta_weight: float,
    request_rate_qps: float,
) -> argparse.Namespace:
    run_args = argparse.Namespace(**vars(base_args))
    run_args.experiment = "router"
    run_args.utilities = list(utilities)
    run_args.lambda_weight = float(lambda_weight)
    run_args.delta_weight = float(delta_weight)
    run_args.request_rate_qps = float(request_rate_qps)
    return run_args


async def _async_main(argv: Sequence[str]) -> None:
    wrapper_args, forwarded_argv = _parse_wrapper_args(argv)
    exp_mod = _exp()
    base_args = _parse_experiment_args(forwarded_argv)
    if str(base_args.experiment).strip().lower() not in {"router"}:
        raise ValueError(
            "Sweep wrapper only supports router mode. "
            "Pass --experiment router (or omit it)."
        )

    points = _build_run_points(wrapper_args, base_args)
    if not points:
        raise ValueError("No sweep points were produced. Check --sweep arguments.")

    wrapper_args.output_dir.mkdir(parents=True, exist_ok=True)
    session_stamp = time.strftime("%Y%m%d_%H%M%S")

    requests, request_manifest, prompt_source_info = exp_mod._build_request_set(base_args)
    request_set = _request_set_payload(requests, request_manifest)
    instances, instance_costs, instance_metadata = exp_mod.load_instances(
        base_args.instances_config
    )

    outputs: list[dict[str, Any]] = []

    try:
        for point_index, point in enumerate(points, start=1):
            point_value = float(point["value"])
            point_kind = str(point["kind"])
            run_args = _as_run_args(
                base_args=base_args,
                utilities=list(point["utilities"]),
                lambda_weight=float(point["lambda_weight"]),
                delta_weight=float(point["delta_weight"]),
                request_rate_qps=float(point["request_rate_qps"]),
            )
            if point.get("arrival_process") is not None:
                run_args.arrival_process = str(point["arrival_process"])
            if point.get("mmpp2_rate_ratio") is not None:
                run_args.mmpp2_rate_ratio = float(point["mmpp2_rate_ratio"])
            if point.get("mmpp2_high_fraction") is not None:
                run_args.mmpp2_high_fraction = float(point["mmpp2_high_fraction"])
            if point.get("mmpp2_correlation_time_s") is not None:
                run_args.mmpp2_correlation_time_s = float(point["mmpp2_correlation_time_s"])

            run_arrival_process = str(run_args.arrival_process).strip().lower()
            run_mmpp2_rate_ratio = (
                float(run_args.mmpp2_rate_ratio)
                if run_arrival_process == "mmpp2"
                else None
            )
            run_mmpp2_high_fraction = (
                float(run_args.mmpp2_high_fraction)
                if run_arrival_process == "mmpp2"
                else None
            )
            run_mmpp2_correlation_time_s = (
                float(run_args.mmpp2_correlation_time_s)
                if run_arrival_process == "mmpp2"
                else None
            )
            point_meta = {
                "index": point_index,
                "total_points": len(points),
                "kind": point_kind,
                "value": point_value,
                "run_mode": str(point["run_mode"]),
            }
            if point_kind == "qps_arrival":
                point_meta.update(
                    {
                        "qps": float(point.get("qps_value", run_args.request_rate_qps)),
                        "arrival_id": str(point.get("arrival_id", "")),
                        "arrival_process": str(point.get("arrival_process", run_args.arrival_process)),
                        "mmpp2_rate_ratio": (
                            float(point["mmpp2_rate_ratio"])
                            if point.get("mmpp2_rate_ratio") is not None
                            else None
                        ),
                        "mmpp2_high_fraction": (
                            float(point["mmpp2_high_fraction"])
                            if point.get("mmpp2_high_fraction") is not None
                            else None
                        ),
                        "mmpp2_correlation_time_s": (
                            float(point["mmpp2_correlation_time_s"])
                            if point.get("mmpp2_correlation_time_s") is not None
                            else None
                        ),
                    }
                )

            run_output_path = _point_output_path(
                output_dir=wrapper_args.output_dir,
                output_prefix=wrapper_args.output_prefix,
                session_stamp=session_stamp,
                point_kind=point_kind,
                point_value=point_value,
                point_index=point_index,
                point_label=point.get("output_label"),
            )
            response_map_base_path = run_output_path.with_name(
                f"{run_output_path.stem}_response_map.log"
            )
            request_log_base_path = run_output_path.with_name(
                f"{run_output_path.stem}_predicted_waits.log"
            )

            router_result = await exp_mod.run_router_experiment(
                args=run_args,
                requests=requests,
                instances=instances,
                instance_costs=instance_costs,
                instance_metadata=instance_metadata,
                response_map_base_path=response_map_base_path,
                request_log_base_path=request_log_base_path,
            )

            payload = {
                "config": {
                    "experiment": "router",
                    "num_requests": run_args.num_requests,
                    "seed": run_args.seed,
                    "request_rate_qps": run_args.request_rate_qps,
                    "arrival_process": run_args.arrival_process,
                    "mmpp2_rate_ratio": run_mmpp2_rate_ratio,
                    "mmpp2_high_fraction": run_mmpp2_high_fraction,
                    "mmpp2_correlation_time_s": run_mmpp2_correlation_time_s,
                    "utilities": run_args.utilities,
                    "lambda_weight": run_args.lambda_weight,
                    "delta_weight": run_args.delta_weight,
                    "worker_count": run_args.worker_count,
                    "max_queue_size": run_args.max_queue_size,
                    "max_completion_tokens": run_args.max_completion_tokens,
                    "temperature": run_args.temperature,
                    "top_p": run_args.top_p,
                    "feasible_slo_mode": run_args.feasible_slo_mode,
                    "accuracy_model_path": run_args.accuracy_model_path,
                    "output_length_model_path": run_args.output_length_model_path,
                    "enable_wait_time_polling": bool(run_args.enable_wait_time_polling),
                    "critical_wait_time_timeout_s": float(
                        run_args.critical_wait_time_timeout_s
                    ),
                    "prompt_source": prompt_source_info,
                    "instance_metadata": instance_metadata,
                    "instance_costs": instance_costs,
                    "sweep_point": point_meta,
                },
                "request_set": request_set,
                "router": router_result,
            }
            run_output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

            outputs.append(
                {
                    "index": point_index,
                    "kind": point_kind,
                    "value": point_value,
                    "run_mode": str(point["run_mode"]),
                    "lambda_weight": float(run_args.lambda_weight),
                    "request_rate_qps": float(run_args.request_rate_qps),
                    "arrival_process": run_arrival_process,
                    "arrival_id": (
                        str(point["arrival_id"]) if point.get("arrival_id") is not None else None
                    ),
                    "mmpp2_rate_ratio": run_mmpp2_rate_ratio,
                    "mmpp2_high_fraction": run_mmpp2_high_fraction,
                    "mmpp2_correlation_time_s": run_mmpp2_correlation_time_s,
                    "delta_weight": float(run_args.delta_weight),
                    "utilities": list(run_args.utilities),
                    "output_path": str(run_output_path),
                    "response_map_base_path": str(response_map_base_path),
                    "request_log_base_path": str(request_log_base_path),
                }
            )
    finally:
        if not wrapper_args.keep_instances_open:
            close_tasks = [asyncio.to_thread(instance.close) for instance in instances.values()]
            if close_tasks:
                await asyncio.gather(*close_tasks, return_exceptions=True)

    manifest_path = (
        wrapper_args.output_dir
        / f"{wrapper_args.output_prefix}_{session_stamp}_manifest.json"
    )
    manifest = {
        "session_stamp": session_stamp,
        "sweep": wrapper_args.sweep,
        "points": outputs,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(
        json.dumps(
            {
                "manifest_path": str(manifest_path),
                "num_points": len(outputs),
            },
            indent=2,
        )
    )


def main() -> None:
    asyncio.run(_async_main(sys.argv[1:]))


if __name__ == "__main__":
    main()
