import argparse
import asyncio
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sfs_core.routing.wait_time_scheduler import InstanceClient, WaitTimeScheduler
from sfs_core.paths import (
    BUCKETED_OUTPUTS_ROOT,
    ensure_experiments_root,
)
from sfs_core.regression.two_part_fit import (
    FEATURE_SET_CHOICES,
    build_feature_matrix,
    fit_two_part,
)

from sfs_core.shared.shared_experiment_helpers import (
    build_messages,
    DEFAULT_SYSTEM_PROMPT,
    DEFAULT_RANDOM_SEED,
    get_prompt_bucket_files,
    _metric_summary,
    iter_mixed_then_random_bucketed_prompts,
    resolve_max_completion_tokens,
    resolve_prompt_bucket_dir,
    parse_chat_template_kwargs_json,
    resolve_chat_template_kwargs,
    select_prompt_subset,
    warm_up_instances,
)
from sfs_core.shared.trace_theta import (
    estimate_prefill_theta_from_trace,
    estimate_score_proxy_metrics_from_batch_stats,
    snapshot_trace_file_offsets,
)

USER = os.environ.get("USER")
JOB_ID = os.environ.get("SLURM_JOB_ID", "no-job-id")
if not USER or not JOB_ID:
    raise EnvironmentError("Expected USER and SLURM_JOB_ID environment variables to be set.")

_DEFAULT_EXPERIMENT_DIR = ensure_experiments_root()
EXPERIMENT_DIR = Path(
    os.environ.get("MODEL_METRICS_TRACE_DIR", str(_DEFAULT_EXPERIMENT_DIR))
).expanduser()
EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)
NUM_REQUESTS = 2500 * 4 # 2500 per dataset, 4 datasets (alpaca, govreport, writingprompts, hotpot_qa:distractor)
BATCH_STATS_FLUSH_GRACE_S = 1.1

MODEL_DIR_QWEN3_0_6B = f"/local/{USER}/{JOB_ID}/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca"
MODEL_DIR_QWEN3_8B = f"/local/{USER}/{JOB_ID}/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218"
MODEL_DIR_QWEN3_32B = f"/local/{USER}/{JOB_ID}/models--Qwen--Qwen3-32B/snapshots/9216db5781bf21249d130ec9da846c4624c16137"

def get_instance_clients(
    *,
    model_dir_0_6b: str = MODEL_DIR_QWEN3_0_6B,
    model_dir_8b: str = MODEL_DIR_QWEN3_8B,
    model_dir_32b: str = MODEL_DIR_QWEN3_32B,
    port_0_6b: int = 8002,
    port_8b: int = 8000,
    port_32b: int = 8001,
):
    instance_0_6b = InstanceClient(
        instance_id="vllm-0.6b",
        address=f"http://localhost:{int(port_0_6b)}",
        default_model=model_dir_0_6b,
        model_id="qwen3-0.6b",
    )

    instance_8b = InstanceClient(
        instance_id="vllm-8b",
        address=f"http://localhost:{int(port_8b)}",
        default_model=model_dir_8b,
        model_id="qwen3-8b",
    )

    instance_32b = InstanceClient(
        instance_id="vllm-32b",
        address=f"http://localhost:{int(port_32b)}",
        default_model=model_dir_32b,
        model_id="qwen3-32b",
    )

    return instance_0_6b, instance_8b, instance_32b

INSTANCE_COSTS={
    "vllm-0.6b": {"prompt": 0.05, "output": 0.15},
    "vllm-8b": {"prompt": 0.115, "output": 0.4935},
    "vllm-32b": {"prompt": 0.25, "output": 0.725},
}

PROMPT_BUCKET_DIR = Path(
    BUCKETED_OUTPUTS_ROOT / "qwen3-0.6b" / "scored"
    # BUCKETED_OUTPUTS_ROOT / "qwen3-8b" / "outputs"
    # BUCKETED_OUTPUTS_ROOT / "qwen3-32b" / "outputs"
)


def _estimate_utility_components(
    scheduler_for_model: WaitTimeScheduler,
    client: InstanceClient,
    prompt_text: str,
    prompt_tokens: int = 0,
) -> tuple[int, float, float, float]:
    accuracy_scores = scheduler_for_model._predict_accuracy(prompt_text, prompt_tokens)
    output_length_predictions = scheduler_for_model._predict_output_length(
        prompt_text, prompt_tokens
    )
    instance_id = client.instance_id
    accuracy = float(accuracy_scores.get(instance_id, 0.0))
    predicted_output_tokens = float(output_length_predictions.get(instance_id, 0.0))

    cost_info = scheduler_for_model._instance_costs.get(instance_id) or {}
    prompt_rate = float(cost_info.get("prompt", 0.0))
    output_rate = float(cost_info.get("output", 0.0))
    cost = prompt_rate * prompt_tokens + output_rate * predicted_output_tokens
    
    return accuracy, predicted_output_tokens, float(cost)


def _sanitize_for_filename(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in value)


def _single_model_paths(model_name: str) -> tuple[Path, Path, Path, Path]:
    safe_model_name = _sanitize_for_filename(model_name)
    return (
        EXPERIMENT_DIR / f"predicted_wait_times_{safe_model_name}.log",
        EXPERIMENT_DIR / f"actual_wait_times_{safe_model_name}.log",
        EXPERIMENT_DIR / f"request_response_mapping_{safe_model_name}.log",
        EXPERIMENT_DIR / f"batch_stats_{safe_model_name}.csv",
    )


def _build_calibration_payload(
    *,
    prompt: str,
    max_completion_tokens: int,
    model_id: str | None,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    chat_template_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a calibration request with the same prompt policy as sweep traffic."""
    return {
        "messages": build_messages(prompt, system_prompt=system_prompt),
        "temperature": 0.0,
        "top_p": 1.0,
        "max_completion_tokens": max_completion_tokens,
        "model": model_id,
        "extra_body": {
            "chat_template_kwargs": resolve_chat_template_kwargs(
                chat_template_kwargs
            ),
        },
    }


def build_single_model_wait_time_scheduler(
    model_name: str,
    model_clients: dict[str, InstanceClient],
    *,
    worker_count: int = 1,
    max_queue_size: int = 0,
    enable_wait_time_polling: bool = True,
) -> WaitTimeScheduler:
    """
    Create a scheduler that can route only to one selected model instance.

    This isolates service-rate measurement from any other active InstanceClient.
    """
    if model_name not in model_clients:
        available = ", ".join(sorted(model_clients.keys()))
        raise KeyError(f"Unknown model_name={model_name!r}. Available: {available}")

    client = model_clients[model_name]
    scheduler_instances = {client.instance_id: client}
    predicted_wait_log_path, _, response_map_path, _ = _single_model_paths(model_name)
    return WaitTimeScheduler(
        scheduler_instances,
        request_log_path=str(predicted_wait_log_path),
        response_map_path=str(response_map_path),
        worker_count=worker_count,
        max_queue_size=max_queue_size,
        # accuracy_model_path=str(BUCKETED_OUTPUTS_ROOT / "accuracy_predictor"),
        # output_length_model_path=str(BUCKETED_OUTPUTS_ROOT / "output_length_predictor"),
        lambda_weight=0.3,
        delta_weight=0.5,
        instance_costs=INSTANCE_COSTS.get(model_name),
        enable_wait_time_polling=enable_wait_time_polling,
    )


async def _compute_single_model_metrics(
    client: InstanceClient,
    prompt_records: list[dict[str, str]],
    *,
    request_rate_qps: float | None = None,
    model_name: str | None = None,
    model_id: str | None = None,
    max_completion_tokens: int = 8192,
    context_length: int | None = None,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    chat_template_kwargs: dict[str, Any] | None = None,
    scheduler: WaitTimeScheduler,
    request_id_prefix: str,
    batch_fit_feature_set: str = "legacy",
    batch_fit_nonnegative: bool = False,
) -> dict[str, Any]:
    """
    Submit a subset of requests to a single running model and return serving rate and prefill throughput.

    Assumes all traffic is routed to ``client``.
    """    
    _, actual_wait_log_path, response_map_path, batch_stats_csv_path = _single_model_paths(model_name)
    offsets = snapshot_trace_file_offsets(
        response_map_path=response_map_path,
        actual_wait_log_path=actual_wait_log_path,
        batch_stats_csv_path=batch_stats_csv_path,
    )

    dispatch_tasks: list[asyncio.Task] = []
    request_prompt_tokens: dict[str, int] = {}
    start = time.perf_counter()
    for idx, prompt_record in enumerate(prompt_records):
        if idx > 0 and request_rate_qps and request_rate_qps > 0:
            await asyncio.sleep(1.0 / request_rate_qps)

        request_id = f"{request_id_prefix}-{idx}"
        request_prompt_tokens[request_id] = int(prompt_record["prompt_tokens"])
        prompt = prompt_record["prompt"]
        request_max_completion_tokens = resolve_max_completion_tokens(
            max_completion_tokens,
            request_prompt_tokens[request_id],
            context_length=context_length,
            record_max_completion_tokens=prompt_record.get("max_completion_tokens"),
        )

        payload = _build_calibration_payload(
            prompt=prompt,
            max_completion_tokens=request_max_completion_tokens,
            model_id=model_id,
            system_prompt=system_prompt,
            chat_template_kwargs=chat_template_kwargs,
        )

        dispatch_tasks.append(
            asyncio.create_task(
                scheduler.route_and_submit(
                    request_id=request_id,
                    await_dispatch=True,
                    **payload,
                )
            )
        )

    results = await asyncio.gather(*dispatch_tasks, return_exceptions=True)
    await scheduler.drain()

    elapsed_s = max(time.perf_counter() - start, 1e-9)
    failed = sum(1 for result in results if isinstance(result, Exception))
    succeeded = len(results) - failed
    # BatchStatsLogger flushes once per second. Keep the service-rate timing
    # above independent of this telemetry grace period.
    await asyncio.sleep(BATCH_STATS_FLUSH_GRACE_S)

    resolved_model = (
        model_id
        or getattr(client, "model_id", None)
        or getattr(client, "default_model", None)
        or client.instance_id
    )
    
    theta = estimate_prefill_theta_from_trace(
        request_prompt_tokens=request_prompt_tokens,
        response_map_path=response_map_path,
        actual_wait_log_path=actual_wait_log_path,
        response_map_offset=offsets["response_map_offset"],
        actual_wait_log_offset=offsets["actual_wait_log_offset"],
        batch_stats_csv_path=batch_stats_csv_path,
        batch_stats_offset=offsets.get("batch_stats_offset", 0),
    )
    score_proxy_metrics = estimate_score_proxy_metrics_from_batch_stats(
        batch_stats_csv_path=batch_stats_csv_path,
        batch_stats_offset=offsets.get("batch_stats_offset", 0),
    )
    score_proxy_metrics["prefill_tps"] = float(
        theta["theta_p_tps_from_batch_stats"]
    )
    batch_fit, batch_df = fit_two_part(
        batch_stats_csv_path,
        stall_percentile=99.9,
        feature_set=batch_fit_feature_set,
        start_offset=offsets.get("batch_stats_offset", 0),
        nonnegative_coefficients=batch_fit_nonnegative,
    )
    batch_features, _ = build_feature_matrix(
        batch_df,
        feature_set=batch_fit_feature_set,
    )
    fitted_batch_times = batch_fit.predict_typical(batch_features)
    observed_batch_times = batch_df["exec"].to_numpy()
    total_variation = float(
        np.sum((observed_batch_times - observed_batch_times.mean()) ** 2)
    )
    residual_variation = float(
        np.sum((observed_batch_times - fitted_batch_times) ** 2)
    )
    r2_all_rows = (
        1.0 - residual_variation / total_variation
        if total_variation > 0
        else 1.0
    )
    nonempty_batches = (
        batch_df["prefill"].to_numpy() + batch_df["decode"].to_numpy()
    ) > 0
    if not nonempty_batches.any():
        raise RuntimeError("Batch-latency fit contains no nonempty batches")
    coefficient_by_name = {
        str(name): float(coefficient)
        for name, coefficient in zip(
            batch_fit.feature_names,
            batch_fit.base_model.coef_,
        )
    }
    final_feature_name = (
        "s_sq" if batch_fit_feature_set == "legacy" else "p_x_ctx"
    )
    sfs_simulation = {
        "feature_set": batch_fit_feature_set,
        "intercept": float(batch_fit.base_model.intercept_),
        "prefill_coeff": coefficient_by_name["p"],
        "decode_coeff": coefficient_by_name["d"],
        "sum_coeff": coefficient_by_name["s"],
        "prefill_sq_coeff": coefficient_by_name["p_sq_sum"],
        "sum_sq_coeff": coefficient_by_name[final_feature_name],
        "coefficient_constraint": batch_fit.coefficient_constraint,
        "fit_rows": int(len(batch_df)),
        "fit_inlier_rows": int(batch_fit.inlier_mask.sum()),
        "fit_prediction_diagnostics": {
            "minimum_s": float(fitted_batch_times.min()),
            "negative_rows": int((fitted_batch_times < 0).sum()),
            "minimum_nonempty_s": float(
                fitted_batch_times[nonempty_batches].min()
            ),
            "negative_nonempty_rows": int(
                (fitted_batch_times[nonempty_batches] < 0).sum()
            ),
            "r2_all_rows": float(r2_all_rows),
            "mae_s_all_rows": float(
                np.mean(np.abs(observed_batch_times - fitted_batch_times))
            ),
        },
        "stall_probability": float(batch_fit.stall_probability),
        "mean_stall_delay_s": float(batch_fit.mean_stall_delay),
    }

    result: dict[str, Any] = {
        "model_id": str(resolved_model),
        "num_queries": len(prompt_records),
        "succeeded": succeeded,
        "failed": failed,
        "elapsed_s": elapsed_s,
        "service_rate_qps": succeeded / elapsed_s,
        "prefill_theta": theta,
        "score_proxy": score_proxy_metrics,
        "sfs_simulation": sfs_simulation,
        "actual_wait_log_path": str(actual_wait_log_path),
        "response_map_path": str(response_map_path),
        "batch_stats_csv_path": str(batch_stats_csv_path) if batch_stats_csv_path else None,
    }
    
    return result


async def compute_metrics_for_models(
    model_clients: dict[str, InstanceClient],
    prompt_records: list[dict[str, str]],
    *,
    request_rate_qps: float | None = None,
    max_completion_tokens: int = 8192,
    context_length: int | None = None,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    chat_template_kwargs: dict[str, Any] | None = None,
    output_path: Path,
    batch_fit_feature_set: str = "legacy",
    batch_fit_nonnegative: bool = False,
) -> dict[str, dict[str, Any]]:
    """
    Convenience wrapper to compute service rates and prefill throughputs for different models.

    Start all model servers before calling this function. Each model is routed
    to independently, and all per-model calibration streams run concurrently.
    """
    schedulers = {
        model_name: build_single_model_wait_time_scheduler(
            model_name, model_clients, enable_wait_time_polling=False,
        )
        for model_name in model_clients
    }
    results: dict[str, dict[str, Any]] = {}
    try:
        # Warm every server together, then give telemetry a common flush
        # interval before each measured task snapshots its trace offsets.
        await warm_up_instances(list(model_clients.values()))
        await asyncio.sleep(BATCH_STATS_FLUSH_GRACE_S)

        model_names = list(model_clients)
        model_results = await asyncio.gather(
            *(
                _compute_single_model_metrics(
                    model_clients[model_name],
                    prompt_records,
                    request_rate_qps=request_rate_qps,
                    model_name=model_name,
                    model_id=getattr(
                        model_clients[model_name], "default_model", None
                    ),
                    max_completion_tokens=max_completion_tokens,
                    context_length=context_length,
                    system_prompt=system_prompt,
                    chat_template_kwargs=chat_template_kwargs,
                    scheduler=schedulers[model_name],
                    request_id_prefix=(
                        f"model-metrics-{_sanitize_for_filename(model_name)}"
                    ),
                    batch_fit_feature_set=batch_fit_feature_set,
                    batch_fit_nonnegative=batch_fit_nonnegative,
                )
                for model_name in model_names
            )
        )
        results.update(zip(model_names, model_results))
    finally:
        await asyncio.gather(
            *(scheduler.stop() for scheduler in schedulers.values()),
            return_exceptions=True,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
            
    return results


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Calibrate per-model prefill/decode throughput and mean "
            "decode-batch latency for routing baselines."
        )
    )
    parser.add_argument(
        "--prompt-bucket-dir",
        type=Path,
        default=PROMPT_BUCKET_DIR,
        help=(
            "Directory containing the prompt-bucket JSONL files used for "
            "service-rate calibration."
        ),
    )
    parser.add_argument("--num-requests", type=int, default=NUM_REQUESTS)
    parser.add_argument("--request-rate-qps", type=float, default=None)
    parser.add_argument("--max-completion-tokens", type=int, default=8192)
    parser.add_argument("--context-length", type=int, default=None)
    parser.add_argument("--system-prompt", type=str, default=DEFAULT_SYSTEM_PROMPT)
    parser.add_argument(
        "--chat-template-kwargs-json",
        dest="chat_template_kwargs",
        type=parse_chat_template_kwargs_json,
        default=None,
        help=(
            "JSON object passed to the model chat template. Defaults to Qwen's "
            "non-thinking mode; use '{}' for model families without "
            "enable_thinking."
        ),
    )
    parser.add_argument(
        "--batch-fit-feature-set",
        choices=FEATURE_SET_CHOICES,
        default="legacy",
    )
    parser.add_argument(
        "--batch-fit-nonnegative",
        action="store_true",
        help=(
            "Constrain the SFS batch-latency intercept and slopes to be "
            "nonnegative."
        ),
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=EXPERIMENT_DIR / "model_metrics.json",
    )
    parser.add_argument("--model-dir-0-6b", default=MODEL_DIR_QWEN3_0_6B)
    parser.add_argument("--model-dir-8b", default=MODEL_DIR_QWEN3_8B)
    parser.add_argument("--model-dir-32b", default=MODEL_DIR_QWEN3_32B)
    parser.add_argument("--port-0-6b", type=int, default=8002)
    parser.add_argument("--port-8b", type=int, default=8000)
    parser.add_argument("--port-32b", type=int, default=8001)
    args = parser.parse_args(argv)
    if args.num_requests <= 0:
        parser.error("--num-requests must be > 0")
    if args.request_rate_qps is not None and args.request_rate_qps <= 0:
        parser.error("--request-rate-qps must be > 0 when supplied")
    if args.max_completion_tokens <= 0:
        parser.error("--max-completion-tokens must be > 0")
    if args.context_length is not None and args.context_length <= 0:
        parser.error("--context-length must be > 0 when supplied")
    args.chat_template_kwargs = resolve_chat_template_kwargs(
        args.chat_template_kwargs
    )
    for port_name in ("port_0_6b", "port_8b", "port_32b"):
        if not 1 <= int(getattr(args, port_name)) <= 65535:
            parser.error(f"--{port_name.replace('_', '-')} must be in [1, 65535]")
    return args


### Service Rates and Prefill Throughputs:
def main():
    args = parse_args()
    prompt_bucket_dir = resolve_prompt_bucket_dir(
        args.prompt_bucket_dir,
        label="Calibration prompt-bucket",
    )
    prompt_bucket_files = get_prompt_bucket_files(prompt_bucket_dir)
    per_bucket_limit = (
        args.num_requests + len(prompt_bucket_files) - 1
    ) // len(prompt_bucket_files)
    prompt_records = select_prompt_subset(
        iter_mixed_then_random_bucketed_prompts(
            prompt_bucket_dir,
            per_bucket_limit=per_bucket_limit,
            limit=args.num_requests,
            seed=DEFAULT_RANDOM_SEED,
            include_complete_record=True,
        ),
        args.num_requests,
    )
    
    instance_0_6b, instance_8b, instance_32b = get_instance_clients(
        model_dir_0_6b=args.model_dir_0_6b,
        model_dir_8b=args.model_dir_8b,
        model_dir_32b=args.model_dir_32b,
        port_0_6b=args.port_0_6b,
        port_8b=args.port_8b,
        port_32b=args.port_32b,
    )

    model_clients = {
        "qwen3-0.6b": instance_0_6b,
        "qwen3-8b": instance_8b,
        "qwen3-32b": instance_32b,
    }

    results_metrics = asyncio.run(
        compute_metrics_for_models(
            model_clients,
            prompt_records,
            request_rate_qps=(
                args.request_rate_qps
                if args.request_rate_qps is not None
                else float(args.num_requests)
            ),
            max_completion_tokens=args.max_completion_tokens,
            context_length=args.context_length,
            system_prompt=args.system_prompt,
            chat_template_kwargs=args.chat_template_kwargs,
            output_path=args.output_path,
            batch_fit_feature_set=args.batch_fit_feature_set,
            batch_fit_nonnegative=args.batch_fit_nonnegative,
        )
    )
    print(json.dumps(results_metrics, indent=2))
    
    return results_metrics


if __name__ == "__main__":
    main()
