import asyncio
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sfs_core.routing.wait_time_scheduler import InstanceClient, WaitTimeScheduler
from sfs_core.paths import (
    BUCKETED_OUTPUTS_ROOT,
    ensure_experiments_root,
)

from sfs_core.shared.shared_experiment_helpers import (
    build_messages,
    DEFAULT_SYSTEM_PROMPT,
    DEFAULT_RANDOM_SEED,
    get_prompt_bucket_files,
    _metric_summary,
    iter_bucketed_prompts,
    iter_mixed_bucketed_prompts,
    iter_mixed_then_random_bucketed_prompts,
    iter_random_bucketed_prompts,
    select_prompt_subset,
    warm_up_instances,
)
from sfs_core.shared.trace_theta import estimate_prefill_theta_from_trace, snapshot_trace_file_offsets

USER = os.environ.get("USER")
JOB_ID = os.environ.get("SLURM_JOB_ID", "no-job-id")
if not USER or not JOB_ID:
    raise EnvironmentError("Expected USER and SLURM_JOB_ID environment variables to be set.")

EXPERIMENT_DIR = ensure_experiments_root()
NUM_REQUESTS = 2500 * 4 # 2500 per dataset, 4 datasets (alpaca, govreport, writingprompts, hotpot_qa:distractor)
REQUESTS_PER_SECOND = NUM_REQUESTS

MODEL_DIR_QWEN3_0_6B = f"/local/{USER}/{JOB_ID}/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca"
MODEL_DIR_QWEN3_8B = f"/local/{USER}/{JOB_ID}/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218"
MODEL_DIR_QWEN3_32B = f"/local/{USER}/{JOB_ID}/models--Qwen--Qwen3-32B/snapshots/9216db5781bf21249d130ec9da846c4624c16137"

def get_instance_clients():
    instance_0_6b = InstanceClient(
        instance_id="vllm-0.6b",
        address="http://localhost:8002",
        default_model=MODEL_DIR_QWEN3_0_6B,
        model_id="qwen3-0.6b",
    )

    instance_8b = InstanceClient(
        instance_id="vllm-8b",
        address="http://localhost:8000",
        default_model=MODEL_DIR_QWEN3_8B,
        model_id="qwen3-8b",
    )

    instance_32b = InstanceClient(
        instance_id="vllm-32b",
        address="http://localhost:8001",
        default_model=MODEL_DIR_QWEN3_32B,
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
    scheduler: WaitTimeScheduler,
    request_id_prefix: str,
) -> dict[str, float | int | str | dict[str, int]]:
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
        
        payload = {
            "messages": build_messages(prompt, system_prompt=DEFAULT_SYSTEM_PROMPT),
            "temperature": 0.0,
            "top_p": 1.0,
            "max_completion_tokens": max_completion_tokens,
            "model": model_id,
        }

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

    result: dict[str, float | int | str | dict[str, int]] = {
        "model_id": str(resolved_model),
        "num_queries": len(prompt_records),
        "succeeded": succeeded,
        "failed": failed,
        "elapsed_s": elapsed_s,
        "service_rate_qps": succeeded / elapsed_s,
        "prefill_theta": theta,
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
    output_path: Path,
) -> dict[str, dict[str, float | int | str]]:
    """
    Convenience wrapper to compute service rates and prefill throughputs for different models.

    Start all model servers before calling this function. Each model will be routed to independently.
    """
    results: dict[str, dict[str, float | int | str]] = {}
    for model_name, client in model_clients.items():    
        scheduler_for_model = build_single_model_wait_time_scheduler(
            model_name, model_clients, enable_wait_time_polling=False,
        )
        try:
            await warm_up_instances([client])
            results[model_name] = await _compute_single_model_metrics(
                client,
                prompt_records,
                request_rate_qps=request_rate_qps,
                model_name=model_name,
                model_id=getattr(client, "default_model", None),
                max_completion_tokens=max_completion_tokens,
                scheduler=scheduler_for_model,
                request_id_prefix=f"model-metrics-{_sanitize_for_filename(model_name)}",
            )
        finally:
            await scheduler_for_model.stop()
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
            
    return results


### Service Rates and Prefill Throughputs:
def main():
    prompt_records = select_prompt_subset(
        iter_random_bucketed_prompts(PROMPT_BUCKET_DIR, limit=NUM_REQUESTS, seed=DEFAULT_RANDOM_SEED, include_complete_record=True),
        NUM_REQUESTS,
    )
    
    instance_0_6b, instance_8b, instance_32b = get_instance_clients()

    model_clients = {
        "qwen3-0.6b": instance_0_6b,
        "qwen3-8b": instance_8b,
        "qwen3-32b": instance_32b,
    }

    results_metrics = asyncio.run(
        compute_metrics_for_models(
            model_clients,
            prompt_records,
            request_rate_qps=REQUESTS_PER_SECOND,
            output_path=EXPERIMENT_DIR / "model_metrics.json",
        )
    )
    print(json.dumps(results_metrics, indent=2))
    
    return results_metrics


if __name__ == "__main__":
    main()
