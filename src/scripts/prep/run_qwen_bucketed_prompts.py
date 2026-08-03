#!/usr/bin/env python3
"""
Dispatch bucketed prompts to one OpenAI-compatible model using the async
WaitTimeScheduler machinery and log prompt/output/timing metadata.

This script reuses the scheduler helpers from ``sfs_core.routing``
instead of doing direct synchronous HTTP calls.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from transformers import AutoTokenizer

SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = SCRIPT_DIR.parent.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from sfs_core.routing.wait_time_scheduler import (
    InstanceClient,
    RoutedRequest,
    WaitTimeResult,
    WaitTimeScheduler,
)
from sfs_core.shared.shared_experiment_helpers import (
    build_messages,
    DEFAULT_SYSTEM_PROMPT,
    parse_chat_template_kwargs_json,
    resolve_chat_template_kwargs,
)
from sfs_core.paths import BUCKETED_OUTPUTS_ROOT, DEFAULT_BUCKET_POOL_QWEN3_0_6B

DEFAULT_BUCKET_DIR = DEFAULT_BUCKET_POOL_QWEN3_0_6B
DEFAULT_OUT_ROOT = BUCKETED_OUTPUTS_ROOT
DEFAULT_ID = "Qwen/Qwen3-8B"

@dataclass
class ModelJob:
    label: str
    bucket_dir: Path
    instance_id: str
    address: str
    default_model: str


def load_tokenizer(model_or_path: str):
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_or_path, use_fast=True)
        return tokenizer
    except Exception as exc:
        print(
            f"[WARN] Failed to load tokenizer '{model_or_path}': {exc}. "
            "Falling back to whitespace token counting.",
            file=sys.stderr,
        )
        return None


def count_text_tokens(text: str, tokenizer) -> int:
    if not text:
        return 0
    if tokenizer is None:
        return len(text.split())
    try:
        return len(tokenizer.encode(text, add_special_tokens=False))
    except Exception as exc:
        print(
            f"[WARN] Tokenization failed for text (len={len(text)}): {exc}. "
            "Falling back to whitespace approximation.",
            file=sys.stderr,
        )
        return len(text.split())


def truncate_prompt_tokens(
    prompt: str, tokenizer, max_tokens: Optional[int]
) -> tuple[str, int]:
    if not prompt:
        return "", 0
    if max_tokens is None or max_tokens <= 0:
        return prompt, count_text_tokens(prompt, tokenizer)

    token_ids = tokenizer.encode(prompt, add_special_tokens=False)

    if len(token_ids) <= max_tokens:
        return prompt, len(token_ids)

    truncated_ids = token_ids[:max_tokens]
    truncated_text = tokenizer.decode(truncated_ids)
    return truncated_text, len(truncated_ids)


def iso_timestamp(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()


def iter_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def completion_to_dict(response: Any) -> Any:
    """Best-effort conversion of an OpenAI response to JSON-serializable data."""
    if response is None:
        return None
    if hasattr(response, "model_dump_json"):
        try:
            return json.loads(response.model_dump_json())
        except Exception:
            pass
    if hasattr(response, "model_dump"):
        try:
            return response.model_dump()
        except Exception:
            pass
    if isinstance(response, dict):
        return response
    if hasattr(response, "dict"):
        try:
            return response.dict()
        except Exception:
            pass
    try:
        return json.loads(str(response))
    except Exception:
        return str(response)


def extract_output_text(response: Any) -> tuple[str, Optional[str]]:
    """Pull the first completion text + finish reason from an OpenAI response."""
    choices = None
    if response is None:
        return "", None
    if isinstance(response, dict):
        choices = response.get("choices")
    elif hasattr(response, "choices"):
        choices = getattr(response, "choices")
    if not choices:
        return "", None
    choice = choices[0]
    finish_reason = None
    message = None
    text_field = None

    if isinstance(choice, dict):
        finish_reason = choice.get("finish_reason")
        message = choice.get("message")
        text_field = choice.get("text")
    else:
        finish_reason = getattr(choice, "finish_reason", None)
        message = getattr(choice, "message", None)
        text_field = getattr(choice, "text", None)

    if message is not None:
        if isinstance(message, dict):
            content = message.get("content")
        else:
            content = getattr(message, "content", None)
        if content:
            return content, finish_reason
    if text_field:
        return text_field, finish_reason
    return "", finish_reason


class BucketingWaitTimeScheduler(WaitTimeScheduler):
    """
    Small wrapper around WaitTimeScheduler that captures the response of each
    submitted request and resolves a user-supplied future with metadata.
    """

    async def _dispatch(self, queued) -> None:  # type: ignore[override]
        result_future = queued.payload.pop("_result_future", None)
        context = queued.payload.pop("_context", None)
        target_id: Optional[str] = None
        wait_record = None

        try:
            prompt_text = self._extract_prompt_text(queued.payload)
            prompt_tokens = self._extract_precomputed_prompt_tokens(queued.payload)
            if prompt_tokens is None:
                prompt_tokens = self._get_prompt_tokens(queued.payload, prompt_text)
            completion_cap = self._extract_completion_cap(queued.payload)
            wait_results = await self._collect_wait_times(
                prompt_tokens=prompt_tokens if prompt_tokens and prompt_tokens > 0 else None,
            )
            accuracy_scores, output_lengths = await self._predict_model_scores(
                prompt_text=prompt_text,
                prompt_tokens=prompt_tokens or 0,
                completion_cap=completion_cap,
            )
            target_id = self._select_instance(
                wait_results,
                accuracy_scores,
                output_lengths,
                prompt_tokens or 0,
            )
            target = self._instances[target_id]
            wait_record = wait_results.get(target_id) or target.last_wait_for_mode(
                prompt_tokens=prompt_tokens if prompt_tokens and prompt_tokens > 0 else None,
            )

            if wait_record:
                self._request_log[queued.request_id] = wait_record
                await self._log_to_file(queued.request_id, wait_record)

            submit_task = asyncio.create_task(target.submit_request(**queued.payload))
            self._submit_tasks.add(submit_task)
            submit_task.add_done_callback(self._submit_tasks.discard)
            if self._response_map_path:
                submit_task.add_done_callback(
                    lambda task: self._handle_response_mapping(
                        task,
                        request_id=queued.request_id,
                        instance_id=target_id or "",
                    )
                )

            if result_future is not None:

                def _handle_done(task: asyncio.Task) -> None:
                    payload = {
                        "context": context,
                        "wait_record": wait_record,
                        "completed_perf": time.perf_counter(),
                        "completed_wall": time.time(),
                    }
                    if task.cancelled():
                        payload["error"] = {
                            "type": "CancelledError",
                            "message": "Request cancelled before completion.",
                        }
                    else:
                        exc = task.exception()
                        if exc:
                            payload["error"] = {
                                "type": exc.__class__.__name__,
                                "message": str(exc),
                            }
                        else:
                            payload["response"] = task.result()
                    if not result_future.done():
                        result_future.set_result(payload)

                submit_task.add_done_callback(_handle_done)

            routed = RoutedRequest(
                request_id=queued.request_id,
                instance_id=target_id,
                wait_time_ms=wait_record.wait_ms if wait_record else None,
                wait_time_details=wait_record.raw_payload if wait_record else None,
            )
            if not queued.result_future.done():
                queued.result_future.set_result(routed)
        except Exception as exc:
            if result_future is not None and not result_future.done():
                result_future.set_result(
                    {
                        "context": context,
                        "wait_record": wait_record,
                        "completed_perf": time.perf_counter(),
                        "completed_wall": time.time(),
                        "error": {
                            "type": exc.__class__.__name__,
                            "message": str(exc),
                        },
                    }
                )
            if not queued.result_future.done():
                queued.result_future.set_result(
                    RoutedRequest(
                        request_id=queued.request_id,
                        instance_id=target_id or "dispatch-error",
                        wait_time_ms=wait_record.wait_ms if wait_record else None,
                        wait_time_details=wait_record.raw_payload
                        if wait_record
                        else {"dispatch_error": str(exc)},
                    )
                )


async def process_bucket(
    job: ModelJob,
    bucket_file: Path,
    scheduler: BucketingWaitTimeScheduler,
    args: argparse.Namespace,
    output_dir: Path,
    tokenizer,
    context_length: int,
    prompt_token_limit: int,
) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{bucket_file.stem}.jsonl"
    stats = {"total": 0, "success": 0, "failed": 0, "output_file": str(output_path)}
    loop = asyncio.get_running_loop()
    pending: List[asyncio.Future] = []

    with output_path.open("w", encoding="utf-8") as writer:
        for idx, record in enumerate(iter_jsonl(bucket_file)):
            if args.max_prompts_per_bucket is not None and idx >= args.max_prompts_per_bucket:
                break
            original_prompt = record.get("prompt", "")
            prompt, prompt_only_tokens = truncate_prompt_tokens(
                original_prompt, tokenizer, prompt_token_limit
            )
            metadata = {k: v for k, v in record.items() if k != "prompt"}
            request_id = f"{job.label}-{bucket_file.stem}-{idx}"
            start_perf = time.perf_counter()
            
            messages = build_messages(prompt, args.system_prompt)
            total_prompt_tokens = 0
            if tokenizer is not None:
                try:
                    prompt_token_ids = tokenizer.apply_chat_template(
                        messages,
                        tokenize=True,
                        add_generation_prompt=True,
                        **args.chat_template_kwargs,
                    )
                    total_prompt_tokens = len(prompt_token_ids)
                except Exception as exc:
                    print(
                        f"[WARN] apply_chat_template failed for request {request_id}: {exc}. "
                        "Falling back to approximate token counts.",
                        file=sys.stderr,
                    )
            if total_prompt_tokens <= 0:
                total_prompt_tokens = count_text_tokens(prompt, tokenizer)
                if args.system_prompt:
                    total_prompt_tokens += count_text_tokens(args.system_prompt, tokenizer)
            
            remaining_context = context_length - total_prompt_tokens
            if args.max_completion_tokens is not None:
                remaining_context = min(remaining_context, args.max_completion_tokens)
            max_completion_tokens = max(1, remaining_context)
            context = {
                "bucket": bucket_file.stem,
                "prompt_index": idx,
                "prompt": prompt,
                "prompt_metadata": metadata,
                "job_label": job.label,
                "request_id": request_id,
                "enqueued_wall": time.time(),
                "start_perf": start_perf,
                "prompt_tokens": total_prompt_tokens,
                "prompt_only_tokens": prompt_only_tokens,
                "max_completion_tokens": max_completion_tokens,
            }
            payload = {
                "model": job.default_model,
                "messages": messages,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "max_completion_tokens": max_completion_tokens,
            }
            payload["extra_body"] = {
                "chat_template_kwargs": dict(args.chat_template_kwargs)
            }
            if args.stop:
                payload["stop"] = args.stop
            if args.seed is not None:
                payload["seed"] = args.seed

            fut: asyncio.Future = loop.create_future()
            payload["_context"] = context
            payload["_result_future"] = fut
            stats["total"] += 1
            await scheduler.route_and_submit(request_id=request_id, **payload)
            pending.append(fut)
            if stats["total"] % 64 == 0:
                await asyncio.sleep(0)

        if not pending:
            return stats

        completed = 0
        for fut in asyncio.as_completed(pending):
            result = await fut
            context = result.get("context") or {}
            response_payload = completion_to_dict(result.get("response"))
            output_text, finish_reason = extract_output_text(response_payload)
            end_perf = result.get("completed_perf", time.perf_counter())
            e2e_time = None
            if context.get("start_perf") is not None:
                e2e_time = end_perf - context["start_perf"]
            response_id = None
            completion_tokens = None
            total_tokens = None
            if isinstance(response_payload, dict):
                response_id = response_payload.get("id")
                usage = response_payload.get("usage") or {}
                completion_tokens = usage.get("completion_tokens")
                total_tokens = usage.get("total_tokens")
            entry = {
                "bucket": context.get("bucket"),
                "model_label": job.label,
                "model_id": job.default_model,
                "prompt_index": context.get("prompt_index"),
                "request_id": context.get("request_id"),
                "prompt": context.get("prompt"),
                "prompt_metadata": context.get("prompt_metadata"),
                "prompt_tokens": context.get("prompt_tokens"),
                "prompt_only_tokens": context.get("prompt_only_tokens"),
                "max_completion_tokens": context.get("max_completion_tokens"),
                "response": {
                    "output_text": output_text,
                    "id": response_id,
                    "completion_tokens": completion_tokens,
                    "total_tokens": total_tokens,
                    "finish_reason": finish_reason,
                },
                "timings": {
                    "e2e_time_s": e2e_time,
                    "enqueued_at": iso_timestamp(context.get("enqueued_wall", time.time())),
                    "completed_at": iso_timestamp(result.get("completed_wall", time.time())),
                },
                "error": result.get("error"),
            }
            if entry["error"]:
                stats["failed"] += 1
            else:
                stats["success"] += 1
            writer.write(json.dumps(entry, ensure_ascii=False))
            writer.write("\n")
            writer.flush()
            completed += 1
            # if args.log_interval and completed % args.log_interval == 0:
            #     print(
            #         f"[{job.label}] {bucket_file.stem}: "
            #         f"{completed}/{len(pending)} completions recorded "
            #         f"(success={stats['success']}, failed={stats['failed']})",
            #         flush=True,
            #     )

    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a single bucketed prompt set through a model using WaitTimeScheduler."
    )
    parser.add_argument(
        "--bucket-dir",
        type=Path,
        default=DEFAULT_BUCKET_DIR,
        help="Directory containing bucketed_prompts JSONL splits.",
    )
    parser.add_argument(
        "--job-label",
        type=str,
        default="qwen3",
        help="Label used for output subdirectories.",
    )
    parser.add_argument(
        "--instance-address",
        type=str,
        default="http://localhost:8001",
        help="Base address (without /v1) for the target vLLM server.",
    )
    parser.add_argument(
        "--instance-id",
        type=str,
        default="vllm-0",
        help="Identifier for the instance (used in logs).",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default=DEFAULT_ID,
        help="Model path or alias served by the target instance.",
    )
    parser.add_argument(
        "--tokenizer-id",
        type=str,
        default=DEFAULT_ID,
        help="Tokenizer identifier used to count prompt tokens.",
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--system-prompt", type=str, default=DEFAULT_SYSTEM_PROMPT)
    parser.add_argument(
        "--chat-template-kwargs-json",
        dest="chat_template_kwargs",
        type=parse_chat_template_kwargs_json,
        default=None,
        help=(
            "JSON object passed to the tokenizer chat template and vLLM chat API. "
            "Defaults to Qwen's non-thinking mode; use '{}' for model families "
            "without Qwen's enable_thinking option."
        ),
    )
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument(
        "--context-length",
        type=int,
        default=65536,
        help="Maximum context window for the target model.",
    )
    parser.add_argument(
        "--max-completion-tokens",
        type=int,
        default=16384,
        help="Optional hard cap on completion tokens; defaults to context minus prompt tokens.",
    )
    parser.add_argument(
        "--prompt-token-limit",
        type=int,
        default=16384,
        help="Trim user prompts to this many tokens before submission.",
    )
    parser.add_argument("--stop", type=str, nargs="*", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max-prompts-per-bucket", type=int, default=2500)
    parser.add_argument("--log-interval", type=int, default=25)
    parser.add_argument("--worker-count", type=int, default=8)
    parser.add_argument("--max-queue-size", type=int, default=0)
    parser.add_argument(
        "--disable-wait-time-polling",
        action="store_true",
        help="Skip polling the /wait_time endpoint before dispatch and use cached wait data only.",
    )
    parser.add_argument("--request-log-path", type=Path, default=None)
    args = parser.parse_args()
    args.chat_template_kwargs = resolve_chat_template_kwargs(
        args.chat_template_kwargs
    )
    return args


async def run_job(
    job: ModelJob,
    args: argparse.Namespace,
    run_dir: Path,
    tokenizer,
    context_length: int,
) -> Dict[str, Any]:
    if not job.bucket_dir.exists():
        print(f"[WARN] Bucket directory missing for {job.label}: {job.bucket_dir}", file=sys.stderr)
        return {}

    instance = InstanceClient(
        instance_id=job.instance_id,
        address=job.address,
        default_model=job.default_model,
        model_id=(job.label or job.default_model),
    )
    scheduler = BucketingWaitTimeScheduler(
        {job.instance_id: instance},
        request_log_path=str(args.request_log_path) if args.request_log_path else None,
        worker_count=args.worker_count,
        max_queue_size=args.max_queue_size,
        enable_wait_time_polling=not args.disable_wait_time_polling,
    )

    await scheduler.start()
    job_output_dir = run_dir / job.label
    bucket_stats: Dict[str, Any] = {}
    jsonl_files = sorted(
        p for p in job.bucket_dir.glob("*.jsonl") if p.name != "summary.json"
    )
    if not jsonl_files:
        print(f"[WARN] No bucket JSONL files found in {job.bucket_dir}", file=sys.stderr)
        await scheduler.stop()
        return {}

    print(f"[INFO] Processing {len(jsonl_files)} splits for {job.label} (instance={job.address})")
    try:
        for bucket_file in jsonl_files:
            print(f"[INFO] -> {job.label} / {bucket_file.name}")
            bucket_stats[bucket_file.stem] = await process_bucket(
                job,
                bucket_file,
                scheduler,
                args,
                job_output_dir,
                tokenizer,
                context_length,
                args.prompt_token_limit,
            )
    finally:
        await scheduler.drain()
        await scheduler.stop()
    return bucket_stats


async def async_main(args: argparse.Namespace) -> None:
    output_root = args.output_root.expanduser().resolve()
    timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = output_root / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = load_tokenizer(args.tokenizer_id)

    job = ModelJob(
        label=args.job_label,
        bucket_dir=args.bucket_dir.expanduser().resolve(),
        instance_id=args.instance_id,
        address=args.instance_address.rstrip("/"),
        default_model=args.model_id,
    )

    summary: Dict[str, Any] = {
        "run_dir": str(run_dir),
        "created_at": datetime.now(tz=timezone.utc).isoformat(),
        "args": {
            "model_id": args.model_id,
            "max_completion_tokens": args.max_completion_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_prompts_per_bucket": args.max_prompts_per_bucket,
            "system_prompt": args.system_prompt,
            "chat_template_kwargs": args.chat_template_kwargs,
            "bucket_dir": str(job.bucket_dir),
            "instance_address": job.address,
            "context_length": args.context_length,
            "tokenizer_id": args.tokenizer_id,
            "prompt_token_limit": args.prompt_token_limit,
            "disable_wait_time_polling": args.disable_wait_time_polling,
        },
        "job": {},
    }

    stats = await run_job(job, args, run_dir, tokenizer, args.context_length)
    if stats:
        summary["job"] = {
            "label": job.label,
            "model_id": job.default_model,
            "address": job.address,
            "bucket_dir": str(job.bucket_dir),
            "buckets": stats,
        }

    manifest_path = run_dir / "run_summary.json"
    manifest_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\n[DONE] Results saved under: {run_dir}")


def main() -> None:
    args = parse_args()
    try:
        asyncio.run(async_main(args))
    except KeyboardInterrupt:
        print("Interrupted by user.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
