"""Exercise chat admission, fitted prediction, transport, and snapshots on CPU."""
from __future__ import annotations

import argparse
import asyncio
import json
import math
from pathlib import Path
import re
from types import SimpleNamespace
from unittest.mock import patch


def model_paths(root: Path, family: str) -> list[Path]:
    if family == "qwen":
        from scripts.runs.qwen_baselines import HF_NAMES, PINS
        return [root.parent / ".cache/huggingface/hub" / f"models--Qwen--{name}" /
                "snapshots" / pin for name, pin in zip(HF_NAMES, PINS)]
    launcher = (root / "src/slurm/runs/ministral3_router_common.sh").read_text()
    pins = re.findall(r"ministral3_stage_model (models--\S+) ([a-f0-9]{40})", launcher)
    if len(pins) != 3:
        raise ValueError("Cannot resolve all three actual Ministral launcher pins")
    return [root.parent / ".cache/huggingface/hub" / name / "snapshots" / pin
            for name, pin in pins]


async def validate_model(root: Path, path: Path, predictor_dir: Path, family: str) -> dict:
    import msgspec
    import torch
    from vllm.config import CacheConfig, ModelConfig, SchedulerConfig, VllmConfig
    from vllm.entrypoints.openai.protocol import ChatCompletionRequest
    from vllm.entrypoints.openai.serving_engine import OpenAIServing
    from vllm.sampling_params import SamplingParams
    from vllm.v1.core.sched.scheduler import Scheduler
    from vllm.v1.engine import EngineCoreRequest
    from vllm.v1.engine.async_llm import AsyncLLM
    from vllm.v1.engine.output_length_predictor import OutputLengthPredictor
    from vllm.v1.engine.processor import Processor
    from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig, KVCacheGroupSpec
    from vllm.v1.outputs import ModelRunnerOutput
    from vllm.v1.request import Request
    from vllm.v1.structured_output import StructuredOutputManager

    mistral = family == "ministral3"
    overrides = {} if mistral else {"rope_scaling": {
        "rope_type": "yarn", "factor": 4.0, "original_max_position_embeddings": 32768}}
    model = ModelConfig(model=str(path), tokenizer=str(path), dtype="auto",
        max_model_len=131072, tokenizer_mode="mistral" if mistral else "auto",
        config_format="mistral" if mistral else "auto", hf_overrides=overrides)
    cache = CacheConfig(block_size=16, enable_prefix_caching=False, swap_space=0)
    cache.num_gpu_blocks = 10000  # Bookkeeping capacity only; no device tensors are allocated.
    config = VllmConfig(model_config=model, cache_config=cache,
        scheduler_config=SchedulerConfig(max_model_len=131072, max_num_seqs=512,
            max_num_batched_tokens=32768, enable_chunked_prefill=True,
            enable_snapshot_shm_publishing=True, snapshot_shm_publish_interval_ms=0,
            snapshot_shm_name="cpu_contract_preflight", snapshot_shm_size_bytes=8388608,
            output_length_model_path=str(predictor_dir), async_scheduling=False))
    processor = Processor(config)
    engine = AsyncLLM.__new__(AsyncLLM)
    engine.model_config = model
    engine.vllm_config = config
    engine.processor = processor
    engine.output_length_predictor = OutputLengthPredictor(str(predictor_dir))
    serving = OpenAIServing(engine_client=engine, model_config=model, models=None,
                            request_logger=None)
    serving._processor = processor
    template = None if mistral else (root / "src/assets/templates/chat_template_qwen3.jinja").read_text()
    cases = [("warmup", [{"role": "user", "content": "warm-up"}], 1),
             ("chat", [{"role": "system", "content": "You are a helpful assistant."},
                       {"role": "user", "content": "Summarize the purpose of a queue in two sentences."}], 32),
             ("long_chat", [{"role": "system", "content": "You are a helpful assistant."},
                            {"role": "user", "content": "Summarize this report: " + "A request enters a queue. " * 300}], 8192)]
    reports = []
    try:
        for label, messages, max_tokens in cases:
            request = ChatCompletionRequest(model=path.parent.parent.name, messages=messages,
                temperature=0, top_p=1, max_completion_tokens=max_tokens)
            _, request_prompts, prompts = await serving._preprocess_chat(request,
                processor.tokenizer, messages, template, "auto",
                chat_template_kwargs={} if mistral else {"enable_thinking": False})
            text, _, _ = serving._get_prompt_components(request_prompts[0])
            params = SamplingParams(temperature=0, top_p=1, max_tokens=max_tokens)
            core, _ = await serving._process_inputs(label, prompts[0], params,
                lora_request=None, trace_headers=None, priority=0)
            before = list(core.prompt_token_ids)
            engine._attach_output_length_prediction(core, text)
            # The actual cross-process payload and the engine-side conversion.
            transported = msgspec.msgpack.decode(msgspec.msgpack.encode(core), type=EngineCoreRequest)
            scheduled_request = Request.from_engine_core_request(transported, block_hasher=None)
            scheduler = Scheduler(vllm_config=config,
                kv_cache_config=KVCacheConfig(num_blocks=10000, kv_cache_tensors=[],
                    kv_cache_groups=[KVCacheGroupSpec(["layer"],
                        FullAttentionSpec(16, 1, 1, torch.float32, False))]),
                structured_output_manager=StructuredOutputManager(config), log_stats=False)
            snapshots = []
            scheduler.set_snapshot_consumer(snapshots.append, interval_s=0)
            scheduler.add_request(scheduled_request)
            scheduled = scheduler.schedule()  # Reproduces the observed engine assertion before the fix.
            assert snapshots and snapshots[-1].inflight_batch is not None
            predicted = transported.predicted_output_tokens_mean
            assert predicted is not None and math.isfinite(predicted) and predicted > 0
            assert core.prompt_token_ids == before == transported.prompt_token_ids
            scheduler.update_from_output(scheduled, ModelRunnerOutput(req_ids=[label],
                req_id_to_index={label: 0}, sampled_token_ids=[[processor.tokenizer.eos_token_id]],
                logprobs=None, prompt_logprobs_dict={}, pooler_output=[]))
            assert label not in scheduler.requests
            reports.append({"case": label, "token_only_frontend_prompt": text is None,
                            "prompt_tokens": len(before), "predicted_output_tokens": predicted,
                            "snapshots": len(snapshots)})
    finally:
        serving._tokenizer_executor.shutdown(wait=True)
    return {"model_path": str(path), "cases": reports, "status": "PASS"}


def validate(root: Path, family: str, predictor_dir: Path, *, first_model_only=False) -> dict:
    from vllm.platforms.cpu import CpuPlatform
    paths = model_paths(root, family)
    if first_model_only:
        paths = paths[:1]
    # All real tokenizer, predictor, protocol, and scheduler code runs on CPU.
    # Forward execution is represented by one EOS token; this is not a GPU smoke.
    with patch("vllm.platforms._current_platform", CpuPlatform()):
        reports = [asyncio.run(validate_model(root, path, predictor_dir, family)) for path in paths]
    return {"status": "PASS", "family": family, "models": reports,
            "gpu_executed": False, "model_forward_executed": False,
            "actual_chat_preprocessing": True, "actual_request_processor": True,
            "real_fitted_predictor": True, "actual_request_serialization": True,
            "actual_scheduler_and_snapshot": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sfs-root", required=True, type=Path)
    parser.add_argument("--family", required=True, choices=("ministral3", "qwen"))
    parser.add_argument("--predictor-dir", required=True, type=Path)
    parser.add_argument("--first-model-only", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = validate(args.sfs_root, args.family, args.predictor_dir,
                      first_model_only=args.first_model_only)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
