"""Local GPU pool with explicit ownership; no Slurm environment is fabricated."""
import contextlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import urllib.request
import uuid

from scripts.cloud.common import ROOT, read, write, set_option, locks


def config(family, definition, metrics, ports, tag):
    if family == "qwen":
        from scripts.runs.qwen_baselines import pool_config
        return pool_config({}, ports, tag)
    from scripts.runs.ministral3_latency import ALIASES
    rows = metrics.get("models", metrics)
    instances, costs = [], {}
    for i, (model, alias) in enumerate(ALIASES.items()):
        key = "vllm-" + model
        instances.append({"instance_id": key, "model_id": model, "default_model": alias,
            "served_model_name": alias, "address": f"http://127.0.0.1:{ports[i]}",
            "snapshot_shm_name": f"sfs_{tag}_{i}", "snapshot_shm_size_bytes": 8*1024*1024,
            "max_num_batched_tokens": 32768, "max_num_seqs": 512,
            "chunked_prefill_enabled": True, "long_prefill_token_threshold": 0,
            "ttft_batch_model": {k: rows[model]["sfs_simulation"][k] for k in
                ("intercept", "prefill_coeff", "prefill_sq_coeff", "decode_coeff", "sum_coeff", "sum_sq_coeff")}})
        costs[key] = {"prompt": (.10, .15, .20)[i], "output": (.10, .15, .20)[i]}
    return {"instances": instances, "instance_costs": costs, "cost_units": "USD per million tokens",
            "serving_profile": definition["profile"]}


def server_argv(family, model_path, row, index, output, length_predictor):
    if family == "qwen":
        from scripts.runs.qwen_baselines import server_argv as canonical_argv
        argv = canonical_argv(ROOT, model_path, row, index, output)
        argv = set_option(argv, "--output-length-model-path", length_predictor)
    else:
        argv = [sys.executable, "-m", "vllm.entrypoints.openai.api_server", "--model", str(model_path),
            "--served-model-name", row["default_model"], "--tokenizer-mode", "mistral",
            "--config-format", "mistral", "--load-format", "mistral", "--dtype", "auto",
            "--max-model-len", "131072", "--enable-chunked-prefill", "--no-enable-prefix-caching",
            "--max-num-batched-tokens", "32768", "--max-num-seqs", "512", "--gpu-memory-utilization", ".90",
            "--tensor-parallel-size", "1", "--limit-mm-per-prompt", '{"image":0}', "--mm-processor-cache-gb", "0",
            "--batch-stats-file", str(output/f"batch_stats_{row['model_id']}.csv"),
            "--port", row["address"].rsplit(":", 1)[1], "--no-enable-wait-time-simulation",
            "--enable-snapshot-shm-publishing", "--snapshot-shm-name", row["snapshot_shm_name"],
            "--snapshot-shm-size-bytes", str(row["snapshot_shm_size_bytes"]), "--snapshot-shm-publish-interval-ms", "0",
            "--output-length-model-path", str(length_predictor), "--disable-uvicorn-access-log"]
        for key, value in row["ttft_batch_model"].items():
            argv += ["--simulation-" + key.replace("_", "-"), str(value)]
    return [*argv, "--host", "127.0.0.1"]


def hardware(gpus):
    query = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid,name,memory.total,driver_version",
                                    "--format=csv,noheader,nounits"], text=True)
    selected = []
    for line in query.splitlines():
        fields = [v.strip() for v in line.split(",")]
        if fields[0] in gpus:
            if "H100" not in fields[2] or float(fields[3]) < 79000:
                raise ValueError("This campaign requires complete H100 80GB GPUs")
            selected.append(fields)
    if len(selected) != len(gpus) or len(set(gpus)) != len(gpus):
        raise ValueError("GPU selection is unavailable or duplicated")
    return {"host": socket.gethostname(), "boot_id": Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
            "gpus": selected, "topology": subprocess.check_output(["nvidia-smi", "topo", "-m"], text=True)}


@contextlib.contextmanager
def pool(family, definition, model_paths, bundle, output, gpus, state, length_predictor):
    if len(gpus) not in ((4,) if family == 'qwen' else (3, 4)):
        raise ValueError("Qwen requires four GPUs; Ministral requires three (legacy four-GPU lanes also accepted)")
    fingerprint = hardware(gpus)
    # Same lock namespace across different output directories/checkouts.
    lockdir = Path('/dev/shm') / f"sfs-cloud-locks-{os.getuid()}"
    with locks(lockdir, ["gpu-" + row[1] for row in fingerprint["gpus"]]):
        tag = uuid.uuid4().hex[:10]
        local = Path(state).resolve() / "ipc" / tag
        from scripts.runs.serving_ipc import qwen_ipc_environment
        env = qwen_ipc_environment(local, os.environ)
        reserved = []
        processes, logs = [], []
        try:
            for _ in range(3):
                sock = socket.socket(); sock.bind(("127.0.0.1", 0)); reserved.append(sock)
            ports = [s.getsockname()[1] for s in reserved]
            cfg = config(family, definition, read(Path(bundle)/family/'bridges_metrics.json'), ports, tag)
            path = Path(output) / "instances.json"
            write(path, cfg); write(Path(output)/"hardware.json", fingerprint)
            for i, row in enumerate(cfg["instances"]):
                argv = server_argv(family, model_paths[row["model_id"]], row, i, Path(output), length_predictor)
                visible = gpus[i] if family == "ministral" or i < 2 else ",".join(gpus[2:])
                server_env = dict(env, CUDA_VISIBLE_DEVICES=visible, VLLM_USE_V1="1",
                    VLLM_ATTENTION_BACKEND="FLASH_ATTN", VLLM_USE_FLASHINFER_SAMPLER="0",
                    VLLM_PER_REQUEST_WAIT_LOG_PATH=str(Path(output)/f"wait_{row['model_id']}.log"),
                    XDG_CACHE_HOME=str(Path(state)/"cache"), TRITON_CACHE_DIR=str(local/"triton"),
                    CUDA_CACHE_PATH=str(local/"nv"))
                write(Path(output)/f"server_argv_{row['model_id']}.json", argv)
                log = (Path(output)/f"server_{row['model_id']}.log").open("x"); logs.append(log)
                reserved[i].close()
                processes.append(subprocess.Popen(argv, cwd=ROOT/'src', env=server_env,
                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True))
            deadline = time.monotonic() + 1800
            for row in cfg["instances"]:
                while True:
                    if any(p.poll() is not None for p in processes):
                        raise RuntimeError("Server exited before readiness; inspect server logs")
                    try:
                        with urllib.request.urlopen(row['address']+'/v1/models', timeout=3) as response:
                            available = json.load(response)
                        if row['default_model'] in {r['id'] for r in available['data']}:
                            break
                    except (OSError, ValueError):
                        pass
                    if time.monotonic() > deadline:
                        raise TimeoutError("Server readiness exceeded 30 minutes")
                    time.sleep(2)
            yield path, fingerprint, processes
        finally:
            for sock in reserved:
                sock.close()
            for p in processes:
                try: os.killpg(p.pid, signal.SIGTERM)
                except ProcessLookupError: pass
            deadline = time.monotonic() + 30
            for p in processes:
                try: p.wait(timeout=max(.1, deadline-time.monotonic()))
                except subprocess.TimeoutExpired:
                    try: os.killpg(p.pid, signal.SIGKILL)
                    except ProcessLookupError: pass
                    p.wait()
            for log in logs: log.close()
