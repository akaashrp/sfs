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


def parse_remaining_length_rules(specs):
    """'MODEL=MODE[:QUANTILE[:CONDITIONING]]' entries -> {model: {mode, quantile, conditioning}}."""
    from vllm.v1.core.sched.remaining_length import CONDITIONINGS, MODES
    rules = {}
    for spec in specs or ():
        model, _, rest = spec.partition("=")
        parts = rest.split(":") if rest else []
        rule = {"mode": parts[0] if parts else "off", "quantile": float(parts[1]) if len(parts) > 1 else 0.5,
                "conditioning": parts[2] if len(parts) > 2 else "prompt_bin"}
        if not model or rule["mode"] not in MODES or rule["conditioning"] not in CONDITIONINGS or not 0 < rule["quantile"] <= 1:
            raise ValueError(f"Invalid remaining-length rule {spec!r}; expected MODEL=MODE[:QUANTILE[:CONDITIONING]]")
        rules[model] = rule
    return rules


def remaining_length_provenance(definition, remaining_length):
    """instances.json block labelling each engine's remaining-decode target rule.

    remaining_length is None (current rule everywhere) or {"tables": DIR, "rules": {model: {mode,
    quantile, conditioning}}}; DIR holds one <model_id>.json survival table per model built by
    scripts.prep.remaining_length_tables. Models without a rule, or with mode "off", keep the current rule.
    """
    from vllm.v1.core.sched.remaining_length import RemainingLengthTable, rule_name
    rules = dict((remaining_length or {}).get("rules") or {})
    if set(rules) - set(definition["models"]):
        raise ValueError(f"Remaining-length rules name unknown models: {sorted(set(rules) - set(definition['models']))}")
    models, labels = {}, []
    for model in definition["models"]:
        rule = dict(rules.get(model) or {"mode": "off", "quantile": 0.5, "conditioning": "prompt_bin"})
        rule["rule"] = rule_name(rule["mode"], rule["quantile"], rule["conditioning"])
        if rule["mode"] != "off":
            path = Path(remaining_length["tables"]).resolve() / f"{model}.json"
            table = RemainingLengthTable.load(path)
            if table.model != model:
                raise ValueError(f"Remaining-length table {path} is for {table.model}, not {model}")
            rule["table"] = {"path": str(path), "sha256": table.sha256, "support": {k: table.support(k) for k in
                             (*map(str, range(len(table.prompt_bin_edges) - 1)), "all")}}
            labels.append(f"{model}={rule['rule']}")
        models[model] = rule
    return {"rule": ";".join(labels) if labels else "current", "models": models,
            "semantics": "running requests with complete prefill: running_all = Q_q(total length | conditioning, total > "
                         "generated) - generated for every request, exhausted_only = the same only once prediction + reserve "
                         "leaves <= 1 token, both floored at generated + 1 and capped at max_tokens / decode budget; waiting "
                         "requests and probes keep prediction + reserve; the router pending-dispatch overlay of an engine with a "
                         "rule uses the table's unconditional median when the prediction is missing (1.0 default)"}


def server_argv(family, model_path, row, index, output, length_predictor, remaining_length=None):
    rule = ((remaining_length or {}).get("models") or {}).get(row["model_id"]) or {"mode": "off"}
    if family == "qwen":
        from scripts.runs.qwen_baselines import server_argv as canonical_argv
        argv = canonical_argv(ROOT, model_path, row, index, output)
        argv = set_option(argv, "--output-length-model-path", length_predictor)
        if rule["mode"] != "off":
            argv = set_option(argv, "--remaining-length-mode", rule["mode"])
            argv = set_option(argv, "--remaining-length-table", rule["table"]["path"])
            argv = set_option(argv, "--remaining-length-quantile", rule["quantile"])
            argv = set_option(argv, "--remaining-length-conditioning", rule["conditioning"])
    else:
        if rule["mode"] != "off":
            raise ValueError("The remaining-length rule is only plumbed for the Qwen family")
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


def instance_config(family, definition, bundle, ports, tag, profile="canonical", coefficients=None):
    if profile == "fcfs":
        from scripts.cloud.fcfs.config import instances
        return instances(Path(bundle), ports, tag, coefficients)
    return config(family, definition, read(Path(bundle)/family/'bridges_metrics.json'), ports, tag)


def instance_argv(family, model_path, row, index, output, length_predictor, bundle, profile="canonical", remaining_length=None):
    """remaining_length is the instances.json provenance block (remaining_length_provenance) or None."""
    if profile == "fcfs":
        from scripts.cloud.fcfs.config import server_argv as fcfs_argv
        return fcfs_argv(Path(bundle), model_path, row, index, Path(output), remaining_length)
    return server_argv(family, model_path, row, index, Path(output), length_predictor, remaining_length)


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
def pool(family, definition, model_paths, bundle, output, gpus, state, length_predictor, profile="canonical", coefficients=None,
         remaining_length=None):
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
            cfg = instance_config(family, definition, bundle, ports, tag, profile, coefficients)
            cfg["remaining_length"] = remaining_length_provenance(definition, remaining_length)
            path = Path(output) / "instances.json"
            write(path, cfg); write(Path(output)/"hardware.json", fingerprint)
            for i, row in enumerate(cfg["instances"]):
                argv = instance_argv(family, model_paths[row["model_id"]], row, i, output, length_predictor, bundle, profile,
                                     cfg["remaining_length"])
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
