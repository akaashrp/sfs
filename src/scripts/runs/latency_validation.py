"""Source-bound validation gates for the vLLM-SR adaptation."""
import hashlib,json
from pathlib import Path
from sfs_core.routing.latency_history import audit_history

def source_hashes(root):
    root=Path(root)
    paths=[*sorted((root/'src/sfs_core/routing').glob('latency_*.py')),
           root/'src/sfs_core/routing/wait_time_scheduler.py',root/'src/sfs_core/routing/methodology_scheduler.py',
           root/'src/scripts/runs/experiments.py',root/'src/scripts/runs/experiments_sweep.py',
           root/'src/scripts/runs/qwen_baselines.py',root/'src/scripts/runs/qwen_prefill_bootstrap.py',root/'src/scripts/runs/latency_validation.py',
           root/'src/scripts/runs/ministral3_methodology_stage.py',root/'src/sfs_core/shared/shared_experiment_helpers.py',
           root/'vllm/vllm/v1/engine/async_llm.py',root/'vllm/vllm/v1/engine/output_length_predictor.py',
           root/'vllm/vllm/v1/core/sched/scheduler.py',root/'vllm/vllm/engine/arg_utils.py',
           root/'src/scripts/reporting/collate_qwen_baselines.py',root/'src/scripts/reporting/router_qps_sweep_summary.py',
           root/'src/slurm/runs/qwen_baselines.sbatch',root/'src/slurm/runs/qwen_prefill_bootstrap.sbatch',
           root/'src/scripts/runs/ministral3_prefill_bootstrap.py',root/'src/scripts/runs/ministral3_figure5.py',
           *sorted((root/'src/assets/vllm_sr_latency').glob('*'))]
    return {str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths if p.is_file()}

def validate_cpu(path, root):
    if not path:raise ValueError('A source-bound CPU validation report is required')
    report=json.loads(Path(path).read_text())
    if report.get('status')!='PASS_CPU' or report.get('source_sha256')!=source_hashes(root) or not report.get('upstream_differential_passed'):
        raise ValueError('CPU validation missing/stale; heavy runs remain gated')
    return report

def validate_selector_smoke(stage,root):
    stage=Path(stage)
    gate=json.loads((stage/'latency_smoke_gate.json').read_text())
    result_path=stage/'smoke_vllm_sr_latency.json'
    if gate.get('status')!='PASS_GPU_SMOKE' or gate.get('source_sha256')!=source_hashes(root) or gate.get('result_sha256')!=hashlib.sha256(result_path.read_bytes()).hexdigest():
        raise ValueError('Missing/stale latency GPU smoke evidence')
    run=json.loads(result_path.read_text())['runs'][0]
    report=audit_history(run['methodology_config'])
    if report['selections']!=len(run['per_request']) or gate.get('requests')!=len(run['per_request']):
        raise ValueError('Incomplete latency GPU smoke')
    from scripts.runs.ministral3_methodology_stage import audit_run
    audit_run(run,192)
