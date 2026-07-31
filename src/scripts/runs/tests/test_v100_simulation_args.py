from __future__ import annotations

from pathlib import Path

from vllm.utils import FlexibleArgumentParser

from scripts.runs.service_metrics_config import build_simulation_args


V100_BATCH_SCRIPT = (
    Path(__file__).resolve().parents[3]
    / "slurm"
    / "runs"
    / "router"
    / "v100_rebuttal_qps.sbatch"
)


def test_real_vllm_parser_accepts_negative_cross_term_values_as_single_tokens():
    row = {
        "sfs_simulation": {
            "feature_set": "cross_term",
            "intercept": -0.061533336926986426,
            "prefill_coeff": 5.70555732257655e-6,
            "prefill_sq_coeff": 3.946809369486287e-8,
            "decode_coeff": 0.0014919342780246518,
            "sum_coeff": -9.6838576659913324e-7,
            "sum_sq_coeff": 0.0,
        }
    }
    # Exercise the exact FlexibleArgumentParser preprocessing that caused the
    # failed job, without constructing device-dependent EngineArgs on a CPU
    # test node.
    parser = FlexibleArgumentParser()
    parser.add_argument("--simulation-batch-time-feature-set")
    parser.add_argument("--simulation-intercept", type=float)
    parser.add_argument("--simulation-prefill-coeff", type=float)
    parser.add_argument("--simulation-prefill-sq-coeff", type=float)
    parser.add_argument("--simulation-decode-coeff", type=float)
    parser.add_argument("--simulation-sum-coeff", type=float)
    parser.add_argument("--simulation-sum-sq-coeff", type=float, default=0.0)
    parser.add_argument("--simulation-prefill-x-context-coeff", type=float)

    args = parser.parse_args(build_simulation_args(row))

    assert args.simulation_batch_time_feature_set == "cross_term"
    assert args.simulation_intercept == row["sfs_simulation"]["intercept"]
    assert args.simulation_sum_coeff == row["sfs_simulation"]["sum_coeff"]
    assert args.simulation_prefill_x_context_coeff == 0.0
    assert args.simulation_sum_sq_coeff == 0.0


def test_sweep_startup_gate_does_not_attach_to_snapshot_shared_memory():
    script = V100_BATCH_SCRIPT.read_text(encoding="utf-8")
    gate_start = script.index('if [[ "$SWEEP_STARTUP_GATE" == "1" ]]')
    gate_end = script.index("\nfi\n\nINSTANCES_CONFIG=", gate_start)
    gate = script[gate_start:gate_end]

    assert gate.count("smoke_request ") == 3
    assert "python " not in gate
    assert "from multiprocessing import shared_memory" not in script
    assert "SnapshotShmClient" not in script
    assert "read_snapshot_shm_header_once" not in script


def test_v100_batch_job_does_not_rebuild_or_install_its_runtime():
    script = V100_BATCH_SCRIPT.read_text(encoding="utf-8")

    assert "compile_vllm_scheduler_sim.sh" not in script
    assert "pip install" not in script
