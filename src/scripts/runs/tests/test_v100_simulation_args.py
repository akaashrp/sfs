from __future__ import annotations

from vllm.utils import FlexibleArgumentParser

from scripts.runs.service_metrics_config import build_simulation_args


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
