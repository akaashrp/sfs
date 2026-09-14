from __future__ import annotations

from copy import deepcopy

import pytest

from sfs_core.routing.methodology_snapshot import parse_baseline_snapshot


def _request(request_id, prompt, computed, output, status="RUNNING"):
    return {
        "request_id": request_id,
        "status": status,
        "num_prompt_tokens": prompt,
        "num_computed_tokens": computed,
        "num_output_processed_tokens": output,
        "num_output_target_tokens": 900000,  # SFS heuristic must be ignored.
        "kv_block_counts": [2],
    }


def _payload():
    return {
        "version": 8,
        "created_at": 10.0,
        "num_running": 3,
        "num_waiting": 1,
        "running_request_ids": ["partial", "decode", "finishing-prefill"],
        "waiting_request_ids": ["waiting"],
        "requests": {
            "partial": _request("partial", 12, 8, 0),
            "decode": _request("decode", 8, 11, 3),
            "finishing-prefill": _request("finishing-prefill", 8, 8, 0),
            "waiting": _request("waiting", 16, 0, 0, "WAITING"),
        },
        "config": {
            "max_num_batched_tokens": 128,
            "max_num_seqs": 512,
            "max_model_len": 4096,
            "chunked_prefill_enabled": True,
        },
        "kv_cache_config": {
            "block_size": 16,
            "kv_cache_groups": [{"kv_cache_spec": {"block_size": 16}}],
            "kv_cache_free_blocks": 100,
            "kv_cache_total_blocks": 256,
        },
        "parallel_config": {"decode_context_parallel_size": 1},
        "inflight_batch": {
            "batch_prefill_tokens": 8,
            "batch_decode_tokens": 1,
            "batch_total_context_len": 27,
            "projected_output_request_indices": [1, 2],
            "scheduled_tokens_by_request": {
                "partial": 4,
                "decode": 1,
                "finishing-prefill": 4,
            },
        },
        "decode_backlog_total_tokens": 900000,
        "decode_reserve_tokens": 100000,
    }


def _idle_payload():
    result = _payload()
    result.update(num_running=0, num_waiting=0, running_request_ids=[], waiting_request_ids=[], requests={}, inflight_batch=None)
    return result


def test_committed_progress_preserves_entire_inflight_prefill_and_actual_outputs():
    state = parse_baseline_snapshot(_payload(), observed_at=10.1)
    partial = state.requests["partial"]
    assert partial.committed_computed_tokens == 4
    assert partial.computed_prompt_tokens == 4
    assert partial.remaining_prompt_tokens == 8
    assert partial.scheduled_prefill_tokens == 4
    finishing = state.requests["finishing-prefill"]
    assert finishing.remaining_prompt_tokens == 4
    assert finishing.generated_tokens == 0  # Projected output is not observed.
    decode = state.requests["decode"]
    assert decode.remaining_prompt_tokens == 0
    assert decode.generated_tokens == 3
    assert decode.scheduled_decode_tokens == 1
    assert state.decode_batch_size == 1
    assert state.decode_running_count == 1
    assert state.num_running == 3  # Prefill residents are not decode batch size.
    assert state.requests["waiting"].queue_phase == "waiting"
    assert state.age_ms == pytest.approx(100)
    assert state.tpot_features == {"decode_tokens": 1.0, "prefill_tokens": 8.0, "context_tokens": 27.0}


def test_newer_committed_snapshot_updates_progress_without_inventing_active_batch():
    raw = _payload()
    raw["inflight_batch"] = None
    raw["requests"]["decode"]["num_output_processed_tokens"] = 4
    raw["requests"]["finishing-prefill"]["num_output_processed_tokens"] = 1
    state = parse_baseline_snapshot(raw, observed_at=10)
    assert state.requests["partial"].remaining_prompt_tokens == 4
    assert state.requests["finishing-prefill"].remaining_prompt_tokens == 0
    assert state.requests["finishing-prefill"].generated_tokens == 1
    assert state.decode_running_count == 2
    assert state.decode_batch_size == 0
    assert state.tpot_features == {"decode_tokens": 0.0, "prefill_tokens": 0.0, "context_tokens": 0.0}


def test_decode_tokens_for_tpot_training_are_distinct_from_sequence_occupancy():
    raw = _payload()
    raw["requests"]["decode"].update(num_computed_tokens=13, num_output_processed_tokens=5)
    raw["inflight_batch"]["scheduled_tokens_by_request"]["decode"] = 3
    raw["inflight_batch"]["batch_decode_tokens"] = 3
    state = parse_baseline_snapshot(raw, observed_at=10)
    assert state.decode_batch_size == 1
    assert state.tpot_features["decode_tokens"] == 3


def test_reobserving_does_not_extrapolate_execution_or_duplicate_work():
    state = parse_baseline_snapshot(_payload(), observed_at=10)
    later = state.observed_again(20)
    assert later.version == state.version
    assert later.requests == state.requests
    assert later.requests["partial"].remaining_prompt_tokens == 8
    assert len(later.requests) == 4
    assert later.age_ms == 10000


def test_adapter_ignores_all_sfs_output_prediction_and_reserve_fields():
    raw = _payload()
    before = parse_baseline_snapshot(raw, observed_at=10)
    raw["decode_backlog_total_tokens"] = -999
    raw["decode_reserve_tokens"] = "not a number"
    for request in raw["requests"].values():
        request.pop("num_output_target_tokens")
    assert before == parse_baseline_snapshot(raw, observed_at=10)


@pytest.mark.parametrize("remove", [True, False])
def test_old_nonempty_inflight_payload_fails_with_actionable_diagnostic(remove):
    raw = _payload()
    if remove:
        raw["inflight_batch"].pop("scheduled_tokens_by_request")
    else:
        raw["inflight_batch"]["scheduled_tokens_by_request"] = None
    with pytest.raises(ValueError, match="restart servers"):
        parse_baseline_snapshot(raw, observed_at=10)


def test_empty_legacy_snapshot_needs_no_inflight_extension():
    state = parse_baseline_snapshot(_idle_payload(), observed_at=10)
    assert state.requests == {}
    assert state.inflight_total_tokens == 0


@pytest.mark.parametrize("change,match", [
    (lambda p: p["inflight_batch"]["scheduled_tokens_by_request"].update(partial=5), "batch totals"),
    (lambda p: p["requests"]["partial"].update(num_computed_tokens=2), "exceeds planned"),
    (lambda p: p["requests"]["partial"].update(num_prompt_tokens=1), "prefill totals"),
    (lambda p: p["waiting_request_ids"].append("partial"), "duplicate/overlapping"),
    (lambda p: p.update(num_running=5), "queue counts"),
    (lambda p: p["requests"].pop("partial"), "active queues"),
])
def test_inconsistent_payload_rejected(change, match):
    raw = _payload()
    change(raw)
    with pytest.raises(ValueError, match=match):
        parse_baseline_snapshot(raw, observed_at=10)


def test_header_version_timestamp_are_checked():
    with pytest.raises(ValueError, match="version mismatch"):
        parse_baseline_snapshot(_payload(), observed_at=10, expected_version=9)
    with pytest.raises(ValueError, match="timestamp mismatch"):
        parse_baseline_snapshot(_payload(), observed_at=10, expected_created_at=9)


def test_unused_sequence_limit_is_not_admission_evidence():
    state = parse_baseline_snapshot(_payload(), observed_at=10)
    evidence = state.admission_evidence(prompt_tokens=32, predicted_output_tokens=8)
    assert state.num_running < state.max_num_seqs
    assert not evidence.free_decode_slot
    assert evidence.reason == "waiting_requests_have_admission_priority"


def test_idle_admission_checks_kv_and_local_assignments_and_staleness():
    state = parse_baseline_snapshot(_idle_payload(), observed_at=10)
    evidence = state.admission_evidence(prompt_tokens=32, predicted_output_tokens=8)
    assert evidence.free_decode_slot
    assert evidence.proxy == "conservative_capacity_proxy"
    assert evidence.required_kv_blocks == 3
    assert not state.admission_evidence(prompt_tokens=32, predicted_output_tokens=8, local_outstanding_requests=1).free_decode_slot
    assert state.observed_again(12).admission_evidence(prompt_tokens=32, predicted_output_tokens=8).reason == "stale_snapshot"
    raw = _idle_payload()
    raw["kv_cache_config"]["kv_cache_free_blocks"] = 2
    assert parse_baseline_snapshot(raw, observed_at=10).admission_evidence(prompt_tokens=32, predicted_output_tokens=8).reason == "insufficient_kv_blocks"


@pytest.mark.parametrize("change,reason", [
    (lambda p: p["config"].update(max_num_seqs=0), "no_sequence_capacity"),
    (lambda p: p["config"].update(max_model_len=32), "predicted_context_exceeds_model_limit"),
    (lambda p: p["config"].update(max_num_batched_tokens=16, chunked_prefill_enabled=False), "insufficient_prefill_token_budget"),
    (lambda p: p["parallel_config"].update(decode_context_parallel_size=2), "unsupported_kv_topology"),
    (lambda p: p["kv_cache_config"]["kv_cache_groups"].append({}), "unsupported_kv_topology"),
])
def test_idle_admission_capacity_constraints(change, reason):
    raw = _idle_payload()
    change(raw)
    assert parse_baseline_snapshot(raw, observed_at=10).admission_evidence(prompt_tokens=32, predicted_output_tokens=8).reason == reason


def test_preempted_output_progress_does_not_erase_required_reprefill():
    raw = _idle_payload()
    raw.update(num_waiting=1, waiting_request_ids=["preempted"], requests={"preempted": _request("preempted", 32, 0, 10, "PREEMPTED")})
    request = parse_baseline_snapshot(raw, observed_at=10).requests["preempted"]
    assert request.remaining_prompt_tokens == 32
    assert request.generated_tokens == 10


def test_decode_only_server_can_offer_free_slot_with_token_and_kv_evidence():
    raw = _idle_payload()
    raw.update(num_running=1, running_request_ids=["decode"], requests={"decode": _request("decode", 8, 11, 4)})
    state = parse_baseline_snapshot(raw, observed_at=10)
    evidence = state.admission_evidence(prompt_tokens=32, predicted_output_tokens=8)
    assert evidence.free_decode_slot
    assert evidence.reason == "decode_only_sequence_token_and_kv_capacity"
    assert evidence.required_kv_blocks == 4  # New context plus resident growth.
    assert evidence.available_sequence_slots == 511
    assert evidence.available_iteration_tokens == 127
    raw["config"]["max_num_batched_tokens"] = 1
    assert parse_baseline_snapshot(raw, observed_at=10).admission_evidence(prompt_tokens=32, predicted_output_tokens=8).reason == "insufficient_prefill_token_budget"


def test_prefill_residents_do_not_prove_spare_next_iteration_token_budget():
    raw = _idle_payload()
    raw.update(num_running=1, running_request_ids=["partial"], requests={"partial": _request("partial", 32, 16, 0)})
    evidence = parse_baseline_snapshot(raw, observed_at=10).admission_evidence(prompt_tokens=32, predicted_output_tokens=8)
    assert not evidence.free_decode_slot
    assert evidence.reason == "unfinished_prefills_no_admission_proof"


def test_payload_not_mutated():
    raw = _payload()
    original = deepcopy(raw)
    parse_baseline_snapshot(raw, observed_at=10)
    assert raw == original
