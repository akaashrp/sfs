from __future__ import annotations

import random

import pytest

from sfs_core.routing.methodology_policies import (
    PrefillWork, RequestLifetimeLedger, RouteBalanceCandidate,
    RouteBalanceWeights, estimate_routebalance_latency_ms,
    merge_prefill_work, routebalance_cost, routebalance_lpt_order,
    select_lmdeploy, select_mooncake, select_routebalance,
)


def test_lmdeploy_uses_n_not_n_plus_one_and_randomizes_idle_ties():
    # N/speed picks idle slow; (N+1)/speed would pick the busy fast instance.
    decision = select_lmdeploy({"slow": 0, "fast": 1}, {"slow": 1, "fast": 100}, random.Random(9))
    assert decision.selected_instance_id == "slow"
    winners = {
        select_lmdeploy({"slow": 0, "fast": 0}, {"slow": 1, "fast": 100}, random.Random(seed))
        .selected_instance_id for seed in range(20)
    }
    assert winners == {"slow", "fast"}


def test_lmdeploy_matches_upstream_shuffle_then_strict_comparison():
    for seed in range(10):
        counts = {"a": 2, "b": 1, "c": 4}
        speeds = {"a": 2.0, "b": 1.0, "c": 3.0}
        order = list(counts)
        random.Random(seed).shuffle(order)
        selected = None
        minimum = float("inf")
        for instance in order:
            latency = counts[instance] / speeds[instance]
            if latency < minimum:
                selected, minimum = instance, latency
        actual = select_lmdeploy(counts, speeds, random.Random(seed))
        assert actual.selected_instance_id == selected
        assert actual.tie_candidates == ("a", "b")


def test_lifetime_reservations_survive_engine_observation_and_release_exactly_once():
    ledger = RequestLifetimeLedger(["a", "b"])
    ledger.reserve("success", "a", prompt_tokens=50, predicted_output_tokens=100)
    ledger.reserve("failed", "a")
    observed = [PrefillWork("success", 50, 50)]
    # Reconciliation removes completed prefill work but cannot release lifetime N.
    reservations = [PrefillWork(r.request_id, r.prompt_tokens) for r in ledger.reservations("a")]
    assert not merge_prefill_work(observed, reservations)
    assert ledger.unfinished_counts() == {"a": 2, "b": 0}
    assert ledger.release("failed")
    assert not ledger.release("failed")
    assert ledger.unfinished_counts() == {"a": 1, "b": 0}
    assert ledger.release("success")
    assert len(ledger) == 0


def test_duplicate_reservation_never_moves_an_outstanding_request():
    ledger = RequestLifetimeLedger(["a", "b"])
    ledger.reserve("r", "a")
    with pytest.raises(ValueError, match="already reserved"):
        ledger.reserve("r", "b")
    assert ledger.unfinished_counts() == {"a": 1, "b": 0}


def test_mooncake_sums_nonlinear_per_request_times_and_excludes_decode():
    def estimate(prompt, computed):
        return prompt**2 - computed**2
    decision = select_mooncake(
        {"a": [PrefillWork("a1", 4), PrefillWork("a2", 4), PrefillWork("decode", 100, 100)],
         "b": [PrefillWork("b1", 7, 3)]}, 2,
        {"a": estimate, "b": estimate}, random.Random(2),
    )
    # a's two requests cost 32, not 8**2; b's partial prefill costs 49-9.
    assert decision.selected_instance_id == "a"
    assert decision.candidates["a"]["total_prefill_ms"] == 36
    assert decision.candidates["b"]["total_prefill_ms"] == 44


def test_prefill_reconciliation_observation_overrides_reservation_before_filtering():
    merged = merge_prefill_work(
        [PrefillWork("partial", 100, 60), PrefillWork("decode", 20, 20)],
        [PrefillWork("partial", 100), PrefillWork("decode", 20), PrefillWork("unseen", 50)],
    )
    assert merged == (PrefillWork("partial", 100, 60), PrefillWork("unseen", 50))


def test_routebalance_cost_and_normalization_are_per_request():
    cost = routebalance_cost(prompt_tokens=100, predicted_output_tokens=10,
                             input_token_price=2, output_token_price=3)
    assert cost == 230
    decision = select_routebalance(
        {"a": RouteBalanceCandidate(.5, 1, 100), "b": RouteBalanceCandidate(.9, 6, 200)},
        RouteBalanceWeights(.2, .4, .4), random.Random(0),
    )
    assert decision.candidates["a"]["routebalance_score"] == pytest.approx(.1 + .4 * 5/6 + .2)
    assert decision.candidates["b"]["routebalance_score"] == pytest.approx(.18)
    assert decision.selected_instance_id == "a"
    zero = select_routebalance(
        {"a": RouteBalanceCandidate(.5, 0, 0), "b": RouteBalanceCandidate(.9, 0, 0)},
        RouteBalanceWeights(), random.Random(0),
    )
    assert zero.selected_instance_id == "b"
    assert zero.candidates["a"]["normalized_cost_benefit"] == 1
    assert zero.candidates["a"]["normalized_latency_benefit"] == 1


def test_routebalance_free_slot_and_prefill_only_divisor_are_explicit():
    busy, terms = estimate_routebalance_latency_ms(
        tpot_ms=10, pending_decode_tokens=120, decode_batch_size=4, predicted_output_tokens=20)
    assert busy == 500
    free, terms = estimate_routebalance_latency_ms(
        tpot_ms=10, pending_decode_tokens=120, decode_batch_size=4, predicted_output_tokens=20,
        free_decode_slot=True, free_slot_reason="validated_idle_capacity_proxy")
    assert free == 200
    assert terms["waiting_decode_steps"] == 0
    blocked, terms = estimate_routebalance_latency_ms(
        tpot_ms=10, pending_decode_tokens=120, decode_batch_size=0, predicted_output_tokens=20)
    assert blocked == 1400
    assert terms["empty_decode_batch_proxy"] is True
    empty, _ = estimate_routebalance_latency_ms(
        tpot_ms=10, pending_decode_tokens=0, decode_batch_size=0, predicted_output_tokens=20)
    assert empty == 200


def test_lpt_uses_maximum_model_length_and_stable_ties():
    order = routebalance_lpt_order({
        "short": {"a": 5, "b": 7}, "long": {"a": 2, "b": 10},
        "same": {"a": 10, "b": 8},
    })
    assert order == ["long", "same", "short"]


def test_immediate_assignment_work_changes_next_route_and_failure_rolls_back_only_one():
    ledger = RequestLifetimeLedger(["a", "b"])
    pending = {"a": 0, "b": 0}
    decisions = []
    for request_id in ("long", "short"):
        terms = {}
        length = 100 if request_id == "long" else 10
        for instance, work in pending.items():
            latency, diagnostic = estimate_routebalance_latency_ms(
                tpot_ms=1, pending_decode_tokens=work, decode_batch_size=1,
                predicted_output_tokens=length)
            terms[instance] = RouteBalanceCandidate(.5, 1, latency, diagnostic)
        decision = select_routebalance(terms, RouteBalanceWeights(0, 0, 1), random.Random(0))
        chosen = decision.selected_instance_id
        ledger.reserve(request_id, chosen, predicted_output_tokens=length)
        pending[chosen] += length
        decisions.append(chosen)
    assert decisions[0] != decisions[1]
    ledger.release("short")
    assert ledger.reservations()[0].request_id == "long"


@pytest.mark.parametrize("bad", [-1, float("nan"), float("inf"), True])
def test_invalid_inputs_fail_before_selection(bad):
    with pytest.raises(ValueError):
        select_lmdeploy({"a": 0}, {"a": bad}, random.Random(0))
    with pytest.raises(ValueError):
        RouteBalanceCandidate(bad, 1, 1)
    with pytest.raises(ValueError):
        PrefillWork("r", bad)


def test_missing_candidates_weights_and_duplicate_work_are_rejected():
    with pytest.raises(ValueError, match="same pool"):
        select_lmdeploy({"a": 1}, {}, random.Random(0))
    with pytest.raises(ValueError, match="sum to one"):
        RouteBalanceWeights(1, 1, 1)
    with pytest.raises(ValueError, match="within"):
        RouteBalanceCandidate(1.01, 1, 1)
    with pytest.raises(ValueError, match="duplicate"):
        select_mooncake({"a": [PrefillWork("r", 1), PrefillWork("r", 1)]},
                        1, {"a": lambda p, c: p-c}, random.Random(0))
