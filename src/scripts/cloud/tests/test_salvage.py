"""Bounded infrastructure-fault salvage: allowlist, cap, penalty, provenance and the quality join."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.cloud.common import digest, read, write
from scripts.cloud.salvage import (AUTHORIZATION, COVERAGE_FIELDS, MAX_SALVAGEABLE_REQUESTS, SLO_METRICS,
                                   SalvageError, apply, audit_completed_cell, audit_salvaged_cell, build,
                                   cause_of, cell_spec, classify_failures, salvage_cap, salvage_summary)

SHM = 'RuntimeError: Failed to read a consistent scheduler snapshot header from SHM sfs_f289bdc0c8_1'
WAIT = 'RuntimeError: Scheduler simulation failed: Timed out waiting for the parsed scheduler snapshot to catch up'
RULE = 'qwen3-0.6b=running_all_prompt_bin_q50'


def _row(index, qps, error=None):
    """One recorded per-request row, produced the way scripts.runs.experiments records it."""
    met = index % 2 == 0
    row = {'request_id': f'req-{index}', 'bucket': 'alpaca', 'system_entry_offset_s': index/qps,
           'ttft_slo_ms': 200., 'latency_slo_ms': 2000., 'queue_slo_ms': 500.,
           'response_id': f'resp-{index}', 'response_model': 'qwen3-8b', 'instance_id': 'inst-0',
           'queue_delay_ms': 1., 'ttft_ms': 100., 'e2e_ttft_ms': 110.,
           'system_entry_e2e_ttft_ms': 120. if met else 300., 'system_entry_to_dispatch_ms': 2.,
           'latency_ms': 150., 'actual_cost': 1., 'usage_completion_tokens': 10,
           'actual_accuracy': None, 'error': None}
    row.update({flag: met for flag in SLO_METRICS})
    if error is None:
        return row
    # A failed request keeps its identity and arrival telemetry and loses everything else.
    row.update({flag: False for flag in SLO_METRICS})
    row.update({field: None for field in COVERAGE_FIELDS})
    row.update(error=error, response_id=None, response_model=None, instance_id=None,
               actual_cost=None, usage_completion_tokens=None, latency_ms=150.)
    return row


def _summary(rows, denominator=None):
    """The producer's own summary arithmetic: every SLO divides by the full request count."""
    total = denominator if denominator is not None else len(rows)
    failed = [r for r in rows if r['error']]
    summary = {'total_requests': len(rows), 'succeeded_requests': len(rows)-len(failed), 'failed_requests': len(failed)}
    for flag, metric in SLO_METRICS.items():
        summary[metric] = 100.*sum(bool(r.get(flag)) for r in rows)/total
    for field, metric in COVERAGE_FIELDS.items():
        summary[metric] = sum(1 for r in rows if r.get(field) is None)
    return summary


def _payload(requests=4000, qps=6., errors=(), policy='score', **summary_overrides):
    """A completed point whose failing request ids carry the given error strings."""
    errors = dict(errors)
    rows = [_row(i, qps, errors.get(f'req-{i}')) for i in range(requests)]
    summary = {**_summary(rows), **summary_overrides}
    return {'config': {'request_rate_qps': qps, 'lambda_weight': .5},
            'request_set': {'num_requests': requests},
            'router': {'runs': [{'utility': policy, 'per_request': rows, 'summary': summary,
                                 'remaining_length_rule': RULE, 'snapshot_staleness_ms': 0.}]}}


def _cell(requests=4000, qps=6., policy='score', cid='qwen-score-6'):
    return {'id': cid, 'family': 'qwen', 'variant': 'canonical', 'policy': policy, 'qps': qps, 'requests': requests}


def _pool(tmp_path, cell, payload, name='pool'):
    """A released pool directory holding the preserved cell, plus its overlay and state directory."""
    pool = tmp_path/name
    (pool/'cells'/cell['id']).mkdir(parents=True)
    point = pool/'cells'/cell['id']/'point.json'
    write(point, payload)
    campaign = tmp_path/'campaign.json'
    write(campaign, {'schema_version': 1, 'kind': 'sfs_score', 'cells': [cell]})
    evidence = pool/'model_metrics.json'
    write(evidence, {'ok': True})
    qualification = {'status': 'GPU_MEASURED_REVIEW_REQUIRED', 'family': cell['family'], 'variant': cell['variant'],
                     'hardware': {'host': 'vast', 'gpus': [['0', 'uuid', 'NVIDIA H100 80GB HBM3', '81559', '580']]},
                     'source_sha256': {'src/scripts/cloud/worker.py': 'a'*64}, 'bundle_sha256': 'b'*64,
                     'files': {'model_metrics.json': digest(evidence)}, 'campaign_sha256': digest(campaign),
                     'campaign_kind': 'sfs_score', 'remaining_length_rule': RULE,
                     'remaining_length': {'rule': RULE, 'models': {}}, 'snapshot_staleness_levels_ms': [0.]}
    write(pool/'qualification.json', qualification)
    write(pool/'release.json', {'status': 'RELEASED', 'qualification_sha256': digest(pool/'qualification.json'),
                                'timing_review': 'reviewed', 'load_review': 'reviewed'})
    state = tmp_path/'state'
    (state/'completed').mkdir(parents=True, exist_ok=True)
    return SimpleNamespace(pool=pool, point=point, campaign=campaign, state=state, cell=cell)


def test_cap_is_the_smaller_of_five_requests_and_five_hundredths_of_a_percent():
    assert salvage_cap(16000) == MAX_SALVAGEABLE_REQUESTS == 5 and salvage_cap(100000) == 5
    assert salvage_cap(8000) == 4 and salvage_cap(4000) == 2 and salvage_cap(100) == 0
    assert cause_of(SHM) == 'shm_header_read' and cause_of(WAIT) == 'snapshot_wait_timeout'
    assert cause_of('Timed out waiting for the parsed scheduler snapshot') is None  # unwrapped: not the native fault
    for other in ('httpx.ConnectError: connection refused', 'CancelledError', 'asyncio.TimeoutError',
                  'RuntimeError: Engine core proc died', '', None):
        assert cause_of(other) is None


def test_salvageable_cell_is_admitted_with_penalty_and_provenance(tmp_path):
    cell = _cell()
    setup = _pool(tmp_path, cell, _payload(errors={'req-11': SHM, 'req-12': SHM}))
    result = apply(setup.point, cell, setup.pool, setup.campaign, setup.state)
    assert result['action'] == 'written'
    entry = read(setup.state/'completed'/'qwen-score-6.json')
    # The audit_cell-equivalent fields are unchanged; the salvage block is what marks the cell.
    assert entry['status'] == 'PASS_CELL' and entry['requests'] == 4000
    assert entry['realized_qps'] == pytest.approx(6., rel=1e-9)
    assert entry['cell'] == cell and entry['point_sha256'] == digest(setup.point)
    assert entry['remaining_length_rule'] == RULE and entry['snapshot_staleness_ms'] == 0.
    assert entry['qualification_sha256'] == digest(setup.pool/'qualification.json')
    assert entry['campaign_sha256'] == digest(setup.campaign) and entry['bundle_sha256'] == 'b'*64
    salvage = entry['salvage']
    assert salvage['status'] == 'SALVAGED' and salvage['failed_requests'] == 2
    assert salvage['cause_counts'] == {'shm_header_read': 2} and salvage['request_ids'] == ['req-11', 'req-12']
    assert salvage['cap'] == {'cap_used': 2, 'max_requests': 5, 'max_fraction_pct': .05, 'cell_requests': 4000}
    assert salvage['authorization'] == AUTHORIZATION == 'user authorised 2026-09-17'
    assert salvage['authorized_by'] == 'user' and 'never dropped' in salvage['penalty']
    assert set(salvage['causes_by_request']) == {'req-11', 'req-12'}
    evidence = salvage['penalty_evidence']
    assert evidence['denominator'] == 4000 and evidence['penalised_requests'] == 2
    # Recorded attainments keep the failures in the denominator; dropping them would score higher.
    for metric, value in evidence['slo_attainment_pct_recorded'].items():
        assert value == pytest.approx(100.*1999/4000) and evidence['slo_attainment_pct_without_penalty'][metric] > value
    record = read(setup.point.parent/'salvage.json')
    # The record is the ledger entry plus its provenance: the collation discovers the cell through
    # it, because the pool never wrote an audit.json for a run that failed.
    assert all(record[k] == v for k, v in entry.items() if k != 'recorded_at')
    assert record['record'] == 'salvage' and record['pool'] == str(setup.pool)
    assert record['campaign'] == str(setup.campaign) and record['ledger_entry'] == 'qwen-score-6.json'
    assert salvage_summary(entry) == {'status': 'SALVAGED', 'penalised_requests': 2,
                                      'cause_counts': {'shm_header_read': 2}, 'request_ids': ['req-11', 'req-12'],
                                      'authorization': AUTHORIZATION}


def test_one_request_over_the_cap_is_refused(tmp_path):
    cell = _cell()
    at_cap = _payload(errors={'req-11': SHM, 'req-12': WAIT})
    audit, salvage = audit_salvaged_cell(at_cap, cell)
    assert salvage['failed_requests'] == 2 and audit['status'] == 'PASS_CELL'
    over = _payload(errors={'req-11': SHM, 'req-12': SHM, 'req-13': SHM})
    assert classify_failures(over, cell)['within_allowlist'] and not classify_failures(over, cell)['salvageable']
    with pytest.raises(SalvageError, match='exceed the salvage cap of 2'):
        audit_salvaged_cell(over, cell)
    setup = _pool(tmp_path, cell, over)
    with pytest.raises(SalvageError, match='exceed the salvage cap'):
        apply(setup.point, cell, setup.pool, setup.campaign, setup.state)
    assert not (setup.state/'completed'/'qwen-score-6.json').exists()
    assert not (setup.point.parent/'salvage.json').exists()


def test_the_absolute_five_request_bound_binds_on_a_large_cell():
    big = _cell(requests=20000)
    assert salvage_cap(20000) == 5
    five = _payload(requests=20000, errors={f'req-{i}': WAIT for i in range(5)})
    assert audit_salvaged_cell(five, big)[1]['failed_requests'] == 5
    six = _payload(requests=20000, errors={f'req-{i}': WAIT for i in range(6)})
    with pytest.raises(SalvageError, match='exceed the salvage cap of 5'):
        audit_salvaged_cell(six, big)


@pytest.mark.parametrize('error', ['httpx.ConnectError: [Errno 111] Connection refused',
                                   'asyncio.TimeoutError', 'CancelledError',
                                   'RuntimeError: Engine core initialization failed'])
def test_a_non_allowlisted_error_is_refused(error):
    cell = _cell()
    payload = _payload(errors={'req-11': error})
    assert classify_failures(payload, cell)['salvageable'] is False
    with pytest.raises(SalvageError, match='outside the infrastructure allowlist'):
        audit_salvaged_cell(payload, cell)


def test_mixed_causes_are_refused():
    cell = _cell()
    payload = _payload(errors={'req-11': SHM, 'req-12': 'httpx.ReadTimeout'})
    report = classify_failures(payload, cell)
    assert report['cause_counts'] == {'shm_header_read': 1} and len(report['unsalvageable']) == 1
    assert report['within_cap'] and not report['within_allowlist'] and not report['salvageable']
    with pytest.raises(SalvageError, match='outside the infrastructure allowlist'):
        audit_salvaged_cell(payload, cell)


def test_a_missing_response_without_a_recorded_error_is_refused():
    cell = _cell()
    payload = _payload()
    payload['router']['runs'][0]['per_request'][7]['response_id'] = None
    with pytest.raises(SalvageError, match='outside the infrastructure allowlist|missing its response'):
        audit_salvaged_cell(payload, cell)


def test_failed_requests_must_stay_in_every_denominator():
    cell = _cell()
    rows = _payload(errors={'req-11': SHM})['router']['runs'][0]['per_request']
    # A summary that quietly divided by the surviving requests instead is refused outright.
    dropped = _payload(errors={'req-11': SHM}, **_summary(rows, denominator=3999))
    with pytest.raises(SalvageError, match='penalised attainment over the full denominator'):
        audit_salvaged_cell(dropped, cell)
    counted = _payload(errors={'req-11': SHM}, succeeded_requests=4000)
    with pytest.raises(SalvageError, match='does not account for every request'):
        audit_salvaged_cell(counted, cell)


def test_a_failed_request_recorded_as_meeting_an_slo_is_refused():
    cell = _cell()
    payload = _payload(errors={'req-11': SHM})
    row = next(r for r in payload['router']['runs'][0]['per_request'] if r['request_id'] == 'req-11')
    row['queue_slo_met'] = True
    with pytest.raises(SalvageError, match='recorded as meeting'):
        audit_salvaged_cell(payload, cell)
    row['queue_slo_met'], row['actual_accuracy'] = False, .8
    with pytest.raises(SalvageError, match='nonzero quality score'):
        audit_salvaged_cell(payload, cell)


def test_other_audit_conditions_still_apply():
    cell = _cell()
    slow = _payload(qps=5., errors={'req-11': SHM})
    slow['config']['request_rate_qps'] = 6.
    with pytest.raises(SalvageError, match='more than 10%'):
        audit_salvaged_cell(slow, cell)
    mismatched = _payload(errors={'req-11': SHM}, policy='hard')
    with pytest.raises(SalvageError, match='Cell identity mismatch'):
        audit_salvaged_cell(mismatched, cell)
    missing = _payload(errors={'req-11': SHM})
    missing['router']['runs'][0]['per_request'][-1]['request_id'] = 'req-9999999'
    with pytest.raises(SalvageError, match='request identities'):
        audit_salvaged_cell(missing, cell)
    latency = _payload(errors={'req-11': SHM}, policy='vllm_sr_latency')
    with pytest.raises(SalvageError, match='latency selector trace'):
        audit_salvaged_cell(latency, _cell(policy='vllm_sr_latency'))
    broken = _payload(errors={'req-11': SHM})
    broken['router']['runs'][0]['per_request'][3]['actual_cost'] = None
    with pytest.raises(ValueError, match='Missing/invalid measured actual_cost'):
        audit_salvaged_cell(broken, cell)
    # audit_cell's snapshot-read-fault gate binds a salvaged cell exactly as it binds a clean one.
    lost = _payload(errors={'req-11': SHM})
    lost['router']['runs'][0]['summary']['snapshot_read_faults'] = {'totals': {'requests_without_estimate': 1}}
    with pytest.raises(SalvageError, match='lost every candidate'):
        audit_salvaged_cell(lost, cell)
    lost['router']['runs'][0]['summary']['snapshot_read_faults'] = {'totals': {'requests_without_estimate': 0}}
    assert audit_salvaged_cell(lost, cell)[1]['failed_requests'] == 1


def test_pool_pins_are_checked(tmp_path):
    cell = _cell()
    setup = _pool(tmp_path, cell, _payload(errors={'req-11': SHM}))
    write(setup.pool/'release.json', {'status': 'DRAFT'})
    with pytest.raises(SalvageError, match='qualification and reviewed release'):
        build(setup.point, cell, setup.pool, setup.campaign)
    setup = _pool(tmp_path, cell, _payload(errors={'req-11': SHM}), name='pool2')
    write(setup.pool/'model_metrics.json', {'ok': False})
    with pytest.raises(SalvageError, match='Qualification evidence changed'):
        build(setup.point, cell, setup.pool, setup.campaign)
    setup = _pool(tmp_path, cell, _payload(errors={'req-11': SHM}), name='pool3')
    write(setup.campaign, {'schema_version': 1, 'kind': 'sfs_score', 'cells': [cell], 'note': 'edited'})
    with pytest.raises(SalvageError, match='different campaign overlay'):
        build(setup.point, cell, setup.pool, setup.campaign)


def test_clean_cell_is_unaffected(tmp_path):
    from scripts.cloud.worker import audit_cell, salvage_classification
    cell = _cell()
    payload = _payload()
    audit = audit_cell(payload, cell)
    assert audit == {'status': 'PASS_CELL', 'requests': 4000, 'realized_qps': pytest.approx(6.)}
    assert audit_completed_cell(payload, cell, {'status': 'PASS_CELL'}) == (audit, None)
    assert salvage_classification(payload, cell)['salvageable'] is False
    with pytest.raises(SalvageError, match='no failed requests'):
        audit_salvaged_cell(payload, cell)
    setup = _pool(tmp_path, cell, payload)
    with pytest.raises(SalvageError, match='no failed requests'):
        apply(setup.point, cell, setup.pool, setup.campaign, setup.state)
    # The worker still stops the pool on a salvageable failure during a run.
    failing = _payload(errors={'req-11': SHM})
    assert salvage_classification(failing, cell)['salvageable'] is True
    with pytest.raises(ValueError, match='Trial has request errors or missing responses'):
        audit_cell(failing, cell)


def test_ledger_entry_is_idempotent(tmp_path):
    cell = _cell()
    setup = _pool(tmp_path, cell, _payload(errors={'req-11': SHM}))
    first = apply(setup.point, cell, setup.pool, setup.campaign, setup.state)
    ledger, sidecar = setup.state/'completed'/'qwen-score-6.json', setup.point.parent/'salvage.json'
    before = (ledger.read_bytes(), sidecar.read_bytes())
    assert first['action'] == 'written'
    again = apply(setup.point, cell, setup.pool, setup.campaign, setup.state)
    assert again['action'] == 'unchanged'
    assert (ledger.read_bytes(), sidecar.read_bytes()) == before
    assert apply(setup.point, cell, setup.pool, setup.campaign, setup.state, dry_run=True)['action'] == 'dry_run'
    assert (ledger.read_bytes(), sidecar.read_bytes()) == before
    # A ledger entry that disagrees with this salvage is never silently overwritten.
    write(ledger, {**read(ledger), 'salvage': {**read(ledger)['salvage'], 'failed_requests': 4}})
    with pytest.raises(SalvageError, match='disagrees with this salvage'):
        apply(setup.point, cell, setup.pool, setup.campaign, setup.state)
    write(ledger, {'status': 'PASS_CELL', 'cell': cell})
    with pytest.raises(SalvageError, match='already admitted without salvage'):
        apply(setup.point, cell, setup.pool, setup.campaign, setup.state)


def test_recorded_salvage_must_still_match_the_point(tmp_path):
    cell = _cell()
    setup = _pool(tmp_path, cell, _payload(errors={'req-11': SHM}))
    apply(setup.point, cell, setup.pool, setup.campaign, setup.state)
    entry = read(setup.state/'completed'/'qwen-score-6.json')
    audit, salvage = audit_completed_cell(read(setup.point), cell, entry)
    assert audit['status'] == 'PASS_CELL' and salvage['request_ids'] == ['req-11']
    laundered = {**entry, 'salvage': {**entry['salvage'], 'request_ids': ['req-12']}}
    with pytest.raises(SalvageError, match='no longer matches the point'):
        audit_completed_cell(read(setup.point), cell, laundered)
    forged = {**entry, 'salvage': {**entry['salvage'], 'authorization': 'nobody'}}
    with pytest.raises(SalvageError, match='unrecognised authorisation'):
        audit_completed_cell(read(setup.point), cell, forged)
    # A clean point that acquires a salvage block is refused outright.
    with pytest.raises(SalvageError, match='no failed requests'):
        audit_completed_cell(_payload(), cell, entry)


def test_collation_discovers_a_salvaged_cell_through_its_record(tmp_path):
    from scripts.cloud.collate import cell_records, check_record
    cell = _cell()
    setup = _pool(tmp_path, cell, _payload(errors={'req-11': SHM}))
    apply(setup.point, cell, setup.pool, setup.campaign, setup.state)
    found = cell_records(setup.pool)
    assert found == [setup.point.parent/'salvage.json']
    record = read(found[0])
    assert check_record(record, {cell['id']: cell}, digest(setup.campaign), 'sfs_score', {'qwen': RULE}) == cell['id']
    audit, salvage = audit_completed_cell(read(setup.point), cell, record)
    assert salvage['failed_requests'] == 1 and audit['requests'] == 4000
    # A clean cell is still discovered through the pool's own audit.json, and never through both.
    clean = _pool(tmp_path, _cell(cid='qwen-score-8'), _payload(), name='clean')
    write(clean.point.parent/'audit.json', {'cell': clean.cell})
    assert cell_records(clean.pool) == [clean.point.parent/'audit.json']
    write(clean.point.parent/'salvage.json', {'cell': clean.cell})
    with pytest.raises(ValueError, match='both an audit and a salvage record'):
        cell_records(clean.pool)


def test_cell_spec_comes_from_the_overlay(tmp_path):
    cell = _cell()
    campaign = tmp_path/'campaign.json'
    write(campaign, {'schema_version': 1, 'kind': 'sfs_score', 'cells': [cell]})
    assert cell_spec(campaign, 'qwen-score-6') == cell
    with pytest.raises(SalvageError, match='no cell qwen-score-9'):
        cell_spec(campaign, 'qwen-score-9')


def test_observed_utilities_penalise_a_failed_row_without_raising():
    from scripts.cloud.collate import observed_utilities
    rows = [{'request_id': 'req-0', 'bucket': 'alpaca', 'system_entry_e2e_ttft_slo_met': True,
             'system_entry_e2e_ttft_ms': 100., 'ttft_slo_ms': 200., 'response_model': 'qwen3-8b',
             'actual_cost': 1., 'error': None},
            {'request_id': 'req-1', 'bucket': 'alpaca', 'system_entry_e2e_ttft_slo_met': False,
             'system_entry_e2e_ttft_ms': None, 'ttft_slo_ms': 200., 'response_model': None,
             'actual_cost': None, 'error': SHM}]
    payload = {'config': {'lambda_weight': .5}, 'router': {'runs': [{'utility': 'score', 'per_request': rows,
               'summary': {'system_entry_e2e_ttft_slo_attainment_pct': 50.}}]}}
    mapping = {'req-0': SimpleNamespace(bucket='alpaca', example_id='e0'),
               'req-1': SimpleNamespace(bucket='alpaca', example_id='e1')}
    quality = {'pro': {('qwen3-8b', 'alpaca', 'e0'): 1.}, 'flash': {('qwen3-8b', 'alpaca', 'e0'): .8}}
    summary = observed_utilities(payload, mapping, quality, {('alpaca', 'e0')}, 'pro')
    # The failed row scores zero and stays in the denominator even though its query is not in the cohort.
    assert summary['penalised_requests'] == 1 and summary['salvaged'] is True
    assert summary['utility_denominator'] == 2 and summary['observed_scored_queries'] == 1
    assert summary['ontimeutility'] == {'pro': pytest.approx(.25), 'flash': pytest.approx(.15)}
    assert summary['ttft_denominator'] == 2 and 'scored zero' in summary['quality_definition']
    clean = {'config': {'lambda_weight': .5}, 'router': {'runs': [{'utility': 'score', 'per_request': rows[:1],
             'summary': {'system_entry_e2e_ttft_slo_attainment_pct': 100.}}]}}
    unaffected = observed_utilities(clean, {'req-0': mapping['req-0']}, quality, {('alpaca', 'e0')}, 'pro')
    assert unaffected['penalised_requests'] == 0 and unaffected['salvaged'] is False
    assert unaffected['observed_scored_queries'] == 1 and unaffected['ontimeutility']['pro'] == pytest.approx(.5)
    assert 'scored zero' not in unaffected['quality_definition']
    rows[1]['system_entry_e2e_ttft_slo_met'] = True
    with pytest.raises(ValueError, match='Failed request is recorded as meeting'):
        observed_utilities(payload, mapping, quality, {('alpaca', 'e0')}, 'pro')


def test_quality_join_scores_a_failed_row_zero(tmp_path):
    from scripts.eval.augment_router_actual_accuracy import ReqMapEntry, augment_file
    point = tmp_path/'derived.json'
    rows = [{'request_id': 'req-0', 'bucket': 'alpaca', 'response_model': 'qwen3-8b', 'instance_id': 'i0', 'error': None},
            {'request_id': 'req-1', 'bucket': 'alpaca', 'response_model': None, 'instance_id': None, 'error': SHM}]
    write(point, {'config': {'prompt_source': {'holdout_prompts_per_bucket': 4000}},
                  'router': {'runs': [{'utility': 'score', 'per_request': rows}]}})
    req_map = {'req-0': ReqMapEntry('req-0', 'alpaca', 'e0'), 'req-1': ReqMapEntry('req-1', 'alpaca', 'e1')}
    quality = {('qwen3-8b', 'alpaca', 'e0'): .75}
    stats = augment_file(json_path=point, req_maps_by_holdout={4000: req_map}, quality_index=quality, dry_run=False)
    assert stats.unresolved_model == 0 and stats.missing_quality == 0 and stats.skipped_reason is None
    assert stats.failed_scored_zero == 1 and stats.updated == 1
    joined = read(point)['router']['runs'][0]['per_request']
    assert joined[0]['actual_accuracy'] == .75 and joined[1]['actual_accuracy'] == 0.
    # Rerunning the join is a no-op, and a successful row without a resolvable model is still strict.
    again = augment_file(json_path=point, req_maps_by_holdout={4000: req_map}, quality_index=quality, dry_run=False)
    assert again.failed_scored_zero == 1 and again.updated == 0 and again.unchanged == 1
    rows[0].update(response_model=None, instance_id=None)
    write(point, {'config': {'prompt_source': {'holdout_prompts_per_bucket': 4000}},
                  'router': {'runs': [{'utility': 'score', 'per_request': rows}]}})
    strict = augment_file(json_path=point, req_maps_by_holdout={4000: req_map}, quality_index=quality, dry_run=False)
    assert strict.unresolved_model == 1 and strict.failed_scored_zero == 1


def test_collation_and_status_show_salvaged_cells(tmp_path, capsys):
    from scripts.cloud.collate import _fill
    from scripts.cloud.control import status
    from scripts.cloud.worker import salvage_rule
    cell = _cell()
    setup = _pool(tmp_path, cell, _payload(errors={'req-11': SHM, 'req-12': WAIT}))
    apply(setup.point, cell, setup.pool, setup.campaign, setup.state)
    entry = read(setup.state/'completed'/'qwen-score-6.json')
    row = {'primary_judge': 'pro'}
    _fill(row, {'ontimeutility': {'pro': .4}, 'ttft_slo_attainment_pct': 50.}, entry, 'measured', 'vast')
    assert row['salvaged'] is True and row['penalised_requests'] == 2
    assert row['salvage']['cause_counts'] == {'shm_header_read': 1, 'snapshot_wait_timeout': 1}
    clean = {**entry}; clean.pop('salvage')
    other = {'primary_judge': 'pro'}
    _fill(other, {'ontimeutility': {'pro': .4}, 'ttft_slo_attainment_pct': 50.}, clean, 'measured', 'vast')
    assert other['salvaged'] is False and other['penalised_requests'] == 0 and other['salvage'] is None
    bundle = tmp_path/'bundle'; bundle.mkdir()
    write(bundle/'bundle.json', {'schema_version': 1, 'cells': [cell]})
    status(setup.state, bundle)
    reported = json.loads(capsys.readouterr().out)
    assert reported['completed'] == 1 and reported['penalised_requests'] == 2
    assert reported['salvaged']['qwen-score-6']['cause_counts'] == {'shm_header_read': 1, 'snapshot_wait_timeout': 1}
    assert salvage_rule()['max_requests'] == 5 and salvage_rule()['max_fraction_pct'] == .05
    assert salvage_rule()['authorization'] == AUTHORIZATION


BASELINE = 'RuntimeError: Failed to read consistent baseline scheduler snapshot'


def test_a_torn_baseline_snapshot_read_is_an_infrastructure_fault():
    assert cause_of(BASELINE) == 'baseline_snapshot_read'
    for other in ('RuntimeError: Baseline scheduler snapshot is not published yet',
                  'RuntimeError: Baseline scheduler snapshot version regressed; restart the client'):
        assert cause_of(other) is None
    report = classify_failures(_payload(requests=16000, errors={'req-3': BASELINE, 'req-9': BASELINE}), _cell(requests=16000))
    assert report['salvageable'] and report['cause_counts'] == {'baseline_snapshot_read': 2}


def _serving(tmp_path, cell, payload, overlay_extra=()):
    """A serving-configuration run: cells under the run output, pins in a separate qualification directory."""
    setup = _pool(tmp_path, cell, payload, name='run')
    qualify = tmp_path/'qualify'
    qualify.mkdir()
    write(setup.campaign, {'schema_version': 1, 'kind': 'serving_config', 'cells': [cell], **dict(overlay_extra)})
    for name in ('model_metrics.json',):
        (qualify/name).write_bytes((setup.pool/name).read_bytes())
    qualification = {**read(setup.pool/'qualification.json'), 'campaign_sha256': digest(setup.campaign),
                     'campaign_kind': 'serving_config', 'configuration_id': 'qwen-kv-constrained',
                     'coefficient_policy': 'refit', 'coefficients_sha256': 'c'*64, 'lambda_weights': [],
                     'data_role': 'evaluation'}
    write(qualify/'qualification.json', qualification)
    write(qualify/'release.json', {'status': 'RELEASED', 'qualification_sha256': digest(qualify/'qualification.json'),
                                   'timing_review': 'reviewed', 'load_review': 'reviewed'})
    # The run output itself holds no qualification: its worker was pointed at the directory above.
    (setup.pool/'qualification.json').unlink(); (setup.pool/'release.json').unlink()
    return SimpleNamespace(**vars(setup), qualify=qualify)


def test_a_serving_run_is_salvaged_against_its_separate_qualification(tmp_path):
    cell = _cell(requests=16000, qps=7., policy='score', cid='qwen-kv-constrained-score-7')
    payload = _payload(requests=16000, qps=7., errors={'req-11': BASELINE, 'req-12': BASELINE})
    payload['config']['score_lambda_weight'] = .05
    setup = _serving(tmp_path, cell, payload, {'score_lambda_weight': .05})
    with pytest.raises(FileNotFoundError):
        apply(setup.point, cell, setup.pool, setup.campaign, setup.state)
    result = apply(setup.point, cell, setup.pool, setup.campaign, setup.state, qualification=setup.qualify)
    assert result['action'] == 'written'
    entry = read(setup.state/'completed'/(cell['id']+'.json'))
    # Every provenance field the worker writes for a serving cell is present and taken from the qualification.
    assert entry['configuration_id'] == 'qwen-kv-constrained' and entry['coefficient_policy'] == 'refit'
    assert entry['coefficients_sha256'] == 'c'*64 and entry['data_role'] == 'evaluation'
    assert entry['qualification_sha256'] == digest(setup.qualify/'qualification.json')
    assert entry['lambda_weight'] == entry['score_routing_lambda_weight'] == .05 and entry['lambda_weights'] == []
    assert entry['evaluation_lambda_weight'] == .5
    assert entry['salvage']['cause_counts'] == {'baseline_snapshot_read': 2}
    record = read(setup.point.parent/'salvage.json')
    assert record['qualification'] == str(setup.qualify.resolve())
    audit_completed_cell(read(setup.point), cell, entry)


def test_a_serving_salvage_refuses_an_unrecorded_score_multiplier(tmp_path):
    cell = _cell(requests=16000, qps=7., policy='score', cid='qwen-kv-constrained-score-7')
    setup = _serving(tmp_path, cell, _payload(requests=16000, qps=7., errors={'req-11': BASELINE}),
                     {'score_lambda_weight': .05})
    with pytest.raises(SalvageError, match='SCORE routing multiplier'):
        apply(setup.point, cell, setup.pool, setup.campaign, setup.state, qualification=setup.qualify)
    assert not (setup.state/'completed'/(cell['id']+'.json')).exists()


def test_an_overlay_without_a_cell_list_needs_the_bundle(tmp_path):
    campaign = tmp_path/'campaign.json'
    write(campaign, {'schema_version': 1, 'kind': 'serving_config', 'policies': ['mooncake_prefill']})
    with pytest.raises(SalvageError, match='pass --bundle'):
        cell_spec(campaign, 'qwen-kv-constrained-mooncake_prefill-7')
