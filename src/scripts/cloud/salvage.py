"""Strictly bounded salvage of a completed cell whose only failures are infrastructure faults.

A cell that generated and completed every one of its requests but recorded a handful of router
*infrastructure* faults is admitted from the existing data instead of being rerun.  Admission is
deliberately narrow, and the bounds below are the whole of the rule:

* only the two native snapshot faults on ``SALVAGEABLE_CAUSES`` may be salvaged.  Anything else --
  HTTP/connection errors, engine errors, missing responses, cancellations, timeouts against a
  model -- is policy- or load-induced evidence and is never salvageable;
* at most ``salvage_cap(requests) = min(5, floor(0.05% of the cell's request count))`` requests may
  have failed, and *every* failed request must match the allowlist (mixed causes are refused);
* every other ``scripts.cloud.worker.audit_cell`` condition must pass unchanged -- one run, cell
  identity, the complete request-identity set, the snapshot-read-fault gate, arrival telemetry
  within 10% of the requested rate, and the pool's source/bundle/qualification pins;
* the failed requests are penalised, never dropped.  They stay in the denominator of every SLO
  metric, count as an SLO miss for every one of them, and contribute zero utility.

The producer in ``scripts.runs.experiments`` already gives failed rows exactly that treatment --
every ``*_slo_met`` flag is gated on ``error is None`` and every ``*_slo_attainment_pct`` divides by
the full ``total_requests``.  ``assert_penalty`` re-derives each attainment from the per-request rows
rather than assuming it, and refuses the cell if the recorded numbers disagree.

This module never changes the worker's live behaviour: a cell that fails *during* a run still stops
the pool, as today.  Salvage is an explicit, after-the-fact, user-authorised step run by this CLI.
"""
import argparse
import json
import math
from pathlib import Path
import time

from scripts.cloud.common import digest, read, write, validate_bundle, locks
from scripts.cloud.worker import audit_cell, cell_staleness


# Recorded per-request `error` strings that this rule accepts, by error class.  Each entry is the
# tuple of substrings that must all appear in the recorded string for the row to match that class.
SALVAGEABLE_CAUSES = {
    # csrc/scheduler_sim/bindings.cpp raises the native wait timeout wrapped in the simulation error.
    'snapshot_wait_timeout': ('Scheduler simulation failed', 'Timed out waiting for the parsed scheduler snapshot'),
    # sfs_core/routing/snapshot_shm_client.py fails a torn/short header read of the snapshot SHM.
    'shm_header_read': ('Failed to read a consistent scheduler snapshot header from SHM',),
    # sfs_core/routing/snapshot_shm_client.py: a baseline policy's seqlock read of the snapshot SHM
    # exhausted its 32 retries while the engine was rewriting the payload (a torn read), before dispatch.
    'baseline_snapshot_read': ('Failed to read consistent baseline scheduler snapshot',),
}
SERVING_PROVENANCE = ('configuration_id', 'coefficient_policy', 'coefficients_sha256')
MAX_SALVAGEABLE_REQUESTS = 5
MAX_SALVAGEABLE_FRACTION = 0.0005  # 0.05 percent of the cell's request count
AUTHORIZATION = 'user authorised 2026-09-17'
PENALTY = ('Failed requests stay in the denominator of every SLO metric, count as an SLO miss for every one of them, '
           'and contribute zero utility; they are never dropped from any denominator.')
# Per-request attainment flag -> the run summary percentage it feeds.
SLO_METRICS = {'slo_met': 'slo_attainment_pct', 'queue_slo_met': 'queue_slo_attainment_pct',
               'ttft_slo_met': 'ttft_slo_attainment_pct', 'e2e_ttft_slo_met': 'e2e_ttft_slo_attainment_pct',
               'system_entry_e2e_ttft_slo_met': 'system_entry_e2e_ttft_slo_attainment_pct'}
# Telemetry whose per-run missing count must be exactly the salvaged rows.
COVERAGE_FIELDS = {'queue_delay_ms': 'queue_slo_missing_count', 'ttft_ms': 'ttft_slo_missing_count',
                   'e2e_ttft_ms': 'e2e_ttft_slo_missing_count',
                   'system_entry_e2e_ttft_ms': 'system_entry_e2e_ttft_slo_missing_count'}


class SalvageError(ValueError):
    """A cell is not salvageable, or a recorded salvage no longer holds."""


def salvage_cap(requests):
    """At most 5 failed requests, and at most 0.05% of the cell's requests, whichever is smaller."""
    return int(min(MAX_SALVAGEABLE_REQUESTS, math.floor(int(requests) * MAX_SALVAGEABLE_FRACTION)))


def cause_of(error):
    """The salvageable error class of a recorded per-request `error` string, or None."""
    if not isinstance(error, str) or not error:
        return None
    for name, needles in SALVAGEABLE_CAUSES.items():
        if all(needle in error for needle in needles):
            return name
    return None


def failed_rows(run):
    """Rows the producer did not complete: an error string, or no response at all."""
    return [row for row in run['per_request'] if row.get('error') or not row.get('response_id')]


def classify_failures(payload, cell):
    """Read-only salvage classification of a completed point. Never raises on a merely unsalvageable cell.

    Returns the failed request ids by error class, the ids this rule cannot cover, the cap that
    applies to this cell, and whether the cause list and the cap alone would admit it.  This is a
    report, not an admission: `audit_salvaged_cell` still has to pass every other audit condition.
    """
    run = payload['router']['runs'][0]
    rows, cap = failed_rows(run), salvage_cap(cell['requests'])
    classified = [{'request_id': row.get('request_id'), 'error': row.get('error'), 'cause': cause_of(row.get('error'))}
                  for row in rows]
    counts = {}
    for entry in classified:
        if entry['cause']:
            counts[entry['cause']] = counts.get(entry['cause'], 0) + 1
    unsalvageable = [entry for entry in classified if not entry['cause']]
    return {'failed_requests': len(classified), 'cause_counts': dict(sorted(counts.items())),
            'request_ids': sorted(entry['request_id'] for entry in classified),
            'unsalvageable': unsalvageable, 'cap': cap, 'cell_requests': cell['requests'],
            'within_allowlist': not unsalvageable, 'within_cap': len(classified) <= cap,
            'salvageable': bool(classified) and not unsalvageable and len(classified) <= cap}


def assert_penalty(run, failed):
    """Check -- not assume -- that the recorded run already penalises the failed rows, and price the penalty.

    Every SLO attainment is re-derived from the per-request rows: the failed rows must miss every
    SLO and must remain in the denominator.  Returns the recorded percentages alongside what each
    would have been had the failed rows been dropped from their denominators instead.
    """
    rows, summary = run['per_request'], run['summary']
    ids = {row['request_id'] for row in failed}
    if summary.get('total_requests') != len(rows) or summary.get('failed_requests') != len(failed) \
            or summary.get('succeeded_requests') != len(rows) - len(failed):
        raise SalvageError('Run summary does not account for every request in the denominator')
    for row in failed:
        met = [field for field in SLO_METRICS if row.get(field)]
        if met:
            raise SalvageError(f'Failed request {row["request_id"]} is recorded as meeting {sorted(met)}')
        if row.get('actual_accuracy') not in (None, 0, 0.0):
            raise SalvageError(f'Failed request {row["request_id"]} carries a nonzero quality score')
    recorded, without = {}, {}
    for field, metric in SLO_METRICS.items():
        hits = sum(bool(row.get(field)) for row in rows)
        if any(row.get(field) for row in rows if row['request_id'] in ids):
            raise SalvageError(f'A failed request is counted as a {metric} hit')
        expected = 100.0 * hits / len(rows)
        if not math.isclose(float(summary[metric]), expected, rel_tol=1e-9, abs_tol=1e-9):
            raise SalvageError(f'{metric} is not the penalised attainment over the full denominator')
        recorded[metric] = float(summary[metric])
        survivors = len(rows) - len(failed)
        without[metric] = (100.0 * hits / survivors) if survivors else 0.0
    for field, metric in COVERAGE_FIELDS.items():
        missing = {row['request_id'] for row in rows
                   if not isinstance(row.get(field), (int, float)) or isinstance(row.get(field), bool)
                   or not math.isfinite(row[field]) or row[field] < 0}
        if missing != ids or summary.get(metric) != len(failed):
            raise SalvageError(f'{metric} does not cover exactly the salvaged requests')
    return {'denominator': len(rows), 'penalised_requests': len(failed),
            'slo_attainment_pct_recorded': recorded, 'slo_attainment_pct_without_penalty': without}


def audit_salvaged_cell(payload, cell):
    """`audit_cell` for a cell whose only failures are salvageable, or a loud refusal.

    Returns ``(audit, salvage)``: the audit fields `audit_cell` itself would return, and the
    provenance block the completed-ledger entry carries.

    Every condition of `scripts.cloud.worker.audit_cell` is enforced here, and the per-row checks
    are delegated to the producer's own auditors so they cannot drift.  The conditions that are
    restated rather than delegated -- one run, cell identity, the request-identity set, the summary
    counts and the arrival rate -- must be kept in step with `audit_cell` whenever it gains a new
    one; a salvaged cell is never held to a weaker standard than a clean one.
    """
    if len(payload['router']['runs']) != 1:
        raise SalvageError('Expected exactly one policy per checkpoint')
    run = payload['router']['runs'][0]
    if run.get('utility') != cell['policy'] or payload['config']['request_rate_qps'] != cell['qps']:
        raise SalvageError('Cell identity mismatch')
    if run.get('utility') == 'vllm_sr_latency':
        # The selector's warm-up/selection trace has one entry per routed request; a missing
        # selection cannot be audited after the fact, so such a cell is rerun, never salvaged.
        raise SalvageError('Salvage does not cover the latency selector trace; rerun the cell')
    rows = run['per_request']
    report = classify_failures(payload, cell)
    if not report['failed_requests']:
        raise SalvageError('Cell has no failed requests; admit it through audit_cell')
    if not report['within_allowlist']:
        causes = sorted({entry['error'] for entry in report['unsalvageable']})
        raise SalvageError(f'Failure causes outside the infrastructure allowlist: {causes}')
    if not report['within_cap']:
        raise SalvageError(f'{report["failed_requests"]} failed requests exceed the salvage cap of {report["cap"]}')
    failed = failed_rows(run)
    if any(not row.get('error') for row in failed):
        raise SalvageError('A request is missing its response without a recorded error')
    if {row['request_id'] for row in rows} != {f'req-{i}' for i in range(cell['requests'])}:
        raise SalvageError('Missing or duplicate evaluation request identities')
    # audit_cell's snapshot-read-fault gate applies here unchanged: once the router degrades
    # transient faults per candidate, a decision that lost every candidate is a rerun, never a
    # salvage. The preserved cells this rule was written for predate those counters.
    faults = run['summary'].get('snapshot_read_faults')
    if isinstance(faults, dict) and (faults.get('totals') or {}).get('requests_without_estimate'):
        raise SalvageError('Routing decisions lost every candidate to snapshot read faults')
    penalty = assert_penalty(run, failed)
    # Every strict per-row check still applies to the successful rows, through the producer's own
    # auditors, on a projection of the run with exactly the salvaged rows removed.
    from scripts.runs.measured_audit import require_complete_ttft
    from scripts.runs.ministral3_methodology_stage import audit_run
    survivors = [row for row in rows if row['request_id'] not in {r['request_id'] for r in failed}]
    clean = {**run, 'per_request': survivors,
             'summary': {**run['summary'], 'system_entry_e2e_ttft_slo_missing_count': 0}}
    audit_run(clean, len(rows) - len(failed))
    require_complete_ttft(clean)
    arrivals = [row.get('system_entry_offset_s') for row in rows]
    if any(not isinstance(t, (float, int)) or isinstance(t, bool) or not math.isfinite(t) or t < 0 for t in arrivals):
        raise SalvageError('Missing arrival telemetry')
    realized = (len(rows) - 1) / (max(arrivals) - min(arrivals))
    if abs(realized / cell['qps'] - 1) > .1:
        raise SalvageError('Arrival generator missed the requested rate by more than 10%')
    audit = {'status': 'PASS_CELL', 'requests': len(rows), 'realized_qps': realized}
    salvage = {'status': 'SALVAGED', 'cause_counts': report['cause_counts'], 'request_ids': report['request_ids'],
               'failed_requests': report['failed_requests'],
               'causes_by_request': {entry['request_id']: {'cause': entry['cause'], 'error': entry['error']}
                                     for entry in sorted(
                                         [{'request_id': r.get('request_id'), 'error': r.get('error'),
                                           'cause': cause_of(r.get('error'))} for r in failed],
                                         key=lambda e: e['request_id'])},
               'cap': {'cap_used': report['cap'], 'max_requests': MAX_SALVAGEABLE_REQUESTS,
                       'max_fraction_pct': MAX_SALVAGEABLE_FRACTION * 100, 'cell_requests': cell['requests']},
               'allowlist': {name: list(needles) for name, needles in SALVAGEABLE_CAUSES.items()},
               'penalty': PENALTY, 'penalty_evidence': penalty,
               'authorized_by': 'user', 'authorization': AUTHORIZATION,
               'rule': 'Infrastructure-fault salvage: completed cell, allowlisted causes only, within the cap, '
                       'failed requests penalised in place of a rerun'}
    return audit, salvage


def audit_completed_cell(payload, cell, record=None):
    """Re-audit a completed cell the way its ledger entry claims it was admitted.

    A clean entry is re-audited by `audit_cell` unchanged; an entry carrying a `salvage` block is
    re-audited by the salvage rule, and its recorded cause counts, request ids and cap must still
    match the point.  Used by the collation so a salvaged cell can never be laundered into a clean
    one, nor a clean one silently acquire a salvage block.
    """
    if not (record or {}).get('salvage'):
        return audit_cell(payload, cell), None
    audit, salvage = audit_salvaged_cell(payload, cell)
    recorded = record['salvage']
    for field in ('cause_counts', 'request_ids', 'failed_requests', 'cap'):
        if recorded.get(field) != salvage[field]:
            raise SalvageError(f'Recorded salvage {field} no longer matches the point: {cell["id"]}')
    if recorded.get('authorization') != AUTHORIZATION:
        raise SalvageError(f'Salvaged cell carries an unrecognised authorisation: {cell["id"]}')
    return audit, salvage


def salvage_summary(record):
    """Compact, reportable salvage marker of a completed-ledger entry, or None for a clean cell."""
    block = (record or {}).get('salvage')
    if not block:
        return None
    return {'status': block.get('status'), 'penalised_requests': block.get('failed_requests'),
            'cause_counts': block.get('cause_counts'), 'request_ids': block.get('request_ids'),
            'authorization': block.get('authorization')}


def _pins(pool, cell, campaign, bundle=None):
    """Validate the pool's source/bundle/qualification pins and return the ledger fields they fix.

    `pool` is the directory holding qualification.json and release.json: the pool output itself, or
    for a serving-configuration run the separate qualification directory its worker was given.
    """
    pool = Path(pool)
    qualification, release = read(pool/'qualification.json'), read(pool/'release.json')
    qualification_sha = digest(pool/'qualification.json')
    if release.get('status') != 'RELEASED' or release.get('qualification_sha256') != qualification_sha \
            or not release.get('timing_review') or not release.get('load_review'):
        raise SalvageError('Missing/stale destination qualification and reviewed release')
    if qualification['family'] != cell['family'] or qualification['variant'] != cell['variant']:
        raise SalvageError('Cell does not belong to this pool family/variant')
    if qualification.get('campaign_sha256') != digest(campaign):
        raise SalvageError('Cell completed under a different campaign overlay')
    for name, expected in qualification['files'].items():
        if digest(pool/name) != expected:
            raise SalvageError(f'Qualification evidence changed: {name}')
    if bundle and digest(Path(bundle)/'bundle.json') != qualification['bundle_sha256']:
        raise SalvageError('Bundle does not match the pool qualification')
    return {'source_sha256': qualification['source_sha256'], 'bundle_sha256': qualification['bundle_sha256'],
            'qualification_sha256': qualification_sha, 'hardware': qualification['hardware'],
            'campaign_sha256': qualification.get('campaign_sha256'), 'campaign_kind': qualification.get('campaign_kind'),
            'remaining_length_rule': qualification['remaining_length_rule'],
            'remaining_length': qualification['remaining_length'],
            'snapshot_staleness_levels_ms': qualification['snapshot_staleness_levels_ms'],
            # Provenance a newer worker writes into every ledger entry; copied only when the pool records it.
            'serving': {key: qualification[key] for key in SERVING_PROVENANCE if key in qualification},
            'lambda_weights': qualification.get('lambda_weights'), 'data_role': qualification.get('data_role')}


def cell_spec(campaign, cell_id, bundle=None):
    """The cell's frozen spec, from the overlay (cross-checked against the overlaid manifest when a bundle is given).

    A serving-configuration overlay derives its cells from `policies` and lists none, so its spec
    can only come from the overlaid bundle manifest.
    """
    overlay = read(campaign)
    if 'cells' not in overlay:
        if not bundle:
            raise SalvageError('This overlay derives its cells from the bundle; pass --bundle')
        from scripts.cloud.campaigns import apply_any_campaign
        overlay = {'cells': apply_any_campaign(validate_bundle(bundle), overlay, inspect=True)['cells']}
    cells = {c['id']: c for c in overlay['cells']}
    if cell_id not in cells:
        raise SalvageError(f'Overlay has no cell {cell_id}')
    cell = cells[cell_id]
    if bundle:
        from scripts.cloud.campaigns import apply_any_campaign
        manifest = apply_any_campaign(validate_bundle(bundle), read(campaign), inspect=True)
        active = {c['id']: c for c in manifest['cells']}
        if active.get(cell_id) != cell:
            raise SalvageError(f'Overlay cell disagrees with the overlaid bundle manifest: {cell_id}')
    return cell


def _lambda_fields(payload, cell, campaign, bundle, pins):
    """The lambda provenance the worker writes for a cell, checked against what the router recorded."""
    from scripts.cloud.worker import routing_lambda
    if bundle:
        from scripts.cloud.campaigns import apply_any_campaign
        manifest = apply_any_campaign(validate_bundle(bundle), read(campaign), inspect=True)
    else:
        manifest = read(campaign)
    weight = routing_lambda(manifest, cell)
    if weight is not None and float(payload['config'].get('score_lambda_weight', float('nan'))) != weight:
        raise SalvageError('Router did not record the cell SCORE routing multiplier')
    return {'lambda_weight': weight, 'lambda_weights': pins['lambda_weights'], 'score_routing_lambda_weight': weight,
            'evaluation_lambda_weight': float(payload['config']['lambda_weight']),
            'data_role': cell.get('data_role', pins['data_role'])}


def build(point, cell, pool, campaign, bundle=None, qualification=None):
    """The completed-ledger entry and the salvage record for one preserved cell, or a loud refusal.

    `qualification` is the separate qualification directory of a serving-configuration run, whose
    worker writes cells under its own output (`pool`) but takes its pins from that directory.
    """
    point = Path(point).resolve()
    pool = Path(pool).resolve()
    if not point.is_file() or pool/'cells' not in point.parents:
        raise SalvageError('Point must be a file inside this pool\'s cells directory')
    payload = read(point)
    pins = _pins(Path(qualification).resolve() if qualification else pool, cell, campaign, bundle)
    audit, salvage = audit_salvaged_cell(payload, cell)
    run = payload['router']['runs'][0]
    staleness = cell_staleness(cell)
    if run.get('remaining_length_rule', 'current') != pins['remaining_length_rule']:
        raise SalvageError('Router did not record the pool remaining-length rule')
    if float(run.get('snapshot_staleness_ms', 0) or 0) != staleness:
        raise SalvageError('Router did not record the cell snapshot staleness')
    entry = {**audit, 'cell': cell, 'point': str(point), 'point_sha256': digest(point),
             'source_sha256': pins['source_sha256'], 'bundle_sha256': pins['bundle_sha256'],
             'qualification_sha256': pins['qualification_sha256'], 'hardware': pins['hardware'],
             'campaign_sha256': pins['campaign_sha256'], 'campaign_kind': pins['campaign_kind'],
             'remaining_length_rule': pins['remaining_length_rule'], 'remaining_length': pins['remaining_length'],
             'snapshot_staleness_ms': staleness, 'snapshot_staleness_levels_ms': pins['snapshot_staleness_levels_ms'],
             **pins['serving'], 'salvage': salvage}
    if pins['lambda_weights'] is not None:
        entry.update(_lambda_fields(payload, cell, campaign, bundle, pins))
    # The salvage record is the completed-ledger entry itself plus where it came from: the pool
    # never wrote an audit.json for this cell, so the collation discovers it through this file.
    record = {**entry, 'record': 'salvage', 'pool': str(pool), 'campaign': str(Path(campaign).resolve()),
              'qualification': str(Path(qualification).resolve()) if qualification else None,
              'ledger_entry': cell['id'] + '.json'}
    return entry, record


def _stable(entry):
    """An entry without the fields a rerun of the tool legitimately refreshes."""
    return {k: v for k, v in entry.items() if k != 'recorded_at'}


def apply(point, cell, pool, campaign, state, bundle=None, dry_run=False, qualification=None):
    """Validate and write the ledger entry plus the salvage record. Idempotent; refuses on any conflict."""
    entry, record = build(point, cell, pool, campaign, bundle, qualification)
    ledger, sidecar = Path(state)/'completed'/(cell['id']+'.json'), Path(point).resolve().parent/'salvage.json'
    with locks(Path(state)/'cell-locks', [cell['id']]):
        if ledger.exists():
            previous = read(ledger)
            if not previous.get('salvage'):
                raise SalvageError(f'Completed cell {cell["id"]} is already admitted without salvage')
            if _stable(previous) != _stable(entry):
                raise SalvageError(f'Completed cell {cell["id"]} disagrees with this salvage; review before overwriting')
            entry, action = previous, 'unchanged'
            if sidecar.exists() and _stable(read(sidecar)) == _stable(record):
                record = read(sidecar)
            elif not dry_run:
                record = {**record, 'recorded_at': previous.get('recorded_at', time.time())}
                write(sidecar, record)
        else:
            action, stamp = 'written', time.time()
            entry, record = {**entry, 'recorded_at': stamp}, {**record, 'recorded_at': stamp}
            if not dry_run:
                write(sidecar, record)
                write(ledger, entry)
    return {'cell': cell['id'], 'action': 'dry_run' if dry_run else action, 'ledger': str(ledger),
            'salvage_record': str(sidecar), 'salvage': salvage_summary(entry),
            'penalty_evidence': entry['salvage']['penalty_evidence'], 'realized_qps': entry['realized_qps']}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--point', required=True, help='point.json of the preserved cell')
    p.add_argument('--cell', required=True, help='Cell id, whose spec is taken from the overlay')
    p.add_argument('--campaign', required=True, help='Campaign overlay holding the cell spec')
    p.add_argument('--pool', required=True, help='Pool output directory (source/bundle/qualification pins)')
    p.add_argument('--state', required=True, help='State directory holding completed/')
    p.add_argument('--bundle', help='Frozen artifact bundle, cross-checked against the pool pins when given')
    p.add_argument('--qualification', help='Separate qualification directory of a serving-configuration run '
                                           '(holds qualification.json and release.json); defaults to --pool')
    p.add_argument('--dry-run', action='store_true', help='Validate and report without writing')
    a = p.parse_args()
    cell = cell_spec(a.campaign, a.cell, a.bundle)
    print(json.dumps(apply(a.point, cell, a.pool, a.campaign, a.state, a.bundle, a.dry_run, a.qualification), indent=2))


if __name__ == '__main__':
    main()
