"""Per-lambda table for the SCORE lambda tuning probe (scripts/cloud/score-lambda-sweep-20260917.json).

Reads the probe receipts of <state>/completed-probes, re-verifies each point against its recorded
sha256, and reports, per lambda, the metrics the campaign reports for a SCORE cell:

  * system-entry TTFT SLO attainment over every request of the probe (the producer's own summary,
    re-derived here from the rows as a check);
  * Pro OnTimeUtility, computed by scripts.cloud.collate.observed_utilities on the frozen quality
    index. The 2,000-request probe is the first 2,000 requests of the canonical 16,000-request
    sequence (the holdout pool is mixed and shuffled from the full per-bucket limit with the same
    seed and only then truncated to num_requests), so the frozen request map is restricted to the
    routed ids and passed unchanged; the observed-query cohort stays the frozen full-bundle cohort.
    observed_utilities scores a point with the multiplier that point was routed under, which would
    make four differently-routed lambdas incomparable, so the reported (primary) number fixes the
    campaign's reporting multiplier REPORTING_LAMBDA for every probe, exactly as every reportable
    cell of the campaign is scored today; the as-run value is reported beside it, labelled;
  * median and p90 system-entry end-to-end TTFT;
  * per-model routing counts;
  * predicted-versus-actual error of SCORE's own latency terms for the selected engine.

Not a reportable collation: probes carry data_role tuning_probe and never enter the completed ledger.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import statistics

# Every reportable cell of the campaign - SFS, SCORE and every baseline - is scored with the
# bundle's own multiplier, so OnTimeUtility is comparable across cells only at this fixed value.
REPORTING_LAMBDA = 5e-4


def _percentile(values, q):
    ordered = sorted(values)
    if not ordered:
        return None
    position = (len(ordered)-1) * q
    low, high = int(position), min(int(position)+1, len(ordered)-1)
    return ordered[low] + (ordered[high]-ordered[low]) * (position-low)


def wait_errors(rows):
    """SCORE's predicted latency terms for the selected engine against what the request actually saw.

    predicted_wait_ms is the selected candidate's waiting_time_ms (W_i + queueing of the backlog),
    compared with the request's observed pre-prefill wait (system entry to first token, less the
    measured prefill); predicted_total_latency_ms is W_i + s_i*Shat_i, compared with the realized
    end-to-end latency. Rows without the terms (or without a measured prefill) are skipped and counted.
    """
    wait, total, skipped = [], [], 0
    for row in rows:
        terms = row.get('score_policy_terms') or {}
        candidates = terms.get('candidates') or {}
        selected = candidates.get(terms.get('selected_instance_id'))
        if not selected or row.get('error'):
            skipped += 1
            continue
        observed_ttft, prefill = row.get('system_entry_e2e_ttft_ms'), row.get('prefill_ms')
        if selected.get('waiting_time_ms') is not None and observed_ttft is not None and prefill is not None:
            wait.append(float(selected['waiting_time_ms']) - (float(observed_ttft) - float(prefill)))
        if selected.get('predicted_total_latency_ms') is not None and row.get('latency_ms') is not None:
            total.append(float(selected['predicted_total_latency_ms']) - float(row['latency_ms']))
    def stats(values, name):
        if not values:
            return {f'{name}_rows': 0}
        return {f'{name}_rows': len(values), f'mean_{name}_error_ms': statistics.mean(values),
                f'mean_absolute_{name}_error_ms': statistics.mean(abs(v) for v in values),
                f'median_{name}_error_ms': statistics.median(values)}
    return {'skipped_rows': skipped, **stats(wait, 'wait'), **stats(total, 'total_latency')}


def probe_row(record, bundle, mapping, quality, common):
    from scripts.cloud.common import digest, read, source_digest
    from scripts.cloud.collate import observed_utilities
    from sfs_core.shared.model_label_helpers import normalize_model_label
    point = Path(record['point'])
    if digest(point) != record['point_sha256']:
        raise ValueError(f'Probe point changed since it was written: {point}')
    payload = read(point)
    run = payload['router']['runs'][0]
    rows = run['per_request']
    cell = record['cell']
    if len(rows) != cell['requests'] or payload['config']['lambda_weight'] != cell['lambda_weight']:
        raise ValueError(f'Probe point disagrees with its receipt: {cell["id"]}')
    ttft = [float(r['system_entry_e2e_ttft_ms']) for r in rows]
    met = sum(1 for r in rows if r.get('system_entry_e2e_ttft_slo_met'))
    subset = {rid: entry for rid, entry in mapping.items() if rid in {r['request_id'] for r in rows}}
    reported = dict(payload, config=dict(payload['config'], lambda_weight=REPORTING_LAMBDA))
    observed = observed_utilities(reported, subset, quality, common, 'pro')
    as_run = observed_utilities(payload, subset, quality, common, 'pro')
    return {'cell': cell['id'], 'lambda_weight': cell['lambda_weight'], 'requests': len(rows),
            'reporting_lambda_weight': REPORTING_LAMBDA,
            'pro_ontimeutility_at_as_run_lambda': as_run['ontimeutility']['pro'],
            'data_role': record.get('data_role'), 'point': str(point), 'point_sha256': record['point_sha256'],
            'ttft_slo_attainment_pct': run['summary']['system_entry_e2e_ttft_slo_attainment_pct'],
            'ttft_slo_attainment_pct_recomputed': 100.0*met/len(rows),
            'pro_ontimeutility': observed['ontimeutility']['pro'],
            'flash_ontimeutility': observed['ontimeutility']['flash'],
            'observed_scored_queries': observed['observed_scored_queries'],
            'utility_denominator': observed['utility_denominator'],
            'median_ttft_ms': statistics.median(ttft), 'p90_ttft_ms': _percentile(ttft, .9),
            'mean_ttft_ms': statistics.mean(ttft),
            'route_counts': run['summary'].get('instance_route_counts'),
            'model_counts': dict(Counter(normalize_model_label(r['response_model']) for r in rows if r.get('response_model'))),
            'mean_actual_cost': statistics.mean(float(r['actual_cost']) for r in rows),
            'realized_qps': len(rows)/float(run['summary']['elapsed_s']) if run['summary'].get('elapsed_s') else None,
            'elapsed_s': run['summary'].get('elapsed_s'),
            'mean_predicted_accuracy': (run['summary'].get('predicted_accuracy') or {}).get('mean'),
            'campaign_sha256': record.get('campaign_sha256'), 'qualification_sha256': record.get('qualification_sha256'),
            'source_digest': source_digest(record['source_sha256']), 'hardware_host': (record.get('hardware') or {}).get('host'),
            'remaining_length_rule': record.get('remaining_length_rule'),
            'wait_error': wait_errors(rows)}


def report(bundle, state, output):
    from scripts.cloud.common import digest, read
    from scripts.cloud.collate import observed_qwen_queries
    from scripts.eval.augment_router_actual_accuracy import load_req_map, load_quality_index
    bundle, state = Path(bundle), Path(state)
    mapping = load_req_map(bundle/'qwen/request_map.csv')
    quality = {'pro': load_quality_index(bundle/'qwen/scores'), 'flash': load_quality_index(bundle/'qwen/scores_flash')}
    models = read(bundle/'bundle.json')['families']['qwen']['models']
    common = observed_qwen_queries(bundle, models, quality)
    rows = [probe_row(read(path), bundle, mapping, quality, common)
            for path in sorted((state/'completed-probes').glob('probe-qwen-score-7-lambda*.json'))]
    rows.sort(key=lambda r: r['lambda_weight'])
    result = {'probe': 'scripts/cloud/score-lambda-sweep-20260917.json', 'data_role': 'tuning_probe',
              'bundle_sha256': digest(bundle/'bundle.json'), 'observed_query_cohort': len(common),
              'reporting_lambda_weight': REPORTING_LAMBDA,
              'utility_definition': 'pro_ontimeutility is the mean over every routed request of (frozen Pro judge quality '
                                    'of the answering model on that query - 5e-4 * actual_cost), gated by the system-entry '
                                    'TTFT SLO, with requests outside the frozen observed-judge cohort excluded: the '
                                    'campaign definition, at the campaign reporting multiplier for every probe so the four '
                                    'lambdas are comparable. pro_ontimeutility_at_as_run_lambda repeats it with each '
                                    'probe\'s own routing multiplier and is reported only for completeness',
              'rows': rows}
    if output:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        Path(output).write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle', required=True)
    p.add_argument('--state', required=True)
    p.add_argument('--output')
    a = p.parse_args()
    report(a.bundle, a.state, a.output)
