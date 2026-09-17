"""Evidence extraction for a completed GPU-7 single-model diagnostic (read-only, CPU only).

Inputs: the diagnostic directory written by gpu7_single_model_run.py (dense 1 s read-only
snapshots + measure/point.json with the router's per-request wait estimate and the engine's
measured TTFT) and the dispatch-mode output of replay_candidates.py for that run.

Outputs (JSON, into --output):
  gpu7-<model>-run-summary.json                 what ran, load profile, measured TTFT
  gpu7-<model>-wait-estimate-vs-measured-ttft.json
                                                per-candidate wait-estimate error against measured
                                                engine TTFT, and the oracle decomposition
                                                (target error vs everything-else simulator error)
  gpu7-<model>-dense-snapshot-token-error.json  remaining-token error of every candidate on the
                                                dense snapshots (all, and thinned to reduce the
                                                autocorrelation of repeated observations)
"""
import argparse
import json
import struct
import sys
from collections import defaultdict
from pathlib import Path

import msgspec
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_tail import metrics, fold_of, q  # noqa: E402
from replay_candidates import Estimators, HEADER  # noqa: E402


def rank(a):
    a = np.asarray(a, dtype=float); r = np.empty(len(a)); r[np.argsort(a)] = np.arange(len(a)); return r


def spearman(a, b):
    ra, rb = rank(a), rank(b)
    return float(np.corrcoef(ra, rb)[0, 1]) if len(a) > 2 else float('nan')


def wait_stats(est, ttft, slo):
    err = est - ttft; ae = np.abs(err)
    big = ttft > 1000
    ratio = est[big] / ttft[big]
    return {'n': int(len(err)), 'mae_ms': float(ae.mean()), 'median_ae_ms': q(ae, .5), 'p90_ae_ms': q(ae, .9), 'p95_ae_ms': q(ae, .95), 'p99_ae_ms': q(ae, .99),
            'bias_ms': float(err.mean()), 'median_err_ms': q(err, .5), 'mean_under_ms': float(np.clip(-err, 0, None).mean()), 'mean_over_ms': float(np.clip(err, 0, None).mean()),
            'under_by_gt_1s_pct': float((err < -1000).mean() * 100), 'over_by_gt_1s_pct': float((err > 1000).mean() * 100),
            'within_20pct_or_1s_pct': float(((ae <= 0.2 * ttft) | (ae <= 1000)).mean() * 100),
            'ratio_est_over_ttft_p10_p50_p90_ttft_gt_1s': [q(ratio, .1), q(ratio, .5), q(ratio, .9)] if big.any() else None,
            'spearman_rank_corr_with_ttft': spearman(est, ttft),
            'slo_decision_agreement_pct': float(((est <= slo) == (ttft <= slo)).mean() * 100)}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--diag', type=Path, required=True)
    ap.add_argument('--replay', type=Path, required=True)
    ap.add_argument('--online', type=Path, required=True)
    ap.add_argument('--calibration', type=Path, required=True)
    ap.add_argument('--thin-every', type=int, default=30, help='keep every k-th snapshot for the thinned token-error table')
    ap.add_argument('--output', type=Path, required=True)
    a = ap.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)

    point = json.load((a.diag / 'measure/point.json').open())['router']['runs'][0]
    per_request = [r for r in point['per_request'] if not r.get('error')]
    model = per_request[0]['response_model']
    result = json.load((a.diag / 'result.json').open())
    status = json.load((a.diag / 'status.json').open())

    online = [json.loads(l) for l in a.online.open()]
    calibration = [json.loads(l) for l in a.calibration.open()]
    for r in calibration:
        r['L'] = int(r['completion_tokens']); r['pt'] = int(r['prompt_tokens'])
    est = Estimators(online, calibration)
    repeat1 = {r['response_id']: r for r in online if r['run'] == 'repeat1'}
    lookup = {}
    for r in per_request:
        base = repeat1.get(r['response_id'])
        lookup[r['response_id']] = {'L': int(r['usage_completion_tokens']), 'p': float(r['predicted_output_tokens']), 'pt': int(r['prompt_tokens']),
                                    'fold': base['fold'] if base else fold_of(r['request_id']), 'bucket': r.get('bucket'), 'ttft_ms': r['ttft_ms'],
                                    'in_repeat1_rows': base is not None}

    # ---------------- snapshots ----------------
    snaps = []
    for path in sorted((a.diag / 'snapshots').glob('*.bin')):
        raw = path.read_bytes(); h = HEADER.unpack(raw[:HEADER.size])
        assert h[0] == b'VLLMSHM1' and len(raw) == HEADER.size + h[6]
        snaps.append((path.name, h, msgspec.msgpack.decode(raw[HEADER.size:])))
    snaps.sort(key=lambda t: t[2]['created_at'])
    t0 = snaps[0][2]['created_at']
    profile = np.array([(s['created_at'] - t0, s['num_waiting'], len(s['running_request_ids']), s['kv_cache_config'].get('kv_cache_usage', float('nan')), s['decode_reserve_tokens']) for _, _, s in snaps], dtype=float)

    ttft = np.array([r['ttft_ms'] for r in per_request], dtype=float)
    queued = np.array([r['queued_ts_s'] for r in per_request], dtype=float) - t0
    L_all = np.array([lookup[r['response_id']]['L'] for r in per_request]); pt_all = np.array([r['prompt_tokens'] for r in per_request])
    p_all = np.array([float(r['predicted_output_tokens']) for r in per_request])
    summary = {
        'model': model, 'status': status.get('state'), 'requests': len(per_request), 'qps_configured': result['qps'], 'elapsed_s': result['elapsed_s'],
        'realized_arrival_qps': float(len(per_request) / (queued.max() - queued.min())),
        'subset_source': 'gpu7_subsets.json (requests the qps8p6-repeat1 control routed to this model, in arrival order)',
        'engine': 'one canonical instance, GPU 7, CPUs 84-95, canonical serving args (see server_argv_*.json); router = canonical hard policy with one instance, so routing is fixed and the wait estimate cannot influence the measured TTFT',
        'snapshots': len(snaps), 'snapshot_interval_s': float(np.median(np.diff(profile[:, 0]))),
        'measured_ttft_ms': {k: q(ttft, p) for k, p in (('p10', .1), ('p50', .5), ('p90', .9), ('p99', .99))} | {'mean': float(ttft.mean()), 'max': float(ttft.max())},
        'ttft_slo_attainment_pct': float(np.mean([r['ttft_slo_met'] for r in per_request]) * 100),
        'load_profile': {'num_waiting': {'p50': q(profile[:, 1], .5), 'p90': q(profile[:, 1], .9), 'max': float(profile[:, 1].max()), 'pct_snapshots_with_waiting_ge_10': float((profile[:, 1] >= 10).mean() * 100)},
                         'num_running_decode': {'p50': q(profile[:, 2], .5), 'p90': q(profile[:, 2], .9), 'max': float(profile[:, 2].max())},
                         'kv_cache_usage': {'p50': q(profile[:, 3], .5), 'p90': q(profile[:, 3], .9), 'pct_snapshots_ge_0_95': float((profile[:, 3] >= .95).mean() * 100)},
                         'decode_reserve_tokens': {'min': float(profile[:, 4].min()), 'p50': q(profile[:, 4], .5), 'max': float(profile[:, 4].max())},
                         'first_snapshot_with_waiting_ge_10_s': float(profile[np.argmax(profile[:, 1] >= 10), 0]) if (profile[:, 1] >= 10).any() else None},
        'request_mix': {'prompt_tokens_p50': q(pt_all, .5), 'prompt_ge_8192_pct': float((pt_all >= 8192).mean() * 100), 'final_length_p50': q(L_all, .5),
                        'final_length_capped_8192_pct': float((L_all >= 8192).mean() * 100), 'pred_le_1_pct': float((p_all <= 1.5).mean() * 100),
                        'buckets': dict(sorted(defaultdict(int, {b: sum(1 for r in per_request if r.get('bucket') == b) for b in {r.get('bucket') for r in per_request}}).items(), key=str)),
                        'requests_matched_to_repeat1_rows_pct': float(np.mean([v['in_repeat1_rows'] for v in lookup.values()]) * 100)},
        'phases': {'ttft_p50_first_300s_ms': q(ttft[queued < 300], .5), 'ttft_p50_after_300s_ms': q(ttft[queued >= 300], .5)},
    }
    json.dump(summary, (a.output / f'gpu7-{model}-run-summary.json').open('w'), indent=1)

    # ---------------- wait estimate vs measured TTFT ----------------
    replay = json.load(a.replay.open())
    rows = [r for r in replay['rows'] if 'oracle' in r['estimates_ms']]
    names = sorted({k for r in rows for k in r['estimates_ms']})
    T = np.array([r['ttft_ms'] for r in rows]); S = np.array([r['ttft_slo_ms'] for r in rows]); R = np.array([r['router_wait_ms'] for r in rows])
    # The *_all variants only exist when the snapshot had waiting requests; otherwise they equal their base variant exactly.
    base_of = {'oracle_all': 'oracle', 'cal_prompt_q50_all': 'cal_prompt_q50'}
    E = {k: np.array([r['estimates_ms'].get(k, r['estimates_ms'].get(base_of.get(k), np.nan)) for r in rows], dtype=float) for k in names}
    assert all(np.isfinite(v).all() for v in E.values()), 'missing variant estimates'
    wait = {'model': model, 'aligned_requests': len(rows), 'total_requests': replay['total_requests'], 'dropped_rows_without_running_decode': replay['aligned_requests'] - len(rows),
            'staleness_s_p50_p90': [q([r['staleness_s'] for r in rows], .5), q([r['staleness_s'] for r in rows], .9)],
            'scope': ('Each request is simulated through the deployed native simulator from the last read-only snapshot published before its engine enqueue, with '
                      'running-decode targets replaced by each candidate; the estimate is compared with the engine-measured TTFT of that same request. The measured '
                      'TTFT is independent of the candidate (single instance, routing fixed). No pending-dispatch overlay; staleness <= 1 s.'),
            'replay_validation': {'replayed_current_vs_router_logged_mae_ms': replay['summary']['replayed_current_vs_router_logged']['mae_ms'],
                                  'replayed_current_vs_router_logged_median_abs_ms': replay['summary']['replayed_current_vs_router_logged']['median_abs_ms']},
            'all_requests': {'router_logged': wait_stats(R, T, S)} | {k: wait_stats(E[k], T, S) for k in names}}
    bins = [(0, 1000, 'ttft_lt_1s'), (1000, 10000, 'ttft_1_10s'), (10000, 60000, 'ttft_10_60s'), (60000, 1e12, 'ttft_gt_60s')]
    wait['by_measured_ttft_bin'] = {}
    for lo, hi, tag in bins:
        m = (T >= lo) & (T < hi)
        if m.sum() < 5:
            continue
        wait['by_measured_ttft_bin'][tag] = {'n': int(m.sum())} | {k: {'mae_ms': float(np.abs(E[k][m] - T[m]).mean()), 'median_err_ms': q(E[k][m] - T[m], .5), 'under_by_gt_1s_pct': float(((E[k][m] - T[m]) < -1000).mean() * 100), 'over_by_gt_1s_pct': float(((E[k][m] - T[m]) > 1000).mean() * 100)} for k in names}
    # Decomposition: measured - current = (oracle - current) [remaining-length targets] + (measured - oracle) [everything else in the simulator]
    comp = {'target_component_oracle_minus_current_ms': E['oracle'] - E['current'], 'simulator_component_measured_minus_oracle_ms': T - E['oracle'], 'total_measured_minus_current_ms': T - E['current']}
    wait['decomposition'] = {k: {'mean': float(v.mean()), 'p10': q(v, .1), 'p50': q(v, .5), 'p90': q(v, .9), 'mean_abs': float(np.abs(v).mean())} for k, v in comp.items()}
    wait['decomposition']['share_of_mean_current_under_estimate_explained_by_targets_pct'] = float(100 * comp['target_component_oracle_minus_current_ms'].mean() / comp['total_measured_minus_current_ms'].mean())
    if 'oracle_all' in E:
        # Split the simulator component further: waiting-request targets (oracle_all also gives waiting requests their true length) vs the rest.
        comp2 = {'waiting_target_component_oracle_all_minus_oracle_ms': E['oracle_all'] - E['oracle'], 'residual_measured_minus_oracle_all_ms': T - E['oracle_all']}
        wait['decomposition'].update({k: {'mean': float(v.mean()), 'p10': q(v, .1), 'p50': q(v, .5), 'p90': q(v, .9), 'mean_abs': float(np.abs(v).mean())} for k, v in comp2.items()})
    # Candidate vs oracle (pure target effect, no simulator error)
    wait['candidate_vs_oracle'] = {k: {'mae_ms': float(np.abs(E[k] - E['oracle']).mean()), 'median_err_ms': q(E[k] - E['oracle'], .5), 'p90_abs_ms': q(np.abs(E[k] - E['oracle']), .9),
                                       'mean_under_ms': float(np.clip(E['oracle'] - E[k], 0, None).mean()), 'mean_over_ms': float(np.clip(E[k] - E['oracle'], 0, None).mean())} for k in names if k != 'oracle'}
    json.dump(wait, (a.output / f'gpu7-{model}-wait-estimate-vs-measured-ttft.json').open('w'), indent=1)

    # ---------------- dense-snapshot token error ----------------
    obs = []
    for si, (name, h, s) in enumerate(snaps):
        reserve = int(s['decode_reserve_tokens'])
        for rid in s['running_request_ids']:
            r = s['requests'][rid]
            if r['num_prompt_processed_tokens'] < r['num_prompt_tokens']:
                continue
            base = lookup.get(rid)
            if base is None:
                continue
            g = int(r['num_output_processed_tokens']); Tt = int(r['num_output_target_tokens'])
            cap = min(int(r['max_tokens']), int(s['config']['max_model_len']) - int(r['num_prompt_tokens']))
            L = base['L']
            assert 0 <= g <= L <= cap, (rid, g, L, cap)
            targets = est.targets(model, g, cap, base['p'], int(r['num_prompt_tokens']), base['fold'], Tt)
            obs.append({'snap': si, 'id': rid, 'g': g, 'R': L - g, 'L': L, 'cap': cap, 'ctx': int(r['num_prompt_tokens']) + g, 'exhausted': (Tt - g) <= 1 and g < cap,
                        'est': {k: v - g for k, v in targets.items()}})
    names_t = list(obs[0]['est'])
    def table(sub):
        act = [o['R'] for o in sub]; ctx = [o['ctx'] for o in sub]
        out = {k: metrics([o['est'][k] for o in sub], act, ctx) for k in names_t}
        by_snap = defaultdict(list)
        for i, o in enumerate(sub):
            by_snap[o['snap']].append(i)
        for k in names_t:
            e = np.array([o['est'][k] for o in sub], dtype=float); ac = np.array(act, dtype=float)
            ratios = np.array([e[idx].sum() / max(ac[idx].sum(), 1) for idx in by_snap.values()])
            out[k]['backlog_ratio'] = {'median': q(ratios, .5), 'p10': q(ratios, .1), 'p90': q(ratios, .9), 'mean_abs_log': float(np.mean(np.abs(np.log(ratios))))}
        return out
    thin = [o for o in obs if o['snap'] % a.thin_every == 0]
    token = {'model': model, 'observations': len(obs), 'unique_requests': len({o['id'] for o in obs}), 'snapshots_with_observations': len({o['snap'] for o in obs}),
             'exhausted_pct': float(np.mean([o['exhausted'] for o in obs]) * 100), 'capped_final_length_pct': float(np.mean([o['L'] >= o['cap'] for o in obs]) * 100),
             'note': ('Observations are the same requests seen in consecutive 1 s snapshots, so they are strongly autocorrelated; "thinned" keeps every '
                      f'{a.thin_every}th snapshot. Candidate tables are the same held-out tables as evaluate_tail.py (cal_* calibration-only; online_* prompt folds).'),
             'all_snapshots': table(obs), 'thinned': {'observations': len(thin), 'snapshots': len({o['snap'] for o in thin})} | table(thin)}
    json.dump(token, (a.output / f'gpu7-{model}-dense-snapshot-token-error.json').open('w'), indent=1)

    print(json.dumps({'run': {k: summary[k] for k in ('requests', 'realized_arrival_qps', 'measured_ttft_ms', 'ttft_slo_attainment_pct')},
                      'wait_mae_s': {k: round(v['mae_ms'] / 1000, 2) for k, v in wait['all_requests'].items()},
                      'wait_median_err_s': {k: round(v['median_err_ms'] / 1000, 2) for k, v in wait['all_requests'].items()},
                      'decomposition': wait['decomposition'],
                      'token_p90_thinned': {k: round(v['p90_ae']) for k, v in token['thinned'].items() if isinstance(v, dict) and 'p90_ae' in v},
                      'backlog_ratio_thinned': {k: round(v['backlog_ratio']['median'], 2) for k, v in token['thinned'].items() if isinstance(v, dict) and 'backlog_ratio' in v}}, indent=1))


if __name__ == '__main__':
    main()
