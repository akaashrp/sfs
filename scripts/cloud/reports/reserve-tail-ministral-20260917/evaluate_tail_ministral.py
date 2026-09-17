"""Ministral 3 driver for the remaining-decode target tail analysis (controller, CPU only).

Mirrors scripts/cloud/reports/reserve-tail-20260916/evaluate_tail.py for the Ministral family and
imports its table construction (SurvivalTable, Backoff, build_tables), bins (prompt_bin, pred_bin),
folds (fold_of) and metrics. No Ministral engine snapshots exist, so the evaluation uses:

  synthetic held-out   every online request observed at 8 evenly spaced generation points,
                       token-weighted to mimic a snapshot population (evaluate_tail.py's construction);
  pseudo-snapshots     the real running set of each (run, model) reconstructed from per-request
                       timestamps every 5 s, with generated-so-far tokens interpolated linearly over the
                       request's measured decode time (measured composition, approximated progress).

Inputs: the compact extracts of extract_inputs_ministral.py. Outputs JSON evidence into --output.
"""
import argparse
import importlib.util
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location('evaluate_tail', HERE.parent / 'reserve-tail-20260916' / 'evaluate_tail.py')
et = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(et)

MODELS = ('ministral3-3b', 'ministral3-8b', 'ministral3-14b')
BUCKETS = ('alpaca', 'govreport-summarization', 'hotpot_qa', 'writingprompts')
CAP = 8192
MAX_MODEL_LEN = 131072
POINTS = 8
FLOOR_TOKENS = 32          # scheduler._decode_floor_tokens at kv_usage <= 0.8
RESERVE_QUANTILES = (0.5, 0.65, 0.75, 0.85)   # cold / warm-low / medium / high congestion
prompt_bin, pred_bin, fold_of, q = et.prompt_bin, et.pred_bin, et.fold_of, et.q

# Memoize survival estimates: the estimate is a step function of generated-so-far, so at most
# cap+1 distinct lookups per (table, quantile).
_orig_estimate = et.SurvivalTable.estimate


def _cached_estimate(self, generated, quantile):
    cache = self.__dict__.setdefault('_cache', {})
    key = (int(generated), quantile)
    if key not in cache:
        cache[key] = _orig_estimate(self, generated, quantile)
    return cache[key]


et.SurvivalTable.estimate = _cached_estimate


def wmetrics(est, act, w, ctx=None):
    """Weighted analogue of evaluate_tail.metrics (weights = tokens, i.e. snapshot population)."""
    est = np.asarray(est, float); act = np.asarray(act, float); w = np.asarray(w, float)
    err = est - act; ae = np.abs(err)
    order = np.argsort(ae); cw = np.cumsum(w[order]) / w.sum()

    def wq(p):
        return float(ae[order][min(int(np.searchsorted(cw, p)), len(ae) - 1)])
    out = {'n': int(len(err)), 'mae': float((ae * w).sum() / w.sum()), 'median_ae': wq(.5), 'p90_ae': wq(.9), 'p95_ae': wq(.95),
           'p99_ae': wq(.99), 'max_ae': float(ae.max()) if len(ae) else float('nan'),
           'mean_under': float((np.clip(-err, 0, None) * w).sum() / w.sum()), 'mean_over': float((np.clip(err, 0, None) * w).sum() / w.sum()),
           'coverage_pct': float(((err >= 0) * w).sum() / w.sum() * 100), 'bias': float((err * w).sum() / w.sum()),
           'sum_est_over_sum_actual': float((est * w).sum() / max((act * w).sum(), 1))}
    if ctx is not None:
        ctx = np.asarray(ctx, float)
        out['context_weighted_under'] = float((np.clip(-err, 0, None) * ctx * w).sum() / (ctx * w).sum())
    return out


def engine_residual_quantile(sorted_residuals, quantile):
    """scheduler._decode_residual_quantile: ceil(q*n)-1 index on the sorted positive residuals."""
    n = len(sorted_residuals)
    if n < 200:
        return 0.0
    idx = min(max(int(math.ceil(quantile * n)) - 1, 0), n - 1)
    return float(sorted_residuals[idx])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--qwen-evidence', type=Path, default=HERE.parent / 'reserve-tail-20260916' / 'evidence')
    ap.add_argument('--qwen-tables', type=Path, default=HERE.parent.parent.parent.parent / '.scratch' / 'reserve-tail-20260916' / 'tables')
    ap.add_argument('--snapshot-interval-s', type=float, default=5.0)
    a = ap.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)

    online = [json.loads(l) for l in (a.data / 'online_rows.jsonl').open()]
    calibration_all = [json.loads(l) for l in (a.data / 'calibration_outputs.jsonl').open()]
    inventory = json.load((a.data / 'inventory.json').open())
    for r in online:
        r['L'] = int(r['usage_completion_tokens']); r['pt'] = int(r['prompt_tokens'])
        r['p'] = None if r['predicted_output_tokens'] is None else float(r['predicted_output_tokens'])
        r['fold'] = fold_of(r['example_id'])
    for r in calibration_all:
        r['L'] = int(r['completion_tokens']); r['pt'] = int(r['prompt_tokens'])
    calibration = [r for r in calibration_all if r['source'] == 'calibration']
    holdout2 = [r for r in calibration_all if r['source'] == 'holdout2']
    eval_rows = [r for r in online if r['p'] is not None]
    print('online rows', len(online), 'with prediction', len(eval_rows), 'calibration', len(calibration), 'holdout2', len(holdout2))

    # ---------------- 0. Inventory additions: length reproducibility, cap rates ----------------
    inv = {'online_rows_total': len(online), 'online_rows_with_prediction': len(eval_rows),
           'online_rows_with_router_prediction': sum(1 for r in eval_rows if r.get('p_source') == 'router'),
           'online_rows_with_imputed_prediction': sum(1 for r in eval_rows if r.get('p_source') == 'imputed'),
           'prediction_handling': inventory.get('prediction'),
           'calibration_rows': len(calibration), 'holdout2_rows': len(holdout2), 'checks': inventory['checks'],
           'cells': [{k: c[k] for k in ('host', 'cell', 'requests', 'failed', 'with_sfs_prediction', 'by_model')} for c in inventory['cells']],
           'by_model': {}, 'length_reproducibility': {}}
    by_pair = defaultdict(list)
    for r in online:
        by_pair[(r['request_id'], r['model'])].append((r['host'], r['L']))
    for m in MODELS:
        rows = [r for r in online if r['model'] == m]
        ev = [r for r in eval_rows if r['model'] == m]
        cal = [r for r in calibration if r['model'] == m]
        h2 = [r for r in holdout2 if r['model'] == m]
        inv['by_model'][m] = {
            'online_rows': len(rows), 'online_distinct_prompts': len({r['example_id'] for r in rows}),
            'online_rows_with_prediction': len(ev), 'online_prompts_with_prediction': len({r['example_id'] for r in ev}),
            'online_capped_pct': 100 * np.mean([r['L'] >= CAP for r in rows]), 'online_near_cap_ge_8181_pct': 100 * np.mean([r['L'] >= 8181 for r in rows]),
            'online_zero_length': sum(1 for r in rows if r['L'] == 0), 'online_p50': q([r['L'] for r in rows], .5), 'online_p90': q([r['L'] for r in rows], .9),
            'online_p99': q([r['L'] for r in rows], .99),
            'calibration_rows': len(cal), 'calibration_capped_pct': 100 * np.mean([r['L'] >= r['max_completion_tokens'] for r in cal]),
            'calibration_finish_length': sum(1 for r in cal if r['finish_reason'] == 'length'),
            'calibration_p50': q([r['L'] for r in cal], .5), 'calibration_p90': q([r['L'] for r in cal], .9), 'calibration_p99': q([r['L'] for r in cal], .99),
            'holdout2_rows': len(h2), 'holdout2_capped_pct': 100 * np.mean([r['L'] >= r['max_completion_tokens'] for r in h2]),
            'holdout2_p50': q([r['L'] for r in h2], .5), 'holdout2_p90': q([r['L'] for r in h2], .9),
            'pred_le_1_pct': 100 * np.mean([r['p'] <= 1.5 for r in ev]) if ev else None,
            'by_host': {h: {'rows': sum(1 for r in rows if r['host'] == h), 'capped_pct': 100 * np.mean([r['L'] >= CAP for r in rows if r['host'] == h])}
                        for h in ('vast', 'bridges')}}
        # reproducibility of lengths for the same (request, model) across cells and hosts
        diffs_same_host, diffs_cross_host, pairs = [], [], 0
        for (rid, mm), obs in by_pair.items():
            if mm != m or len(obs) < 2:
                continue
            pairs += 1
            Ls = defaultdict(list)
            for h, L in obs:
                Ls[h].append(L)
            for h, v in Ls.items():
                if len(v) > 1:
                    diffs_same_host.append(max(v) - min(v))
            if len(Ls) == 2:
                diffs_cross_host.append(abs(np.median(Ls['vast']) - np.median(Ls['bridges'])))
        inv['length_reproducibility'][m] = {
            'repeated_pairs': pairs,
            'same_host_range_p50': q(diffs_same_host, .5) if diffs_same_host else None, 'same_host_range_p90': q(diffs_same_host, .9) if diffs_same_host else None,
            'same_host_identical_pct': 100 * np.mean([d == 0 for d in diffs_same_host]) if diffs_same_host else None,
            'cross_host_absdiff_p50': q(diffs_cross_host, .5) if diffs_cross_host else None, 'cross_host_absdiff_p90': q(diffs_cross_host, .9) if diffs_cross_host else None,
            'cross_host_identical_pct': 100 * np.mean([d == 0 for d in diffs_cross_host]) if diffs_cross_host else None}
    json.dump(inv, (a.output / 'data-inventory.json').open('w'), indent=1)

    # ---------------- 1. Reserve reconstruction (engine rule replayed on finished requests) ----------------
    reserve = {'rule': 'reserve = max(floor, quantile of positive residuals max(0, L - ceil(p)) over the last 4096 finished '
                       'requests with p > 0); floor 32/64/128 at kv usage <=0.8/<=0.9/>0.9; quantile 0.50 until 200 samples, '
                       'then 0.65 (kv<=0.8), 0.75 (kv<=0.9), 0.85 (kv>0.9). Replayed per (run, model) over the run\'s finished requests.',
               'by_run_model': {}, 'by_model': {}, 'reserve_used': {}}
    per_model_q = defaultdict(lambda: defaultdict(list))
    for m in MODELS:
        runs = sorted({r['run'] for r in eval_rows if r['model'] == m})
        for run in runs:
            rows = [r for r in eval_rows if r['model'] == m and r['run'] == run]
            rows.sort(key=lambda r: (r['first_token_ts_s'] or 0) + (r['decode_ms'] or 0) / 1000)   # finish order
            res = sorted(max(0.0, r['L'] - math.ceil(r['p'])) for r in rows[-4096:] if r['p'] > 0)
            entry = {'finished': len(rows), 'window': len(res), 'positive_residual_pct': 100 * np.mean([x > 0 for x in res]) if res else None}
            for qq in RESERVE_QUANTILES:
                v = max(FLOOR_TOKENS, engine_residual_quantile(res, qq))
                entry[f'reserve_q{int(qq*100)}'] = v
                per_model_q[m][qq].append(v)
            reserve['by_run_model'][f'{m}::{run}'] = entry
        reserve['by_model'][m] = {f'reserve_q{int(qq*100)}_median_over_runs': float(np.median(per_model_q[m][qq])) for qq in RESERVE_QUANTILES}
        reserve['by_model'][m].update({f'reserve_q{int(qq*100)}_range': [float(min(per_model_q[m][qq])), float(max(per_model_q[m][qq]))] for qq in RESERVE_QUANTILES})
        reserve['reserve_used'][m] = int(round(float(np.median(per_model_q[m][0.65]))))
    reserve['note'] = ('reserve_used is the warm, low-congestion (q65) value; the Qwen analysis used the median snapshot reserve '
                       '(0.6B 308, 8B 614, 32B 565). Sensitivity to the q50 and q85 reserves is reported in candidate-evaluation.json.')
    json.dump(reserve, (a.output / 'reserve-reconstruction.json').open('w'), indent=1)
    reserve_used = reserve['reserve_used']
    print('reserve_used', reserve_used)

    # ---------------- 2. Predictor bias ----------------
    bias = {}
    for m in MODELS:
        for b in BUCKETS:
            rows = [r for r in eval_rows if r['model'] == m and r['bucket'] == b]
            allrows = [r for r in online if r['model'] == m and r['bucket'] == b]
            cal = np.array([r['L'] for r in calibration if r['model'] == m and r['bucket'] == b], float)
            h2 = np.array([r['L'] for r in holdout2 if r['model'] == m and r['bucket'] == b], float)
            if not rows:
                continue
            L = np.array([r['L'] for r in rows], float); p = np.array([r['p'] for r in rows], float)
            bias[f'{m}/{b}'] = {'n': len(rows), 'n_all_policies': len(allrows), 'distinct_prompts': len({r['example_id'] for r in rows}),
                                'pred_le_1': int((p <= 1.5).sum()),
                                'online_p50': q(L, .5), 'online_p90': q(L, .9), 'online_p99': q(L, .99), 'online_capped': int((L >= CAP).sum()),
                                'online_capped_pct_all_policies': 100 * np.mean([r['L'] >= CAP for r in allrows]),
                                'pred_p50': q(p, .5), 'pred_p90': q(p, .9), 'pred_over_actual_median_ratio': q(p, .5) / max(q(L, .5), 1),
                                'ratio_p_over_L_median': q(p / np.maximum(L, 1), .5),
                                'resid_p50': q(L - p, .5), 'resid_p90': q(L - p, .9), 'resid_p99': q(L - p, .99), 'resid_positive_pct': 100 * float(np.mean(L - np.ceil(p) > 0)),
                                'calibration_n': int(len(cal)), 'calibration_p50': q(cal, .5), 'calibration_p90': q(cal, .9), 'calibration_capped': int((cal >= CAP).sum()),
                                'holdout2_n': int(len(h2)), 'holdout2_p50': q(h2, .5), 'holdout2_p90': q(h2, .9), 'holdout2_capped': int((h2 >= CAP).sum()),
                                'by_host': {h: {'n': int(sum(1 for r in rows if r['host'] == h)),
                                                'online_p50': q([r['L'] for r in rows if r['host'] == h], .5),
                                                'resid_p50': q([r['L'] - r['p'] for r in rows if r['host'] == h], .5)} for h in ('vast', 'bridges')}}
        rows = [r for r in eval_rows if r['model'] == m]
        L = np.array([r['L'] for r in rows], float); p = np.array([r['p'] for r in rows], float)
        bias[f'{m}/ALL'] = {'n': len(rows), 'pred_le_1': int((p <= 1.5).sum()), 'online_p50': q(L, .5), 'online_p90': q(L, .9), 'online_p99': q(L, .99),
                            'online_capped': int((L >= CAP).sum()), 'pred_p50': q(p, .5), 'resid_p50': q(L - p, .5), 'resid_p90': q(L - p, .9), 'resid_p99': q(L - p, .99),
                            'mae_tokens': float(np.mean(np.abs(L - p))), 'coverage_pred_plus_reserve_pct': 100 * float(np.mean(np.ceil(p) + reserve_used[m] >= L))}
    json.dump(bias, (a.output / 'predictor-bias.json').open('w'), indent=1)

    # ---------------- 3. Tables ----------------
    cal_by_model = {m: [r for r in calibration if r['model'] == m] for m in MODELS}
    h2_by_model = {m: [r for r in holdout2 if r['model'] == m] for m in MODELS}
    cal_tables = {m: et.build_tables(cal_by_model[m], [lambda r: prompt_bin(r['pt']), lambda r: 'all']) for m in MODELS}
    cal_bucket_tables = {m: et.build_tables(cal_by_model[m], [lambda r: r['bucket'], lambda r: 'all']) for m in MODELS}
    h2_tables = {m: et.build_tables(h2_by_model[m], [lambda r: prompt_bin(r['pt']), lambda r: 'all']) for m in MODELS}
    online_tables, online_bucket_tables = {}, {}
    for m in MODELS:
        for f in range(et.FOLDS):
            train = [r for r in eval_rows if r['model'] == m and r['fold'] != f]
            online_tables[(m, f)] = et.build_tables(train, [lambda r: (prompt_bin(r['pt']), pred_bin(r['p'])), lambda r: prompt_bin(r['pt']), lambda r: 'all'])
            online_bucket_tables[(m, f)] = et.build_tables(train, [lambda r: (r['bucket'], pred_bin(r['p'])), lambda r: r['bucket'], lambda r: 'all'])
    table_support = {m: {str(k): len(v.values) for k, v in cal_tables[m].levels[0].items()} | {'all': len(cal_tables[m].levels[1]['all'].values)} for m in MODELS}

    def candidates(o):
        """Same construction as evaluate_tail.candidates, plus holdout2-fitted tables."""
        m, g, cap, p, pt, reserve = o['model'], o['g'], o['cap'], o['p'], o['pt'], o['reserve']
        pred_total = int(math.ceil(p)) if p > 0 else 1
        current_T = min(max(pred_total + reserve, g + 1), cap)
        out = {'current': current_T - g}
        out['rolling_floor'] = min(max(current_T, g + reserve), cap) - g

        def total(name, T):
            if T is None:
                T = g + 1
            T = min(max(int(T), g + 1), cap)
            out[name] = T - g
            out['exhausted_only_' + name] = (T - g) if (current_T - g) <= 1 else out['current']
        for qq in (0.5, 0.65, 0.8):
            tag = f'q{int(qq*100)}'
            total(f'cal_model_{tag}', cal_tables[m].levels[-1]['all'].estimate(g, qq))
            total(f'cal_prompt_{tag}', cal_tables[m].estimate([prompt_bin(pt), 'all'], g, qq)[0])
            total(f'cal_bucket_{tag}', cal_bucket_tables[m].estimate([o['bucket'], 'all'], g, qq)[0])
            total(f'online_prompt_pred_{tag}', online_tables[(m, o['fold'])].estimate([(prompt_bin(pt), pred_bin(p)), prompt_bin(pt), 'all'], g, qq)[0])
            total(f'online_bucket_pred_{tag}', online_bucket_tables[(m, o['fold'])].estimate([(o['bucket'], pred_bin(p)), o['bucket'], 'all'], g, qq)[0])
        total('h2_model_q50', h2_tables[m].levels[-1]['all'].estimate(g, 0.5))
        total('h2_prompt_q50', h2_tables[m].estimate([prompt_bin(pt), 'all'], g, 0.5)[0])
        return out

    # ---------------- 4. Synthetic held-out observations ----------------
    obs = []
    for r in eval_rows:
        L = r['L']
        if L <= 0:
            continue
        cap = min(CAP, MAX_MODEL_LEN - r['pt'])
        for j in range(POINTS):
            g = int(L * (j + 0.5) / POINTS)
            pred_total = int(math.ceil(r['p'])) if r['p'] > 0 else 1
            current_T = min(max(pred_total + reserve_used[r['model']], g + 1), cap)
            obs.append({'model': r['model'], 'host': r['host'], 'run': r['run'], 'p_source': r.get('p_source'), 'g': g, 'cap': cap, 'p': r['p'], 'pt': r['pt'], 'bucket': r['bucket'],
                        'fold': r['fold'], 'reserve': reserve_used[r['model']], 'L': L, 'R': L - g, 'E0': current_T - g, 'w': L / POINTS,
                        'ctx': r['pt'] + g, 'exhausted': (current_T - g) <= 1 and g < cap, 'capped': L >= cap})
    print('synthetic observations', len(obs))

    # ---------------- 5. Characterize the current rule ----------------
    char = {'construction': f'{POINTS} evenly spaced generation points per online request with a router prediction; token-weighted (w = L/{POINTS}) '
                            'to mimic the snapshot population; reserve_used per model', 'reserve_used': reserve_used, 'observations': len(obs),
            'requests': len({(o['run'], o['model'], o['bucket'], o['pt'], o['L']) for o in obs}),
            'by_model': {}, 'by_model_host': {}, 'by_model_bucket': {}, 'by_model_exhausted': {}, 'by_model_prompt_bin': {}, 'by_model_capped': {},
            'by_model_pred_bin': {}, 'tail_share': {}, 'by_model_request_weighted': {}}

    def group(keyfn, store, weighted=True):
        groups = defaultdict(list)
        for o in obs:
            groups[keyfn(o)].append(o)
        for k, v in sorted(groups.items(), key=lambda kv: str(kv[0])):
            if weighted:
                d = wmetrics([o['E0'] for o in v], [o['R'] for o in v], [o['w'] for o in v], [o['ctx'] for o in v])
            else:
                d = et.metrics([o['E0'] for o in v], [o['R'] for o in v], [o['ctx'] for o in v])
            w = np.array([o['w'] for o in v], float)
            d.update(actual_remaining_mean=float(np.average([o['R'] for o in v], weights=w)),
                     exhausted_pct=float(np.average([o['exhausted'] for o in v], weights=w) * 100),
                     capped_pct=float(np.average([o['capped'] for o in v], weights=w) * 100),
                     share_of_model_obs_pct=None)
            store[str(k)] = d
    group(lambda o: o['model'], char['by_model'])
    group(lambda o: o['model'], char['by_model_request_weighted'], weighted=False)
    group(lambda o: (o['model'], o['host']), char['by_model_host'])
    group(lambda o: (o['model'], o['bucket']), char['by_model_bucket'])
    group(lambda o: (o['model'], 'exhausted' if o['exhausted'] else 'not_exhausted'), char['by_model_exhausted'])
    group(lambda o: (o['model'], f'prompt_bin{prompt_bin(o["pt"])}'), char['by_model_prompt_bin'])
    group(lambda o: (o['model'], 'capped' if o['capped'] else 'uncapped'), char['by_model_capped'])
    group(lambda o: (o['model'], f'pred_bin{pred_bin(o["p"])}'), char['by_model_pred_bin'])
    for m in MODELS:
        rows = [o for o in obs if o['model'] == m]
        W = sum(o['w'] for o in rows) or 1
        tot_ae = sum(abs(o['E0'] - o['R']) * o['w'] for o in rows) or 1
        tot_under = sum(max(o['R'] - o['E0'], 0) * o['w'] for o in rows) or 1
        tot_ctx_under = sum(max(o['R'] - o['E0'], 0) * o['ctx'] * o['w'] for o in rows) or 1

        def share(pred):
            sub = [o for o in rows if pred(o)]
            return {'obs_pct': 100 * sum(o['w'] for o in sub) / W,
                    'abs_error_share_pct': 100 * sum(abs(o['E0'] - o['R']) * o['w'] for o in sub) / tot_ae,
                    'under_share_pct': 100 * sum(max(o['R'] - o['E0'], 0) * o['w'] for o in sub) / tot_under,
                    'context_weighted_under_share_pct': 100 * sum(max(o['R'] - o['E0'], 0) * o['ctx'] * o['w'] for o in sub) / tot_ctx_under}
        char['tail_share'][m] = {'exhausted': share(lambda o: o['exhausted']), 'capped_final_length': share(lambda o: o['capped']),
                                 'govreport': share(lambda o: o['bucket'] == 'govreport-summarization'),
                                 'writingprompts': share(lambda o: o['bucket'] == 'writingprompts'),
                                 'prompt_ge_8192': share(lambda o: o['pt'] >= 8192), 'pred_le_1': share(lambda o: o['p'] <= 1.5),
                                 'abs_error_ge_1000': share(lambda o: abs(o['E0'] - o['R']) >= 1000),
                                 'exhausted_and_capped': share(lambda o: o['exhausted'] and o['capped'])}
    json.dump(char, (a.output / 'tail-characterization.json').open('w'), indent=1)

    # ---------------- 6. Candidate evaluation on the synthetic held-out set ----------------
    names = None
    est_lists = defaultdict(lambda: defaultdict(list))
    for o in obs:
        c = candidates(o)
        names = names or list(c)
        for k, v in c.items():
            est_lists[o['model']][k].append(v)
    results = {'notes': [
        'synthetic held-out set: every online request with a router prediction observed at 8 evenly spaced generation points; token_weighted mimics the snapshot population.',
        'cal_* tables: 10,000 calibration outputs per model (predictor training prompts, holdout indices 0-2499), disjoint from all evaluation prompts.',
        'h2_* tables: 8,000 scored outputs per model of holdout indices 4500-6499, disjoint from the evaluation prompts and from the predictor training prompts.',
        'online_* tables: fitted on the other four prompt folds (fold = sha256(example_id) mod 5) of the pooled online rows; every evaluated observation is from a prompt absent from its table.',
        'exhausted_only_* variants keep the current target unless the current rule says <=1 token remains.',
        'backlog_ratio = sum of token-weighted estimated remaining over actual remaining (the simulator\'s decode occupancy under a stationary population).'],
        'reserve_used': reserve_used, 'table_support_cal_prompt_bins': table_support, 'synthetic_eval': {}, 'exhausted_cohort': {}, 'capped_cohort': {},
        'by_host': {}, 'reserve_sensitivity': {}}
    for m in MODELS:
        rows = [o for o in obs if o['model'] == m]
        act = np.array([o['R'] for o in rows], float); w = np.array([o['w'] for o in rows], float); ctx = [o['ctx'] for o in rows]
        block = {}
        for k in names:
            est = np.array(est_lists[m][k], float)
            block[k] = {'request_weighted': et.metrics(est, act), 'token_weighted': wmetrics(est, act, w, ctx)}
            block[k]['token_weighted']['backlog_ratio'] = block[k]['token_weighted']['sum_est_over_sum_actual']
        results['synthetic_eval'][m] = block
        idx = np.array([o['exhausted'] for o in rows]); cidx = np.array([o['capped'] for o in rows])
        results['exhausted_cohort'][m] = {'obs_token_share_pct': float(w[idx].sum() / w.sum() * 100)} | {
            k: wmetrics(np.array(est_lists[m][k], float)[idx], act[idx], w[idx]) for k in names} if idx.any() else {}
        results['capped_cohort'][m] = {'obs_token_share_pct': float(w[cidx].sum() / w.sum() * 100)} | {
            k: wmetrics(np.array(est_lists[m][k], float)[cidx], act[cidx], w[cidx]) for k in names} if cidx.any() else {}
        check_names = ('current', 'cal_model_q50', 'cal_prompt_q50', 'cal_prompt_q65', 'exhausted_only_cal_prompt_q50', 'online_prompt_pred_q50')
        for h in ('vast', 'bridges'):
            hidx = np.array([o['host'] == h for o in rows])
            if hidx.any():
                results['by_host'][f'{m}::{h}'] = {k: wmetrics(np.array(est_lists[m][k], float)[hidx], act[hidx], w[hidx]) for k in check_names}
        for src in ('router', 'imputed'):
            sidx = np.array([o['p_source'] == src for o in rows])
            if sidx.any():
                results['by_host'][f'{m}::p_source={src}'] = {k: wmetrics(np.array(est_lists[m][k], float)[sidx], act[sidx], w[sidx]) for k in check_names}
        # reserve sensitivity for the current rule (q50 cold-ish and q85 high-congestion reserves)
        sens = {}
        for tag in ('reserve_q50', 'reserve_q85'):
            rv = int(round(reserve['by_model'][m][f'{tag}_median_over_runs']))
            est = []
            for o in rows:
                pred_total = int(math.ceil(o['p'])) if o['p'] > 0 else 1
                est.append(min(max(pred_total + rv, o['g'] + 1), o['cap']) - o['g'])
            sens[f'current_with_{tag}={rv}'] = wmetrics(est, act, w)
        results['reserve_sensitivity'][m] = sens
    json.dump(results, (a.output / 'candidate-evaluation.json').open('w'), indent=1)

    # ---------------- 7. Pseudo-snapshots from timestamps ----------------
    snap = {'construction': f'every {a.snapshot_interval_s} s of wall clock per (run, model): running set = requests with first_token_ts <= t < first_token_ts + decode_ms; '
                            'generated-so-far interpolated linearly over the measured decode time (L >= 2 only); snapshots with < 5 running skipped.',
            'by_model': {}, 'by_model_run': {}}
    key_names = ['current', 'cal_model_q50', 'cal_prompt_q50', 'cal_prompt_q65', 'exhausted_only_cal_prompt_q50', 'exhausted_only_cal_model_q50',
                 'online_prompt_pred_q50', 'h2_prompt_q50', 'rolling_floor', 'current_run_q65', 'current_run_q75', 'current_run_q85']
    snap['construction'] += (' current_run_q65/q75/q85 replay the current rule with the reserve fitted on that run\'s own finished requests '
                             '(warm state at low / medium / high congestion); current uses reserve_used.')
    per_model_snaps = defaultdict(list)
    for m in MODELS:
        for run in sorted({r['run'] for r in eval_rows if r['model'] == m}):
            rows = [r for r in eval_rows if r['model'] == m and r['run'] == run and r['first_token_ts_s'] and r['decode_ms'] and r['decode_ms'] > 0 and r['L'] >= 2]
            if not rows:
                continue
            run_entry = reserve['by_run_model'].get(f'{m}::{run}', {})
            run_reserve = {qq: int(run_entry.get(f'reserve_q{qq}', reserve_used[m])) for qq in (65, 75, 85)}
            starts = np.array([r['first_token_ts_s'] for r in rows]); ends = starts + np.array([r['decode_ms'] for r in rows]) / 1000.0
            Ls = np.array([r['L'] for r in rows])
            snaps = []
            for t in np.arange(starts.min(), ends.max(), a.snapshot_interval_s):
                idx = np.nonzero((starts <= t) & (t < ends))[0]
                if len(idx) < 5:
                    continue
                sums = defaultdict(float); n_exh = 0; n_cap = 0; act_sum = 0.0
                for i in idx:
                    r = rows[i]; L = int(Ls[i])
                    g = int(min(max(int(L * (t - starts[i]) / (ends[i] - starts[i])), 1), L - 1))
                    o = {'model': m, 'g': g, 'cap': min(CAP, MAX_MODEL_LEN - r['pt']), 'p': r['p'], 'pt': r['pt'], 'bucket': r['bucket'], 'fold': r['fold'], 'reserve': reserve_used[m]}
                    c = candidates(o)
                    pred_total = int(math.ceil(r['p'])) if r['p'] > 0 else 1
                    for qq, rv in run_reserve.items():
                        c[f'current_run_q{qq}'] = min(max(pred_total + rv, g + 1), o['cap']) - g
                    for k in key_names:
                        sums[k] += c[k]
                    act_sum += L - g
                    n_exh += c['current'] <= 1
                    n_cap += L >= CAP
                snaps.append({'t': float(t), 'n_running': int(len(idx)), 'actual_remaining_sum': act_sum, 'exhausted_pct': 100 * n_exh / len(idx),
                              'capped_pct': 100 * n_cap / len(idx)} | {f'ratio_{k}': sums[k] / max(act_sum, 1) for k in key_names})
            if not snaps:
                continue
            per_model_snaps[m].extend(snaps)
            snap['by_model_run'][f'{m}::{run}'] = {'snapshots': len(snaps), 'n_running_p50': q([s['n_running'] for s in snaps], .5), 'n_running_p90': q([s['n_running'] for s in snaps], .9),
                                                   'exhausted_pct_median': q([s['exhausted_pct'] for s in snaps], .5), 'capped_pct_median': q([s['capped_pct'] for s in snaps], .5)} | {
                f'backlog_ratio_{k}_median': q([s[f'ratio_{k}'] for s in snaps], .5) for k in key_names}
        snaps = per_model_snaps[m]
        if snaps:
            snap['by_model'][m] = {'snapshots': len(snaps), 'runs': sum(1 for k in snap['by_model_run'] if k.startswith(m + '::')),
                                   'n_running_p50': q([s['n_running'] for s in snaps], .5), 'n_running_p90': q([s['n_running'] for s in snaps], .9),
                                   'n_running_max': max(s['n_running'] for s in snaps),
                                   'exhausted_pct_median': q([s['exhausted_pct'] for s in snaps], .5), 'exhausted_pct_p90': q([s['exhausted_pct'] for s in snaps], .9),
                                   'capped_pct_median': q([s['capped_pct'] for s in snaps], .5)}
            for k in key_names:
                ratios = np.array([s[f'ratio_{k}'] for s in snaps])
                snap['by_model'][m][k] = {'median_ratio': q(ratios, .5), 'p10_ratio': q(ratios, .1), 'p90_ratio': q(ratios, .9), 'min_ratio': float(ratios.min()),
                                          'max_ratio': float(ratios.max()), 'mean_abs_log_ratio': float(np.mean(np.abs(np.log(np.maximum(ratios, 1e-9)))))}
            # the busiest 10% of snapshots (closest to overload)
            thr = q([s['n_running'] for s in snaps], .9)
            busy = [s for s in snaps if s['n_running'] >= thr]
            snap['by_model'][m]['busiest_decile'] = {'snapshots': len(busy), 'n_running_min': min(s['n_running'] for s in busy)} | {
                f'{k}_median_ratio': q([s[f'ratio_{k}'] for s in busy], .5) for k in key_names}
    json.dump(snap, (a.output / 'pseudo-snapshot-backlog.json').open('w'), indent=1)

    # ---------------- 8. Bimodality diagnostic (Ministral 3B vs Qwen 0.6B) ----------------
    bim = {'construction': 'conditional distribution of total length L among requests with L > g, by prompt bin; share capped = L >= 8192; '
                           'stop_soon = L - g <= 250; sources: Ministral calibration outputs and online rows (all policies); Qwen 0.6B calibration table bins.',
           'ministral': {}, 'qwen3-0.6b_calibration': {}}
    G = (0, 250, 500, 1000, 1500, 2000, 3000, 4000, 6000)

    def cond(lengths, label):
        lengths = np.asarray(lengths, int)
        out = {'n': int(len(lengths)), 'capped_pct': 100 * float(np.mean(lengths >= CAP)) if len(lengths) else None, 'p50': q(lengths, .5), 'p90': q(lengths, .9), 'at_g': {}}
        for g in G:
            tail = lengths[lengths > g]
            if len(tail) < 8:
                out['at_g'][str(g)] = {'survivors': int(len(tail))}
                continue
            out['at_g'][str(g)] = {'survivors': int(len(tail)), 'capped_pct': 100 * float(np.mean(tail >= CAP)), 'stop_within_250_pct': 100 * float(np.mean(tail - g <= 250)),
                                   'remaining_p50': q(tail - g, .5), 'remaining_p90': q(tail - g, .9), 'survival_q50_total': q(tail, .5), 'survival_q65_total': q(tail, .65)}
        return out
    for m in MODELS:
        for bin_id in (3, 4):
            cal = [r['L'] for r in calibration if r['model'] == m and prompt_bin(r['pt']) == bin_id]
            on = [r['L'] for r in online if r['model'] == m and prompt_bin(r['pt']) == bin_id]
            bim['ministral'][f'{m}/prompt_bin{bin_id}/calibration'] = cond(cal, m)
            bim['ministral'][f'{m}/prompt_bin{bin_id}/online'] = cond(on, m)
        gov_on = [r['L'] for r in online if r['model'] == m and r['bucket'] == 'govreport-summarization']
        bim['ministral'][f'{m}/govreport/online'] = cond(gov_on, m)
    qt = a.qwen_tables / 'qwen3-0.6b.json'
    if qt.exists():
        t = json.load(qt.open())
        for bin_id in (3, 4):
            bim['qwen3-0.6b_calibration'][f'prompt_bin{bin_id}'] = cond(t['bins'].get(str(bin_id), []), 'qwen3-0.6b')
        bim['qwen3-0.6b_calibration']['source'] = str(qt)
    json.dump(bim, (a.output / 'bimodality.json').open('w'), indent=1)

    # ---------------- 9. Qwen comparison (same synthetic construction) ----------------
    cmp = {'note': 'Qwen numbers are copied from reserve-tail-20260916/evidence (synthetic_eval token_weighted and real-snapshot by_model); Ministral from this run.',
           'synthetic_token_weighted': {}, 'qwen_real_snapshots_current_rule': {}}
    qc = a.qwen_evidence / 'candidate-evaluation.json'
    if qc.exists():
        qcand = json.load(qc.open())
        for m in et.MODELS:
            cmp['synthetic_token_weighted'][m] = {k: qcand['synthetic_eval'][m][k]['token_weighted'] | {'reserve': qcand['synthetic_eval']['reserve_used'][m]}
                                                  for k in ('current', 'cal_model_q50', 'cal_prompt_q50', 'cal_prompt_q65', 'exhausted_only_cal_prompt_q65', 'exhausted_only_cal_prompt_q50', 'online_prompt_pred_q50')
                                                  if k in qcand['synthetic_eval'][m]}
        qchar = json.load((a.qwen_evidence / 'tail-characterization.json').open())
        cmp['qwen_real_snapshots_current_rule'] = qchar['by_model']
    for m in MODELS:
        cmp['synthetic_token_weighted'][m] = {k: results['synthetic_eval'][m][k]['token_weighted'] | {'reserve': reserve_used[m]}
                                              for k in ('current', 'cal_model_q50', 'cal_prompt_q50', 'cal_prompt_q65', 'exhausted_only_cal_prompt_q65', 'exhausted_only_cal_prompt_q50', 'online_prompt_pred_q50')}
    json.dump(cmp, (a.output / 'qwen-comparison.json').open('w'), indent=1)

    # console summary
    for m in MODELS:
        print(m, 'current', {k: round(v) for k, v in char['by_model'][m].items() if k in ('mae', 'median_ae', 'p90_ae', 'p95_ae', 'p99_ae')},
              'cov', round(char['by_model'][m]['coverage_pct']), 'exh%', round(char['by_model'][m]['exhausted_pct'], 1), 'cap%', round(char['by_model'][m]['capped_pct'], 2),
              'backlog', round(char['by_model'][m]['sum_est_over_sum_actual'], 2))
        for k in ('cal_model_q50', 'cal_prompt_q50', 'cal_prompt_q65', 'exhausted_only_cal_prompt_q50', 'online_prompt_pred_q50'):
            tw = results['synthetic_eval'][m][k]['token_weighted']
            print('   ', k, {kk: round(tw[kk]) for kk in ('mae', 'median_ae', 'p90_ae', 'p95_ae', 'p99_ae')}, 'cov', round(tw['coverage_pct']), 'backlog', round(tw['backlog_ratio'], 2))
        if m in snap['by_model']:
            print('   pseudo-snapshots', snap['by_model'][m]['snapshots'], 'running p50', snap['by_model'][m]['n_running_p50'],
                  {k: round(snap['by_model'][m][k]['median_ratio'], 2) for k in ('current', 'cal_prompt_q50', 'exhausted_only_cal_prompt_q50')})


if __name__ == '__main__':
    main()
