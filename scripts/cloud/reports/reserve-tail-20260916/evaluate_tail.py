"""Offline tail characterization and held-out candidate evaluation for remaining-decode targets.

Read-only analysis over saved evidence; no serving state or production estimator is touched.

Inputs (compact extracts built by extract_inputs.py from the raw controller mirrors):
  per_request.jsonl      one row per completed request in five Vast controls (router prediction, actual length)
  snapshots.json         decoded read-only scheduler snapshots (repeat1: 15, lane-a: 21, lane-b: 21)
  calibration_outputs.jsonl  10,000 outputs per model from calibration indices 0-2499 (predictor training prompts)
  req_map.json           request id -> holdout example id (prompt identity used for held-out folds)

Outputs JSON evidence into --output.
"""
import argparse
import hashlib
import json
import math
from bisect import bisect_right
from collections import defaultdict
from pathlib import Path

import numpy as np

MODELS = ('qwen3-0.6b', 'qwen3-8b', 'qwen3-32b')
PROMPT_EDGES = (0, 128, 512, 2048, 8192, 32768, 10**9)
PRED_EDGES = (0, 1.5, 64, 128, 256, 512, 1024, 10**9)   # first bin is the degenerate p<=1 prediction
FOLDS = 5
CAP = 8192


def prompt_bin(n):
    return bisect_right(PROMPT_EDGES, n) - 1


def pred_bin(p):
    return bisect_right(PRED_EDGES, p) - 1


def fold_of(example_id):
    return int(hashlib.sha256(example_id.encode()).hexdigest(), 16) % FOLDS


def q(a, p):
    return float(np.quantile(a, p)) if len(a) else float('nan')


class SurvivalTable:
    """Quantile of total length among training lengths strictly greater than generated-so-far."""

    def __init__(self, lengths, min_support=8):
        self.values = np.sort(np.asarray(lengths, dtype=np.int64))
        self.min_support = min_support

    def support(self, generated):
        return int(len(self.values) - np.searchsorted(self.values, generated, side='right'))

    def estimate(self, generated, quantile):
        start = int(np.searchsorted(self.values, generated, side='right'))
        tail = self.values[start:]
        if len(tail) < self.min_support:
            return None
        return int(math.ceil(float(np.quantile(tail, quantile))))


class Backoff:
    """Hierarchy of survival tables: most specific cell first, then coarser keys."""

    def __init__(self, levels):
        self.levels = levels  # list of dict key-> SurvivalTable, most specific first

    def estimate(self, keys, generated, quantile):
        for level, key in zip(self.levels, keys):
            table = level.get(key)
            if table is None:
                continue
            value = table.estimate(generated, quantile)
            if value is not None:
                return value, key
        return None, None


def build_tables(rows, key_fns):
    levels = []
    for fn in key_fns:
        groups = defaultdict(list)
        for r in rows:
            groups[fn(r)].append(r['L'])
        levels.append({k: SurvivalTable(v) for k, v in groups.items()})
    return Backoff(levels)


def metrics(est, act, ctx=None):
    est = np.asarray(est, dtype=float); act = np.asarray(act, dtype=float)
    err = est - act; ae = np.abs(err)
    out = {'n': int(len(err)), 'mae': float(ae.mean()), 'median_ae': q(ae, .5), 'p90_ae': q(ae, .9),
           'p95_ae': q(ae, .95), 'p99_ae': q(ae, .99), 'max_ae': float(ae.max()),
           'mean_under': float(np.clip(-err, 0, None).mean()), 'mean_over': float(np.clip(err, 0, None).mean()),
           'coverage_pct': float((err >= 0).mean() * 100), 'bias': float(err.mean()),
           'sum_est_over_sum_actual': float(est.sum() / max(act.sum(), 1))}
    if ctx is not None:
        ctx = np.asarray(ctx, dtype=float)
        out['context_weighted_under'] = float((np.clip(-err, 0, None) * ctx).sum() / ctx.sum())
        out['context_weighted_over'] = float((np.clip(err, 0, None) * ctx).sum() / ctx.sum())
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    a = ap.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)

    per_request = [json.loads(l) for l in (a.data / 'per_request.jsonl').open()]
    snapshots = json.load((a.data / 'snapshots.json').open())
    calibration = [json.loads(l) for l in (a.data / 'calibration_outputs.jsonl').open()]
    req_map = {r['req_id']: r for r in json.load((a.data / 'req_map.json').open())}
    for r in per_request:
        r['example_id'] = req_map[r['request_id']]['example_id']
        r['fold'] = fold_of(r['example_id'])
        r['L'] = int(r['usage_completion_tokens']); r['p'] = float(r['predicted_output_tokens'])
        r['pt'] = int(r['prompt_tokens']); r['model'] = r['response_model']
    for r in calibration:
        r['L'] = int(r['completion_tokens']); r['pt'] = int(r['prompt_tokens'])
    online_by_run = defaultdict(dict)
    for r in per_request:
        online_by_run[r['run']][r['response_id']] = r

    # ---------------- 1. Real snapshot observations ----------------
    obs = []
    for s in snapshots:
        lookup = online_by_run[s['run']]
        reserve = int(s['decode_reserve_tokens'])
        for rid in s['running_request_ids']:
            rq = s['requests'][rid]
            if rq['num_prompt_processed_tokens'] < rq['num_prompt_tokens']:
                continue
            r = lookup.get(rid)
            if r is None:
                continue
            g = int(rq['num_output_processed_tokens']); T = int(rq['num_output_target_tokens'])
            cap = min(int(rq['max_tokens']), int(s['max_model_len']) - int(rq['num_prompt_tokens']))
            L = r['L']
            assert 0 <= g <= L <= cap, (rid, g, L, cap)
            obs.append({'run': s['run'], 'snapshot': s['file'], 'model': s['model'], 'id': rid, 'g': g, 'T': T,
                        'reserve': reserve, 'cap': cap, 'L': L, 'R': L - g, 'E0': T - g, 'pt': int(rq['num_prompt_tokens']),
                        'p': r['p'], 'bucket': r['bucket'], 'fold': r['fold'], 'example_id': r['example_id'],
                        'ctx': int(rq['num_prompt_tokens']) + g, 'exhausted': (T - g) <= 1 and g < cap,
                        'kv_usage': s['kv_cache_usage'], 'num_waiting': s['num_waiting'], 'num_running': s['num_running'],
                        'engine_pred_est': (T - reserve) if (T - g) > 1 and T < cap else None})
    print('snapshot observations', len(obs), 'unique requests', len({o['id'] + o['run'] for o in obs}))

    # ---------------- 2. Characterize the current rule's tail ----------------
    char = {'observations': len(obs), 'unique_requests': len({o['id'] + o['run'] for o in obs}),
            'snapshots': len(snapshots), 'by_model': {}, 'by_model_run': {}, 'by_model_bucket': {}, 'by_model_exhausted': {},
            'by_model_prompt_bin': {}, 'by_model_capped': {}, 'by_model_pred_bin': {}, 'tail_share': {}}
    def group(keyfn, store):
        groups = defaultdict(list)
        for o in obs:
            groups[keyfn(o)].append(o)
        for k, v in sorted(groups.items(), key=lambda kv: str(kv[0])):
            store[str(k)] = dict(metrics([o['E0'] for o in v], [o['R'] for o in v], [o['ctx'] for o in v]),
                                 actual_remaining_mean=float(np.mean([o['R'] for o in v])),
                                 actual_remaining_p50=q([o['R'] for o in v], .5),
                                 exhausted_pct=float(np.mean([o['exhausted'] for o in v]) * 100),
                                 capped_pct=float(np.mean([o['L'] >= o['cap'] for o in v]) * 100))
    group(lambda o: o['model'], char['by_model'])
    group(lambda o: (o['model'], o['run']), char['by_model_run'])
    group(lambda o: (o['model'], o['bucket']), char['by_model_bucket'])
    group(lambda o: (o['model'], 'exhausted' if o['exhausted'] else 'not_exhausted'), char['by_model_exhausted'])
    group(lambda o: (o['model'], f'prompt_bin{prompt_bin(o["pt"])}'), char['by_model_prompt_bin'])
    group(lambda o: (o['model'], 'capped' if o['L'] >= o['cap'] else 'uncapped'), char['by_model_capped'])
    group(lambda o: (o['model'], f'pred_bin{pred_bin(o["p"])}'), char['by_model_pred_bin'])
    # Share of total absolute error and of total under-estimated tokens contributed by subsets.
    for m in MODELS:
        rows = [o for o in obs if o['model'] == m]
        tot_ae = sum(abs(o['E0'] - o['R']) for o in rows) or 1
        tot_under = sum(max(o['R'] - o['E0'], 0) for o in rows) or 1
        tot_ctx_under = sum(max(o['R'] - o['E0'], 0) * o['ctx'] for o in rows) or 1
        def share(pred):
            sub = [o for o in rows if pred(o)]
            return {'obs_pct': 100 * len(sub) / max(len(rows), 1),
                    'abs_error_share_pct': 100 * sum(abs(o['E0'] - o['R']) for o in sub) / tot_ae,
                    'under_share_pct': 100 * sum(max(o['R'] - o['E0'], 0) for o in sub) / tot_under,
                    'context_weighted_under_share_pct': 100 * sum(max(o['R'] - o['E0'], 0) * o['ctx'] for o in sub) / tot_ctx_under}
        char['tail_share'][m] = {
            'exhausted': share(lambda o: o['exhausted']),
            'capped_final_length': share(lambda o: o['L'] >= o['cap']),
            'govreport': share(lambda o: o['bucket'] == 'govreport-summarization'),
            'prompt_ge_8192': share(lambda o: o['pt'] >= 8192),
            'pred_le_1': share(lambda o: o['p'] <= 1.5),
            'abs_error_ge_1000': share(lambda o: abs(o['E0'] - o['R']) >= 1000),
            'exhausted_and_capped': share(lambda o: o['exhausted'] and o['L'] >= o['cap'])}
    json.dump(char, (a.output / 'tail-characterization.json').open('w'), indent=1)

    # ---------------- 3. Candidate estimators ----------------
    cal_by_model = {m: [r for r in calibration if r['model'] == m] for m in MODELS}
    cal_tables = {m: build_tables(cal_by_model[m], [lambda r: prompt_bin(r['pt']), lambda r: 'all']) for m in MODELS}
    cal_bucket_tables = {m: build_tables(cal_by_model[m], [lambda r: r['bucket'], lambda r: 'all']) for m in MODELS}
    online_tables = {}   # (model, fold) -> Backoff trained on other folds
    online_bucket_tables = {}
    for m in MODELS:
        for f in range(FOLDS):
            train = [r for r in per_request if r['model'] == m and r['fold'] != f]
            online_tables[(m, f)] = build_tables(train, [
                lambda r: (prompt_bin(r['pt']), pred_bin(r['p'])),
                lambda r: prompt_bin(r['pt']), lambda r: 'all'])
            online_bucket_tables[(m, f)] = build_tables(train, [
                lambda r: (r['bucket'], pred_bin(r['p'])), lambda r: r['bucket'], lambda r: 'all'])

    def candidates(o, reserve_override=None):
        """Return dict name -> estimated remaining tokens for one observation."""
        m, g, cap, p, pt = o['model'], o['g'], o['cap'], o['p'], o['pt']
        reserve = o['reserve'] if reserve_override is None else reserve_override
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
        # Prediction-anchored: keep the prediction until exhausted, then conditional median
        return out

    results = {'snapshot_eval': {}, 'synthetic_eval': {}, 'notes': [
        'snapshot_eval: 57 real read-only snapshots (repeat1 15, lane-a 21, lane-b 21); observations are repeated requests across snapshots within a run, not independent samples.',
        'online_* tables are fitted on the other four prompt folds (fold = sha256(example_id) mod 5) of the five pooled Vast controls; every evaluated observation is from a prompt absent from its table.',
        'cal_* tables use only the 30,000 calibration outputs (predictor training prompts), disjoint from the 16,000 evaluation prompts.',
        'exhausted_only_* variants keep the current target unless the current rule says <=1 token remains.',
        'synthetic_eval: each held-out online request is observed at 8 evenly spaced generation points; token_weighted approximates the snapshot population (long generations are observed more often).']}
    names = None
    per_model_rows = defaultdict(lambda: defaultdict(list))
    for o in obs:
        c = candidates(o)
        names = names or list(c)
        for k, v in c.items():
            per_model_rows[o['model']][k].append(v)
    for m in MODELS:
        rows = [o for o in obs if o['model'] == m]
        act = [o['R'] for o in rows]; ctx = [o['ctx'] for o in rows]
        results['snapshot_eval'][m] = {k: metrics(per_model_rows[m][k], act, ctx) for k in names}
        # per-snapshot backlog error (sum of remaining estimates vs sum of actual remaining)
        by_snap = defaultdict(list)
        for i, o in enumerate(rows):
            by_snap[o['snapshot']].append(i)
        backlog = {}
        for k in names:
            est = np.array(per_model_rows[m][k], dtype=float); actual = np.array(act, dtype=float)
            ratios = [est[idx].sum() / max(actual[idx].sum(), 1) for idx in by_snap.values()]
            backlog[k] = {'median_ratio': float(np.median(ratios)), 'min_ratio': float(min(ratios)), 'max_ratio': float(max(ratios)),
                          'mean_abs_log_ratio': float(np.mean(np.abs(np.log(ratios))))}
        results['snapshot_eval'][m + '::backlog_ratio'] = backlog
        # cohort: exhausted only
        idx = [i for i, o in enumerate(rows) if o['exhausted']]
        if idx:
            results['snapshot_eval'][m + '::exhausted_cohort'] = {k: metrics([per_model_rows[m][k][i] for i in idx], [act[i] for i in idx]) for k in names}

    # Synthetic held-out evaluation over all online requests (prompt-level folds).
    typical_reserve = {m: int(np.median([s['decode_reserve_tokens'] for s in snapshots if s['model'] == m])) for m in MODELS}
    results['synthetic_eval']['reserve_used'] = typical_reserve
    for m in MODELS:
        rows = [r for r in per_request if r['model'] == m]
        est_lists = defaultdict(list); act = []; w = []
        for r in rows:
            L = r['L']; cap = min(8192, 131072 - r['pt'])
            for j in range(8):
                g = int(L * (j + 0.5) / 8)
                o = {'model': m, 'g': g, 'cap': cap, 'p': r['p'], 'pt': r['pt'], 'bucket': r['bucket'], 'fold': r['fold'], 'reserve': typical_reserve[m]}
                c = candidates(o)
                for k, v in c.items():
                    est_lists[k].append(v)
                act.append(L - g); w.append(L / 8)
        act = np.array(act, dtype=float); w = np.array(w, dtype=float)
        block = {}
        for k in names:
            est = np.array(est_lists[k], dtype=float); err = est - act; ae = np.abs(err)
            order = np.argsort(ae); cw = np.cumsum(w[order]) / w.sum()
            def wq(p):
                return float(ae[order][np.searchsorted(cw, p)]) if p < 1 else float(ae.max())
            block[k] = {'n_points': int(len(ae)), 'request_weighted': {'mae': float(ae.mean()), 'p90_ae': q(ae, .9), 'p95_ae': q(ae, .95), 'p99_ae': q(ae, .99), 'coverage_pct': float((err >= 0).mean() * 100)},
                        'token_weighted': {'mae': float((ae * w).sum() / w.sum()), 'p90_ae': wq(.9), 'p95_ae': wq(.95), 'p99_ae': wq(.99),
                                           'coverage_pct': float(((err >= 0) * w).sum() / w.sum() * 100),
                                           'mean_under': float((np.clip(-err, 0, None) * w).sum() / w.sum()),
                                           'mean_over': float((np.clip(err, 0, None) * w).sum() / w.sum())}}
        results['synthetic_eval'][m] = block
    json.dump(results, (a.output / 'candidate-evaluation.json').open('w'), indent=1)

    # Compact observation table with the main candidate estimates for replay on Vast.
    replay_rows = []
    for o in obs:
        c = candidates(o)
        replay_rows.append({k: o[k] for k in ('run', 'snapshot', 'model', 'id', 'g', 'T', 'reserve', 'cap', 'L', 'pt', 'p', 'bucket', 'fold', 'exhausted')} | {'est': c})
    json.dump({'names': names, 'rows': replay_rows}, (a.output / 'snapshot-observations.json').open('w'))

    # Predictor bias evidence: online final length vs router prediction per model/bucket.
    bias = {}
    for m in MODELS:
        for b in sorted({r['bucket'] for r in per_request}):
            rows = [r for r in per_request if r['model'] == m and r['bucket'] == b]
            if not rows:
                continue
            L = np.array([r['L'] for r in rows], dtype=float); p = np.array([r['p'] for r in rows], dtype=float)
            cal = np.array([r['L'] for r in cal_by_model[m] if r['bucket'] == b], dtype=float)
            bias[f'{m}/{b}'] = {'n': len(rows), 'pred_le_1': int((p <= 1.5).sum()), 'online_p50': q(L, .5), 'online_p90': q(L, .9), 'online_capped': int((L >= 8192).sum()),
                                'pred_p50': q(p, .5), 'resid_p50': q(L - p, .5), 'resid_p90': q(L - p, .9), 'resid_p99': q(L - p, .99),
                                'calibration_p50': q(cal, .5), 'calibration_p90': q(cal, .9), 'calibration_capped': int((cal >= 8192).sum()), 'calibration_n': int(len(cal))}
    json.dump(bias, (a.output / 'predictor-bias.json').open('w'), indent=1)
    print(json.dumps({m: {k: round(v['p95_ae']) for k, v in results['snapshot_eval'][m].items()} for m in MODELS}, indent=0)[:3000])


if __name__ == '__main__':
    main()
