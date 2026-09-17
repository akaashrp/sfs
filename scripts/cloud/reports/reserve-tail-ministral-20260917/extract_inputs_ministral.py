"""Build compact inputs for the Ministral remaining-length tail analysis (controller, CPU only).

Read-only over saved evidence. Writes into --output:
  online_rows.jsonl           one row per completed request in every completed Ministral cell
                              (11 Vast cells, 12 audited Bridges cells); router prediction when the
                              policy carried one (shortest_queue, latency_agnostic, round_robin)
  calibration_outputs.jsonl   10,000 outputs per model from calibration indices 0-2499 (the length
                              predictor's training prompts) plus, flagged source=holdout2, the 8,000
                              per model scored outputs of holdout indices 4500-6499 (disjoint from
                              both the online prompts and the predictor training prompts)
  req_map.json                request id -> holdout example id (prompt identity for held-out folds)
  inventory.json              counts, provenance and the overlap checks
"""
import argparse
import csv
import glob
import hashlib
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

BUNDLE = Path('/ocean/projects/cis250162p/aparthas/sfs_cloud_artifacts_20260916')
CAL_ROOT = Path('/ocean/projects/cis250162p/aparthas/sfs_artifacts/'
                'ministral3_calibration_44849564_judge_45009121_20260902/run/completions')
VAST_MIRROR = Path('/ocean/projects/cis250162p/aparthas/sfs_cloud_results_20260916/sfs-vast/baselines')
BRIDGES_REVIEW = Path('/ocean/projects/cis250162p/aparthas/sfs_cloud_20260914/scripts/cloud/bridges-reuse-review-20260916.json')
MODELS = ('ministral3-3b', 'ministral3-8b', 'ministral3-14b')
BUCKETS = ('alpaca', 'govreport-summarization', 'hotpot_qa', 'writingprompts')
CAP = 8192


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def norm_model(name):
    name = str(name)
    return name[:-len('-instruct')] if name.endswith('-instruct') else name


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--vast-extra', type=Path, default=None,
                    help='directory holding <cell>/point.json copies of Vast cells not yet mirrored')
    a = ap.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    inventory = {'bundle': str(BUNDLE), 'calibration_root': str(CAL_ROOT), 'cap': CAP, 'cells': [], 'calibration': {}, 'checks': {}}

    # ---- request map (prompt identity) ----
    req_map = list(csv.DictReader((BUNDLE / 'ministral/request_map.csv').open()))
    online_ids = {(r['bucket'], r['example_id']) for r in req_map}
    assert len(req_map) == 8000 and len(online_ids) == 8000
    inventory['request_map'] = {'path': str(BUNDLE / 'ministral/request_map.csv'), 'sha256': sha256(BUNDLE / 'ministral/request_map.csv'),
                                'rows': len(req_map), 'holdout_prompt_index_range': [min(int(r['holdout_prompt_index']) for r in req_map), max(int(r['holdout_prompt_index']) for r in req_map)]}
    json.dump([{'req_id': r['req_id'], 'example_id': r['example_id'], 'bucket': r['bucket'], 'holdout_prompt_index': int(r['holdout_prompt_index']),
                'prompt_tokens': int(r['prompt_tokens'])} for r in req_map], (a.output / 'req_map.json').open('w'))
    rm = {r['req_id']: r for r in req_map}

    # ---- online rows ----
    points = []
    for p in sorted(glob.glob(str(VAST_MIRROR / 'ministral-20260916*/cells/*/point.json'))):
        points.append(('vast', p))
    if a.vast_extra:
        for p in sorted(glob.glob(str(a.vast_extra / '*/point.json'))):
            points.append(('vast', p))
    review = json.load(BRIDGES_REVIEW.open())
    audit = json.load(open(review['source_audit']))
    assert sha256(review['source_audit']) == review['source_audit_sha256']
    for p, digest in sorted(audit['point_sha256'].items()):
        assert sha256(p) == digest, p
        points.append(('bridges', p))
    # Policies whose logged predicted_output_tokens is the SFS length predictor's mean head (the same
    # predictor the engines load via --output-length-model-path). routebalance logs its own predictor's
    # value (0 of 6,102 pairs matched the shortest_queue value), mooncake/lmdeploy/vllm_sr log none.
    SFS_PREDICTION_POLICIES = {'shortest_queue', 'latency_agnostic', 'round_robin', 'hard', 'score'}
    seen_cells = set()
    staged = []
    sfs_pred = {}
    pred_conflicts = 0
    for host, p in points:
        d = json.load(open(p))
        cfg = d['config']
        for run in d['router']['runs']:
            cell = f"{host}:{run['utility']}:{cfg['request_rate_qps']}"
            assert cell not in seen_cells, cell
            seen_cells.add(cell)
            rows = run['per_request']
            with_pred = 0
            models = Counter()
            for r in rows:
                assert r.get('error') in (None, '') and r['usage_prompt_tokens'] == r['prompt_tokens'], (cell, r['request_id'])
                L = int(r['usage_completion_tokens'])
                assert 0 <= L <= CAP, (cell, r['request_id'], L)
                m = norm_model(r['response_model'])
                assert m in MODELS, m
                mr = rm[r['request_id']]
                assert mr['bucket'] == r['bucket'] and int(mr['prompt_tokens']) == int(r['prompt_tokens']), (cell, r['request_id'])
                pred = r.get('predicted_output_tokens')
                router_pred = None
                if pred is not None and run['utility'] in SFS_PREDICTION_POLICIES:
                    router_pred = float(pred)
                    with_pred += 1
                    key = (r['request_id'], m)
                    if key in sfs_pred and abs(sfs_pred[key] - router_pred) > 1e-6:
                        pred_conflicts += 1
                    sfs_pred.setdefault(key, router_pred)
                models[m] += 1
                decode_ms = float(r['latency_ms']) - float(r['e2e_ttft_ms'])
                staged.append({'host': host, 'run': cell, 'policy': run['utility'], 'qps': cfg['request_rate_qps'], 'request_id': r['request_id'],
                               'response_id': r.get('response_id'), 'example_id': mr['example_id'], 'bucket': r['bucket'], 'model': m,
                               'prompt_tokens': int(r['prompt_tokens']), 'usage_completion_tokens': L,
                               'router_predicted_output_tokens': None if pred is None else float(pred), 'predicted_output_tokens': router_pred,
                               'p_source': 'router' if router_pred is not None else None,
                               'queued_ts_s': r.get('queued_ts_s'), 'first_token_ts_s': r.get('first_token_ts_s'),
                               'ttft_ms': r.get('ttft_ms'), 'e2e_ttft_ms': r.get('e2e_ttft_ms'), 'latency_ms': r.get('latency_ms'),
                               'decode_ms': decode_ms, 'wait_time_ms': r.get('wait_time_ms')})
            inventory['cells'].append({'host': host, 'cell': cell, 'point': p, 'sha256': sha256(p), 'requests': len(rows),
                                       'failed': run['summary']['failed_requests'], 'with_sfs_prediction': with_pred, 'by_model': dict(models)})
    # Impute the (deterministic) SFS prediction for rows of policies that do not log it.
    n_imputed = 0
    n_unavailable = 0
    for r in staged:
        if r['predicted_output_tokens'] is None:
            v = sfs_pred.get((r['request_id'], r['model']))
            if v is not None:
                r['predicted_output_tokens'] = v
                r['p_source'] = 'imputed'
                n_imputed += 1
            else:
                n_unavailable += 1
    out = (a.output / 'online_rows.jsonl').open('w')
    for r in staged:
        out.write(json.dumps(r) + '\n')
    out.close()
    n_rows = len(staged)
    inventory['online_rows'] = n_rows
    inventory['prediction'] = {'policies_logging_sfs_prediction': sorted(SFS_PREDICTION_POLICIES),
                               'rows_with_router_prediction': sum(1 for r in staged if r['p_source'] == 'router'),
                               'rows_with_imputed_prediction': n_imputed, 'rows_without_prediction': n_unavailable,
                               'distinct_request_model_pairs_with_prediction': len(sfs_pred), 'possible_pairs': 8000 * len(MODELS),
                               'router_prediction_conflicts': pred_conflicts,
                               'note': 'the SFS prediction depends only on (prompt, model); rows of policies that do not log it '
                                       '(mooncake_prefill, lmdeploy_proxy, vllm_sr_latency, routebalance) take the value logged for the same '
                                       '(request, model) in a shortest_queue / latency_agnostic / round_robin cell; routebalance\'s own '
                                       'logged prediction is kept in router_predicted_output_tokens and not used.'}

    # ---- calibration outputs (predictor training prompts, indices 0-2499) ----
    cal_out = (a.output / 'calibration_outputs.jsonl').open('w')
    cal_ids = set()
    n_cal = 0
    for m in MODELS:
        for b in BUCKETS:
            path = CAL_ROOT / m / f'{b}.jsonl'
            lengths = []
            idx = set()
            for line in path.open():
                r = json.loads(line)
                assert not r.get('error') and r['model_label'] == m and r['bucket'] == b
                eid = r['prompt_metadata']['example_id']
                assert (b, eid) not in online_ids, (m, b, eid)
                cal_ids.add((b, eid))
                idx.add(r['prompt_index'])
                L = int(r['response']['completion_tokens'])
                assert 0 <= L <= int(r['max_completion_tokens'])
                lengths.append(L)
                cal_out.write(json.dumps({'source': 'calibration', 'model': m, 'bucket': b, 'example_id': eid, 'prompt_index': r['prompt_index'],
                                          'prompt_tokens': int(r['prompt_tokens']), 'completion_tokens': L, 'finish_reason': r['response']['finish_reason'],
                                          'max_completion_tokens': int(r['max_completion_tokens'])}) + '\n')
                n_cal += 1
            assert idx == set(range(2500)), (m, b)
            inventory['calibration'][f'{m}/{b}'] = {'path': str(path), 'sha256': sha256(path), 'records': len(lengths),
                                                    'capped': sum(1 for L in lengths if L >= CAP), 'p50': sorted(lengths)[len(lengths) // 2]}
    # ---- second held-out set: scored outputs of holdout indices 4500-6499 (bundle scores/, non-overlapping half) ----
    n_h2 = 0
    h2_ids = set()
    for m in MODELS:
        for b in BUCKETS:
            path = BUNDLE / 'ministral/scores' / m / f'{b}_scored.jsonl'
            n_over = 0
            n_keep = 0
            for line in path.open():
                r = json.loads(line)
                assert not r.get('error') and r['model_label'] == m and r['bucket'] == b
                eid = r['prompt_metadata']['example_id']
                if (b, eid) in online_ids:
                    n_over += 1
                    continue
                assert (b, eid) not in cal_ids
                h2_ids.add((b, eid))
                L = int(r['response']['completion_tokens'])
                cal_out.write(json.dumps({'source': 'holdout2', 'model': m, 'bucket': b, 'example_id': eid, 'prompt_index': r['prompt_index'],
                                          'prompt_tokens': int(r['prompt_tokens']), 'completion_tokens': L, 'finish_reason': r['response']['finish_reason'],
                                          'max_completion_tokens': int(r['max_completion_tokens'])}) + '\n')
                n_keep += 1
                n_h2 += 1
            inventory['calibration'][f'holdout2:{m}/{b}'] = {'path': str(path), 'sha256': sha256(path), 'records_total': n_over + n_keep,
                                                             'records_overlapping_online_excluded': n_over, 'records_kept': n_keep}
    cal_out.close()
    test_ids = set(json.load((BUNDLE / 'ministral/length/test_example_ids.json').open())['example_ids'])
    cal_eids = {e for _, e in cal_ids}
    inventory['calibration_rows'] = n_cal
    inventory['holdout2_rows'] = n_h2
    inventory['checks'] = {
        'calibration_prompts_overlapping_online': len({(b, e) for (b, e) in cal_ids} & online_ids),
        'holdout2_prompts_overlapping_online': len(h2_ids & online_ids),
        'holdout2_prompts_overlapping_calibration': len(h2_ids & cal_ids),
        'predictor_test_ids': len(test_ids), 'predictor_test_ids_in_calibration': len(test_ids & cal_eids),
        'predictor_test_ids_in_online': len(test_ids & {e for _, e in online_ids}),
        'predictor_training_summary': json.load((BUNDLE / 'ministral/length/training_summary.json').open())['total_examples'],
        'bundle_scores_note': 'bundle ministral/scores/*_scored.jsonl hold holdout indices 2500-6499: the first 2,000 per bucket ARE the online '
                              'evaluation prompts and were excluded; the remaining 2,000 per bucket are kept as holdout2.'}
    json.dump(inventory, (a.output / 'inventory.json').open('w'), indent=1)
    print(json.dumps({k: v for k, v in inventory.items() if k not in ('cells', 'calibration')}, indent=1))
    for c in inventory['cells']:
        print(c['cell'], c['requests'], 'sfs_pred', c['with_sfs_prediction'], c['by_model'])


if __name__ == '__main__':
    main()
