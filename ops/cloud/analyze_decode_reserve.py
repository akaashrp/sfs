"""Compare reserve rules on frozen snapshots; never modifies serving state."""
import argparse
import copy
import hashlib
import importlib.util
import json
import math
import statistics
import struct
from bisect import bisect_right
from collections import defaultdict
from pathlib import Path

import msgspec


def metrics(rows):
    if not rows:
        return {'observations': 0}
    errors = [r['estimate'] - r['actual'] for r in rows]
    return {
        'observations': len(rows),
        'unique_requests': len({r['id'] for r in rows}),
        'mae_tokens': statistics.mean(abs(x) for x in errors),
        'underestimate_mean_tokens': statistics.mean(max(-x, 0) for x in errors),
        'overestimate_mean_tokens': statistics.mean(max(x, 0) for x in errors),
        'coverage_pct': 100 * statistics.mean(x >= 0 for x in errors),
        'one_remaining_actual_ge128': sum(r['estimate'] <= 1 and r['actual'] >= 128 for r in rows),
        'context_weighted_underestimate_tokens': sum(max(-e, 0) * r['context'] for e, r in zip(errors, rows)) / sum(r['context'] for r in rows),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('point', 'snapshots', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--extension', type=Path)
    parser.add_argument('--calibration', type=Path)
    args = parser.parse_args()
    native = None
    if args.extension:
        import torch  # Load libtorch before the native extension; no GPU work.
        spec = importlib.util.spec_from_file_location('_scheduler_sim', args.extension)
        native = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(native)
    point = json.loads(args.point.read_text())
    requests = point['router']['runs'][0]['per_request']
    lookup = {r['response_id']: r for r in requests}
    assert len(lookup) == 16000 and not any(r.get('error') for r in requests)
    header = struct.Struct('<8sIIQQdQdddddddd')
    policies = ('current', 'rolling_half_reserve', 'rolling_floor', 'rolling_double_reserve', 'exhausted_only_floor')
    calibration = json.loads(args.calibration.read_text()) if args.calibration else None
    if calibration:
        assert calibration['heldout_overlap'] == 0
        policies += ('survival_mean', 'survival_median', 'exhausted_survival_mean', 'exhausted_survival_median')
    support_counts = defaultdict(list)
    observations = defaultdict(list)
    replay = []
    inventory = []
    for path in sorted(args.snapshots.glob('*.bin')):
        raw = path.read_bytes()
        h = header.unpack(raw[:header.size])
        assert h[0] == b'VLLMSHM1' and h[2] == header.size and h[3] % 2 == 0
        assert len(raw) == header.size + h[6]
        state = msgspec.msgpack.decode(raw[header.size:])
        model = path.stem.split('-', 1)[1]
        reserve = state['decode_reserve_tokens']
        variants = {name: copy.deepcopy(state) for name in (*policies, 'oracle_running_only')}
        count = 0
        for rid in state['running_request_ids']:
            r = state['requests'][rid]
            if r['num_prompt_processed_tokens'] < r['num_prompt_tokens']:
                continue
            actual = lookup[rid]
            assert actual['response_model'] == model
            generated = r['num_output_processed_tokens']
            cap = min(r['max_tokens'], state['config']['max_model_len'] - r['num_prompt_tokens'])
            assert 0 <= generated <= actual['usage_completion_tokens'] <= cap
            original_target = r['num_output_target_tokens']
            targets = {
                'current': original_target,
                'rolling_floor': max(original_target, generated + reserve),
                'rolling_half_reserve': max(original_target, generated + math.ceil(reserve / 2)),
                'rolling_double_reserve': max(original_target, generated + 2 * reserve),
                'exhausted_only_floor': generated + reserve if original_target <= generated + 1 else original_target,
                'oracle_running_only': max(generated + 1, actual['usage_completion_tokens']),
            }
            if calibration:
                lengths = calibration['models'][model]
                survivors = lengths[bisect_right(lengths, generated):]
                support_counts[model].append(len(survivors))
                capped = [min(length, cap) for length in survivors]
                mean_target = math.ceil(statistics.mean(capped)) if capped else cap
                median_target = math.ceil(statistics.median(capped)) if capped else cap
                targets.update(survival_mean=mean_target, survival_median=median_target,
                               exhausted_survival_mean=mean_target if original_target <= generated+1 else original_target,
                               exhausted_survival_median=median_target if original_target <= generated+1 else original_target)
            targets = {name: min(target, cap) for name, target in targets.items()}
            assert targets['current'] == r['num_output_target_tokens'], (rid, targets['current'], r)
            overrun = original_target - generated <= 1 and generated < cap
            for name, target in targets.items():
                variants[name]['requests'][rid]['num_output_target_tokens'] = target
                if name in policies:
                    obs = dict(id=rid, estimate=target-generated, actual=actual['usage_completion_tokens']-generated,
                               context=r['num_prompt_tokens']+generated, snapshot=path.name)
                    observations[(model, name, 'all_running_decode')].append(obs)
                    if overrun:
                        observations[(model, name, 'original_one_token_remaining')].append(obs)
            count += 1
        inventory.append(dict(file=path.name, sha256=hashlib.sha256(raw).hexdigest(), running_decode=count,
                              running=state['num_running'], waiting=state['num_waiting'], reserve=reserve))
        if native:
            worker = native.SchedulerSimulationWorker(1., h[9], h[10], h[12], h[13], h[11], h[14])
            for prompt in (128, 4096, 16384):
                results = {}
                for name, variant in variants.items():
                    result = worker.run_simulation_for_test(msgspec.msgpack.encode(variant), prompt, 'prefill_done', (), state['created_at'], 0.)
                    results[name] = result['estimated_wait_ms']
                replay.append(dict(snapshot=path.name, model=model, prompt_tokens=prompt, estimates_ms=results))
        print(json.dumps(inventory[-1]), flush=True)
    output = dict(raw_point_sha256=hashlib.sha256(args.point.read_bytes()).hexdigest(), inventory=inventory,
                  scope='15 periodic snapshots from one completed 8.6 repeat; observations are not independent. '
                        'Only already-running decode requests change. Waiting/prefill requests retain their original targets. '
                        'Oracle uses hindsight lengths and hypothetical probes; it is not measured latency or a deployable estimator. '
                        'Half/double reserves are sensitivity controls, not fitted recommendations. '
                        'Snapshots omit raw engine length predictions; original target is authoritative. '
                        'max(original_target, generated+reserve) implements the rolling floor before caps.',
                  calibration_sha256=hashlib.sha256(args.calibration.read_bytes()).hexdigest() if args.calibration else None,
                  survival_support={model: dict(minimum=min(counts), median=statistics.median(counts), zero=sum(n==0 for n in counts)) for model,counts in support_counts.items()},
                  metrics=[dict(model=m, policy=p, cohort=c, **metrics(rows)) for (m,p,c),rows in observations.items()], replay=replay)
    args.output.write_text(json.dumps(output, indent=2)+'\n')


if __name__ == '__main__':
    main()
