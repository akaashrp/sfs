"""vLLM-SR percentile selector adaptation; Apache-2.0 upstream attribution.

Pinned source and measurement differences: src/assets/vllm_sr_latency/README.md.
Only valid positive finite observations are accepted (explicit extra guard).
Single event-loop owner; update/select contain no awaits.
"""
from collections import deque
import math
import sys

UPSTREAM_COMMIT = '544ce4f15eb5d04c1efaa597660c4a8ed042e22c'
POLICY = 'vllm_sr_latency'


class LatencyHistory:
    def __init__(self, models, *, ttft_percentile=95, tpot_percentile=90):
        self.models = tuple(models)
        if not self.models or len(set(self.models)) != len(self.models) or any(not m or m.strip()!=m for m in self.models):
            raise ValueError('Unique nonempty normalized candidate models required')
        self.percentiles = {'ttft': ttft_percentile, 'tpot': tpot_percentile}
        if any(type(p) is not int or not 0 <= p <= 100 for p in self.percentiles.values()):
            raise ValueError('Percentiles must be integers in [0,100]')
        self.generation = 0
        self.events = []
        self._state = {}
        self._seen = set()

    def reset(self):
        self.generation += 1
        self._state.clear()
        self._seen.clear()
        self.events.append({'type': 'reset', 'sequence': len(self.events), 'generation': self.generation})

    def update(self, model, metric, value, *, request_id, generation):
        if model not in self.models or metric not in self.percentiles:
            raise ValueError('Unknown latency model/metric')
        key = (request_id, metric)
        reason = ('stale_generation' if generation != self.generation else
                  'duplicate' if key in self._seen else
                  'invalid_value' if isinstance(value, bool) or not isinstance(value, (int,float)) or not math.isfinite(value) or value <= 0 else None)
        event = {'type':'observation', 'sequence':len(self.events), 'generation':generation,
                 'request_id':request_id, 'model':model, 'metric':metric, 'accepted':reason is None,
                 'value':value if isinstance(value,(float,int)) and math.isfinite(value) else None,
                 'reason':reason}
        self.events.append(event)
        if reason: return False
        self._seen.add(key)
        state = self._state.setdefault((model,metric), {'values':deque(maxlen=1000), 'ewma':value, 'count':0})
        if state['count']: state['ewma'] = .3*value + .7*state['ewma']
        state['values'].append(value)
        state['count'] += 1
        return True

    def metric(self, model, metric):
        state = self._state.get((model,metric))
        if state is None: return None
        if len(state['values']) < 3: return state['ewma']
        values = sorted(state['values'])
        index = self.percentiles[metric]/100 * (len(values)-1)
        lower = int(index); upper = min(lower+1,len(values)-1); weight = index-lower
        return values[lower]*(1-weight)+values[upper]*weight

    def select(self):
        enabled = [m for m,p in self.percentiles.items() if p]
        rows = {model:{metric:self.metric(model,metric) for metric in enabled} for model in self.models}
        complete = {model:row for model,row in rows.items() if enabled and all(v is not None for v in row.values())}
        fallback = None
        if not complete:
            selected = self.models[0]; scores = {selected:1.0}
            fallback = 'missing_config' if not enabled else 'no_complete_candidate'
        else:
            best = {m:min(row[m] for row in complete.values()) for m in enabled}
            scores = {model:sum(row[m]/(best[m] if best[m]>0 else 1.) for m in enabled)/len(enabled) for model,row in complete.items()}
            selected = min(scores,key=scores.get)
            # Upstream initializes bestScore to MaxFloat64; overflowing scores
            # never replace it and trigger defaultToFirst.
            if scores[selected] >= sys.float_info.max:
                selected = self.models[0]; scores = {selected:1.0}; fallback = 'scoring_failed'
        result = {'selected_model':selected, 'scores':scores, 'metrics':rows, 'fallback':fallback,
                  'counts':{model:{m:self._state.get((model,m),{}).get('count',0) for m in enabled} for model in self.models},
                  'generation':self.generation, 'sequence':len(self.events)}
        self.events.append({'type':'selection', **result})
        return result

    def require_warm(self):
        if any(self._state.get((model,m),{}).get('count',0)<3 for model in self.models for m,p in self.percentiles.items() if p):
            raise ValueError('Every candidate requires at least three valid observations per enabled metric')

    def metadata(self):
        return {'policy':POLICY, 'upstream_commit':UPSTREAM_COMMIT, 'percentiles':self.percentiles,
                'window_observations':1000, 'ewma_alpha':.3, 'candidate_order':list(self.models),
                'measurement':'client_first_parsed_SSE_chunk_and_total_duration_per_completion_token',
                'transport':'streaming_only_for_this_selector', 'events':self.events}


def audit_history(metadata):
    """Replay the causal event trace; fail closed on config/state mismatches."""
    if (metadata.get('upstream_commit') != UPSTREAM_COMMIT or metadata.get('percentiles') != {'ttft':95,'tpot':90}
            or metadata.get('window_observations') != 1000 or metadata.get('ewma_alpha') != .3):
        raise ValueError('Latency selector configuration differs from frozen contract')
    replay = LatencyHistory(metadata['candidate_order'])
    for event in metadata['events']:
        if event['sequence'] != len(replay.events): raise ValueError('Latency event sequence gap')
        if event['type']=='reset': replay.reset()
        elif event['type']=='observation':
            replay.update(event['model'],event['metric'],event['value'],request_id=event['request_id'],generation=event['generation'])
        elif event['type']=='selection': replay.select()
        else: raise ValueError('Unknown latency event type')
        if replay.events[-1] != event: raise ValueError('Latency event replay mismatch')
    replay.require_warm()
    return {'status':'PASS', 'events':len(replay.events),
            'selections':sum(e['type']=='selection' for e in replay.events)}
