"""Additive Ministral selector adapter; the Qwen policy tuple and transport stay intact.

Ministral uses a public served alias and a distinct canonical predictor key.
Only the explicitly configured alias is normalized at this boundary. Unknown
models still fail the shared selector's strict response-attribution check.
"""
from dataclasses import asdict
import json
from pathlib import Path

from scripts.runs.ministral3_methodology_stage import POLICIES as LEGACY_POLICIES
from scripts.runs.ministral3_methodology_stage import smoke_requests
from sfs_core.routing.latency_stream import field

POLICIES = (*LEGACY_POLICIES, "vllm_sr_latency")
ALIASES = {f"ministral3-{size}b": f"ministral3-{size}b-instruct" for size in (3, 8, 14)}
QPS = (6.0125, 7.8625, 8.7875, 9.7125)


class AliasStream:
    def __init__(self, stream, canonical, alias):
        self.stream, self.canonical, self.alias = stream, canonical, alias
        self.iterator = stream.__aiter__()

    def __aiter__(self):
        return self

    async def __anext__(self):
        chunk = await self.iterator.__anext__()
        model = field(chunk, "model")
        if model:
            if model != self.alias:
                raise ValueError("Ministral response differs from the configured served alias")
            if isinstance(chunk, dict):
                return {**chunk, "model": self.canonical}
            return chunk.model_copy(update={"model": self.canonical})
        return chunk

    async def close(self):
        await self.stream.close()


class AliasClient:
    def __init__(self, client):
        self.client = client
        if client.model_id not in ALIASES:
            raise ValueError("Ministral adapter requires a canonical Ministral candidate")

    def __getattr__(self, name):
        return getattr(self.client, name)

    async def submit_request(self, **kwargs):
        response = await self.client.submit_request(**kwargs)
        if kwargs.get("stream"):
            return AliasStream(response, self.model_id, ALIASES[self.model_id])
        return response


def selector_clients(instances):
    return {key: AliasClient(client) for key, client in instances.items()}


def write_warmup(requests, path):
    from sfs_core.routing.latency_warmup import load_warmup
    payload = {"data_role": "calibration", "requests": [asdict(r) for r in smoke_requests(requests, per_bucket=8)]}
    with Path(path).open("x") as stream:
        json.dump(payload, stream)
    load_warmup(path)
    return payload


async def run_selector(*, args, requests, instances, **kwargs):
    from scripts.runs.experiments import run_router_experiment
    if args.utilities != ["vllm_sr_latency"]:
        raise ValueError("Ministral alias normalization is restricted to the selector")
    result = await run_router_experiment(args=args, requests=requests,
        instances=selector_clients(instances), **kwargs)
    for run in result["runs"]:
        run["methodology_config"]["served_model_aliases"] = ALIASES
        run["methodology_config"]["response_model_normalization"] = "strict configured alias to canonical Ministral key"
    return result


def main():
    """Run the additive four-cell extension against an existing Ministral pool."""
    import argparse
    import asyncio
    from scripts.cloud.common import digest, read, write, source_hashes, set_option
    from scripts.cloud.worker import run_point, audit_cell
    from scripts.runs import ministral3_figure5 as legacy
    from scripts.runs import experiments as exp
    from scripts.runs.experiments_sweep import _parse_experiment_args
    from scripts.runs.ministral3_methodology_stage import audit_run, wait_drained
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=['prepare', 'smoke', 'sweep'])
    p.add_argument('--manifest', required=True, type=Path)
    p.add_argument('--calibration-requests', type=Path)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--instances-config', type=Path)
    p.add_argument('--wait-log', action='append', default=[])
    p.add_argument('--smoke-dir', type=Path)
    o = p.parse_args()
    if o.mode == 'prepare':
        if not o.calibration_requests: p.error('Preparation requires the existing calibration request file')
        base = legacy.load_manifest(o.manifest)
        if base['loads']['qps_values'] != list(QPS): raise ValueError('Unexpected Ministral QPS grid')
        prepared = Path(read(Path(base['stage_dir'])/'stage_started.json')['prepared_dir'])
        meta = read(prepared/'metadata.json')
        if (meta.get('data_role') != 'calibration' or meta.get('holdout_start_index') != 0
                or meta.get('requests_sha256') != digest(o.calibration_requests)):
            raise ValueError('Warm-up must use the measured stage calibration pool')
        requests = [exp.ExperimentRequest(**r) for r in (json.loads(line) for line in o.calibration_requests.read_text().splitlines())]
        if len(requests) != 10000: raise ValueError('Expected the full calibration pool')
        o.output.mkdir(parents=True, exist_ok=False)
        warmup = o.output.resolve()/'warmup.json'; write_warmup(requests, warmup)
        files = {**base['file_sha256'], str(o.manifest.resolve()):digest(o.manifest),
                 str(o.calibration_requests.resolve()):digest(o.calibration_requests), str(warmup):digest(warmup)}
        write(o.output/'manifest.json', {'schema_version':1, 'family':'ministral', 'policy':'vllm_sr_latency',
            'qps_values':list(QPS), 'requests_per_cell':8000, 'matrix_cells':4,
            'experiment_argv':set_option(base['experiment_argv'],'--latency-warmup-requests',warmup),
            'calibration_requests':str(o.calibration_requests.resolve()), 'file_sha256':files,
            'source_sha256':source_hashes(), 'legacy_manifest':str(o.manifest.resolve()),
            'legacy_32_cells_unchanged':True, 'gpu_smoke_passed':False})
        return
    manifest = read(o.manifest)
    if manifest.get('policy') != 'vllm_sr_latency' or manifest.get('matrix_cells') != 4 or manifest.get('requests_per_cell') != 8000:
        raise ValueError('Expected the additive four-cell selector manifest')
    for name, expected in manifest['file_sha256'].items():
        if digest(name) != expected: raise ValueError('Frozen Ministral selector input changed')
    if manifest['source_sha256'] != source_hashes(): raise ValueError('Source changed since preparation')
    if not o.instances_config or len(o.wait_log) != 3: p.error('Supply the existing pool config and all three wait logs')
    if o.mode == 'sweep':
        if not o.smoke_dir: p.error('Sweep requires a matching selector smoke directory')
        gate = read(o.smoke_dir/'audit.json')
        if gate['manifest_sha256'] != digest(o.manifest) or gate['source_sha256'] != source_hashes() or gate['point_sha256'] != digest(o.smoke_dir/'smoke/point.json'):
            raise ValueError('Missing/stale Ministral selector GPU smoke')
        audit_run(read(o.smoke_dir/'smoke/point.json')['router']['runs'][0],192)
    o.output.mkdir(parents=True,exist_ok=False)
    async def run():
        from sfs_core.shared.shared_experiment_helpers import warm_up_instances
        from scripts.prep.prepare_methodology_service import PROFILE
        clients,costs,metadata=exp.load_instances(o.instances_config)
        if metadata.get('serving_profile') != PROFILE or {c.model_id for c in clients.values()} != set(ALIASES):
            raise ValueError('Wrong Ministral serving pool')
        try:
            await warm_up_instances(list(clients.values()));await wait_drained(clients)
            args=_parse_experiment_args(manifest['experiment_argv'])
            args.utilities=['vllm_sr_latency'];args.per_request_wait_log=o.wait_log
            if o.mode == 'smoke':
                requests=smoke_requests([exp.ExperimentRequest(**json.loads(line)) for line in Path(manifest['calibration_requests']).read_text().splitlines()])
                args.num_requests=192;args.request_rate_qps=2.
                payload=await run_point('ministral',args,requests,clients,costs,metadata,o.output/'smoke')
                audit_run(payload['router']['runs'][0],192)
                write(o.output/'audit.json',{'status':'PASS_GPU_SMOKE','source_sha256':source_hashes(),
                    'manifest_sha256':digest(o.manifest),'point_sha256':digest(o.output/'smoke/point.json')})
            else:
                requests,_,_=exp._build_request_set(args)
                for rate in QPS:
                    args.request_rate_qps=rate
                    payload=await run_point('ministral',args,requests,clients,costs,metadata,o.output/f'{rate:g}')
                    audit_cell(payload,{'policy':'vllm_sr_latency','qps':rate,'requests':8000})
                write(o.output/'audit.json',{'status':'PASS_FOUR_CELLS','source_sha256':source_hashes(), 'manifest_sha256':digest(o.manifest)})
        finally:
            for client in clients.values():client.close()
    asyncio.run(run())


if __name__ == '__main__': main()
