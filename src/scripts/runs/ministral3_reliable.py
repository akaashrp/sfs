"""Explicit Ministral entry point; never installed in the Qwen path."""
from unittest.mock import patch
from sfs_core.routing.ministral_methodology_scheduler import MinistralMethodologyScheduler


async def run_router_experiment(*, args, requests, instances, _original=None, **kwargs):
    from scripts.runs.experiments import run_router_experiment as default_original
    original = _original or default_original
    if not instances or any(not c.model_id.startswith('ministral3-') for c in instances.values()):
        raise ValueError('Ministral reliability entry point requires a Ministral pool')
    # Pools execute one experiment at a time in separate processes. The scoped
    # replacement reaches the existing function-local scheduler import and is
    # restored on success, failure, or cancellation.
    with patch('sfs_core.routing.methodology_scheduler.MethodologyScheduler', MinistralMethodologyScheduler):
        return await original(args=args, requests=requests, instances=instances, **kwargs)


def main():
    from scripts.runs import experiments, experiments_sweep
    original = experiments.run_router_experiment
    async def run(**kwargs):
        return await run_router_experiment(_original=original, **kwargs)
    with patch.object(experiments, 'run_router_experiment', run):
        experiments_sweep.main()


if __name__ == '__main__':
    main()
