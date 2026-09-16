"""Family guard retaining the shared methodology behavior for Ministral.

The previous one-second age gate and ten-second retry loop have been removed
from both families. Transport coherence and version checks remain shared.
"""
from .methodology_scheduler import MethodologyScheduler


class MinistralMethodologyScheduler(MethodologyScheduler):
    def __init__(self, instances, *, snapshot_wait_timeout_s=None,
                 snapshot_retry_interval_s=None, **kwargs):
        if not instances or any(not c.model_id.startswith('ministral3-') for c in instances.values()):
            raise ValueError('Ministral adapter requires only Ministral instances')
        # Legacy keyword arguments are accepted without imposing dispatch waits.
        super().__init__(instances, **kwargs)

    def run_metadata(self):
        return {**super().run_metadata(), 'ministral_freshness_adapter_version': 2,
                'snapshot_wait_timeout_s': None, 'snapshot_freshness_waits': [],
                'snapshot_freshness_semantics': 'shared_latest_coherent_batch; no_age_gate_or_retry_wait'}
