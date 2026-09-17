"""Kind-dispatched campaign overlays on the frozen bundle (baseline, SFS/SCORE, predictor variants, FCFS configuration)."""
from scripts.cloud.common import read


def apply_any_campaign(bundle, campaign, inspect=False):
    """Return the validated in-memory overlay for a campaign document of any supported kind.

    FCFS configuration overlays (matrix or fcfs_sfs_score) are accepted only for read-only inspection
    (control status, collate); a worker applies them through scripts.cloud.fcfs.campaign under --profile fcfs.
    """
    kind = campaign.get('kind')
    if 'configuration_id' in campaign:
        if not inspect:
            raise ValueError('FCFS configuration overlays run only under worker --profile fcfs')
        from scripts.cloud.fcfs.campaign import apply_campaign
        return apply_campaign(bundle, campaign, 'inspect')
    if kind is None:
        from scripts.cloud.baseline_campaign import apply_campaign
        return apply_campaign(bundle, campaign)
    if kind == 'sfs_score':
        from scripts.cloud.sfs_score_campaign import apply_sfs_score_campaign
        return apply_sfs_score_campaign(bundle, campaign)
    if kind == 'predictor_variants':
        from scripts.cloud.variant_campaign import apply_variant_campaign
        return apply_variant_campaign(bundle, campaign)
    raise ValueError(f'Unknown campaign kind: {kind!r}')


def load_campaign(bundle, path, inspect=False):
    return apply_any_campaign(bundle, read(path), inspect)
