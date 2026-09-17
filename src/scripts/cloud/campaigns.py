"""Kind-dispatched campaign overlays on the frozen bundle (baseline, SFS/SCORE, predictor variants, staleness sweep)."""
from scripts.cloud.common import read


def apply_any_campaign(bundle, campaign):
    """Return the validated in-memory overlay for a campaign document of any supported kind."""
    kind = campaign.get('kind')
    if kind is None:
        from scripts.cloud.baseline_campaign import apply_campaign
        return apply_campaign(bundle, campaign)
    if kind == 'sfs_score':
        from scripts.cloud.sfs_score_campaign import apply_sfs_score_campaign
        return apply_sfs_score_campaign(bundle, campaign)
    if kind == 'predictor_variants':
        from scripts.cloud.variant_campaign import apply_variant_campaign
        return apply_variant_campaign(bundle, campaign)
    if kind == 'staleness_sweep':
        from scripts.cloud.staleness_campaign import apply_staleness_campaign
        return apply_staleness_campaign(bundle, campaign)
    raise ValueError(f'Unknown campaign kind: {kind!r}')


def load_campaign(bundle, path):
    return apply_any_campaign(bundle, read(path))
