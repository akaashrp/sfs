"""Kind-dispatched campaign overlays on the frozen bundle (baseline, SFS/SCORE, predictor variants, FCFS configuration, staleness sweep)."""
from scripts.cloud.common import read


def apply_any_campaign(bundle, campaign, inspect=False):
    """Return the validated in-memory overlay for a campaign document of any supported kind.

    Serving-configuration overlays (the FCFS matrix, fcfs_sfs_score and serving_config kinds, all carrying a
    configuration_id) are accepted only for read-only inspection (control status, collate); a worker applies
    them through scripts.cloud.serving.campaign under the matching --profile.
    """
    kind = campaign.get('kind')
    if 'configuration_id' in campaign:
        from scripts.cloud.serving.profiles import for_configuration
        name = for_configuration(campaign['configuration_id']).name
        if not inspect:
            raise ValueError(f'Serving configuration overlays run only under worker --profile {name}')
        from scripts.cloud.serving.campaign import apply_profile_campaign
        return apply_profile_campaign(name, bundle, campaign, 'inspect')
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


def load_campaign(bundle, path, inspect=False):
    return apply_any_campaign(bundle, read(path), inspect)
