"""Kind-dispatched campaign overlays on the frozen bundle (baseline, SFS/SCORE, predictor variants, staleness sweep, SCORE lambda probe)."""
import math

from scripts.cloud.common import read


def tuned_score_lambda(campaign):
    """An overlay-wide SCORE routing multiplier and the sweep that chose it, or (None, None).

    SCORE's lambda is the multiplier of its own constraint formulation, so tuning it is part of
    running that published method faithfully.  It never reaches the campaign objective: cells still
    carry the bundle's --lambda-weight, so SCORE is scored on the same OnTimeUtility as every other
    policy.  A tuned value must cite the sweep it came from, which keeps the choice auditable and
    stops a multiplier from drifting in silently.
    """
    weight = campaign.get('score_lambda_weight')
    if weight is None:
        if campaign.get('score_lambda_evidence') is not None:
            raise ValueError('score_lambda_evidence without a score_lambda_weight')
        return None, None
    weight = float(weight)
    if not math.isfinite(weight) or weight <= 0:
        raise ValueError('score_lambda_weight must be finite and positive')
    evidence = campaign.get('score_lambda_evidence')
    if not isinstance(evidence, dict) or not evidence.get('sweep'):
        raise ValueError('A tuned score_lambda_weight must cite the sweep that chose it')
    return weight, dict(evidence)


def apply_tuned_score_lambda(result, campaign):
    """Carry a validated tuned multiplier onto an applied overlay that actually runs SCORE cells."""
    weight, evidence = tuned_score_lambda(campaign)
    if weight is None:
        return result
    if not any(str(cell.get('policy')) == 'score' for cell in result.get('cells', ())):
        raise ValueError('score_lambda_weight is only meaningful for an overlay with SCORE cells')
    result['score_lambda_weight'] = weight
    result['score_lambda_evidence'] = evidence
    return result


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
    if kind == 'delta_sweep':
        from scripts.cloud.delta_sweep_campaign import apply_delta_sweep_campaign
        return apply_delta_sweep_campaign(bundle, campaign)
    if kind == 'length_robustness':
        from scripts.cloud.length_robustness_campaign import apply_length_robustness_campaign
        return apply_length_robustness_campaign(bundle, campaign)
    if kind == 'staleness_sweep':
        from scripts.cloud.staleness_campaign import apply_staleness_campaign
        return apply_staleness_campaign(bundle, campaign)
    if kind == 'score_lambda_sweep':
        from scripts.cloud.score_lambda_campaign import apply_score_lambda_campaign
        return apply_score_lambda_campaign(bundle, campaign)
    raise ValueError(f'Unknown campaign kind: {kind!r}')


def load_campaign(bundle, path):
    return apply_any_campaign(bundle, read(path))
