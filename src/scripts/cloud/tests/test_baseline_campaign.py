from copy import deepcopy
import pytest
from scripts.cloud.baseline_campaign import apply_campaign, RATES
from scripts.cloud.common import ROOT, read


def test_active_and_fallback_campaign_preserve_bundled_artifacts():
    campaign = read(ROOT/'scripts/cloud/baseline-campaign-20260916.json')
    bundle = {'families': {f: {'qps': [99], 'policies': ['hard'], 'requests': n} for f,n in [('qwen',16000),('ministral',8000)]},
              'files': {'quality': 'immutable'}, 'cells': []}
    saved = deepcopy(bundle)
    active = apply_campaign(bundle, campaign)
    assert bundle == saved and active['files'] == saved['files']
    # 4 Qwen policies x 4 rates x 16,000 plus 4 Ministral policies x 4 rates x 8,000; the Ministral
    # grid gained 8.1 QPS on 18 September 2026 (scripts/cloud/baseline_campaign.RATES).
    assert len(active['cells']) == 32
    assert active['requests_total'] == 384000
    assert not any(c['policy'] in ('score','hard','hard_prefill_tps') for c in active['cells'])
    assert len(campaign['fallback_cells']) == 7
    assert all(c['qps'] in RATES[c['family']] for c in campaign['fallback_cells'])
    bad = deepcopy(campaign); bad['cells'][-1]['qps'] = 9.7125
    with pytest.raises(ValueError): apply_campaign(bundle, bad)
    bad = deepcopy(campaign); bad['cells'][0]['policy'] = 'score'
    with pytest.raises(ValueError): apply_campaign(bundle, bad)


def test_accepted_prior_source_digests_are_carried_and_validated():
    from scripts.cloud.common import source_digest
    from scripts.cloud.worker import completed_source_accepted
    campaign = read(ROOT/'scripts/cloud/baseline-campaign-20260916.json')
    bundle = {'families': {f: {'qps': [99], 'policies': ['hard'], 'requests': n} for f,n in [('qwen',16000),('ministral',8000)]},
              'files': {'quality': 'immutable'}, 'cells': []}
    active = apply_campaign(bundle, campaign)
    assert active['accepted_prior_source_digests'] == campaign['accepted_prior_source_digests']
    assert all(len(k) == 64 and v for k, v in active['accepted_prior_source_digests'].items())
    current = {'src/a.py': 'x'}; prior = {'src/a.py': 'y'}
    assert completed_source_accepted({'source_sha256': current}, current, active)
    assert not completed_source_accepted({'source_sha256': prior}, current, active)
    accepted = dict(active, accepted_prior_source_digests={source_digest(prior): 'reviewed'})
    assert completed_source_accepted({'source_sha256': prior}, current, accepted)
    bad = deepcopy(campaign); bad['accepted_prior_source_digests'] = {'short': 'reason'}
    with pytest.raises(ValueError): apply_campaign(bundle, bad)
    bad = deepcopy(campaign); bad['accepted_prior_source_digests'] = {'a'*64: ''}
    with pytest.raises(ValueError): apply_campaign(bundle, bad)
