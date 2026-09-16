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
    assert len(active['cells']) == 28
    assert active['requests_total'] == 352000
    assert not any(c['policy'] in ('score','hard','hard_prefill_tps') for c in active['cells'])
    assert len(campaign['fallback_cells']) == 7
    assert all(c['qps'] in RATES[c['family']] for c in campaign['fallback_cells'])
    bad = deepcopy(campaign); bad['cells'][-1]['qps'] = 9.7125
    with pytest.raises(ValueError): apply_campaign(bundle, bad)
    bad = deepcopy(campaign); bad['cells'][0]['policy'] = 'score'
    with pytest.raises(ValueError): apply_campaign(bundle, bad)
