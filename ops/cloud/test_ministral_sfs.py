import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location('ministral_sfs', Path(__file__).with_name('ministral_sfs.py'))
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def test_completed_cell_skips_only_verified_result(tmp_path):
    point = tmp_path/'point.json'
    point.write_text('{}')
    done = tmp_path/'completed.json'
    assert not runner.checked_completed(done, 'bundle', 'runner', {'source':'pin'})
    done.write_text(json.dumps({'point':str(point), 'point_sha256':runner.digest(point),
        'bundle_sha256':'bundle', 'runner_sha256':'runner', 'source_sha256':{'source':'pin'}}))
    assert runner.checked_completed(done, 'bundle', 'runner', {'source':'pin'})
    with pytest.raises(ValueError):
        runner.checked_completed(done, 'changed', 'runner', {'source':'pin'})
    point.write_text('{"changed":true}')
    with pytest.raises(ValueError):
        runner.checked_completed(done, 'bundle', 'runner', {'source':'pin'})
