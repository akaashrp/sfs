import os
from pathlib import Path
import subprocess

import pytest

from scripts.runs.frozen_slurm import read_settings, render


def test_empty_slurm_environment_preserves_literal_settings_and_directives(tmp_path):
    source = tmp_path/"original.sh"
    source.write_text('#!/bin/bash\n#SBATCH --gres=gpu:h100-80:4\nset -eu\n'
                      ': "${SFS_ROOT:?Set the active worktree}"\n'
                      '[[ "$VALIDATE_ONLY" == 1 ]]\n'
                      '[[ "$SLURM_EXPORT_ENV" == ALL ]]\n'
                      'printf "%s" "$PAYLOAD"\n')
    empty = {"PATH": os.environ["PATH"], "SBATCH_EXPORT": "NONE", "SFS_PREFLIGHT_ONLY": "1"}
    failed = subprocess.run(["bash", str(source)], env=empty, capture_output=True, text=True)
    assert failed.returncode != 0 and "Set the active worktree" in failed.stderr
    payload = "literal $(touch forbidden) `whoami` $HOME 'quoted'"
    batch = tmp_path/"frozen.sbatch"
    batch.write_text(render(source, {"SFS_ROOT": str(tmp_path), "PAYLOAD": payload}))
    assert batch.read_text().index("#SBATCH") < batch.read_text().index("set -euo")
    result = subprocess.run(["bash", str(batch)], env=empty, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout == payload
    source.write_text(source.read_text()+"# drift\n")
    assert subprocess.run(["bash", str(batch)], env=empty, capture_output=True).returncode != 0


def test_production_does_not_inherit_validate_only(tmp_path):
    source = tmp_path/"source.sh"
    source.write_text('#!/bin/bash\n[[ ! -v VALIDATE_ONLY ]]\n')
    batch = tmp_path/"frozen.sh"
    batch.write_text(render(source, {"SFS_ROOT": str(tmp_path)}))
    assert subprocess.run(["bash", str(batch)], env={"PATH":os.environ["PATH"], "VALIDATE_ONLY":"1"}).returncode == 0


@pytest.mark.parametrize("value", ["export OTHER=x\n", "export SFS_ROOT=a b\n", "source elsewhere\n", "export SFS_ROOT=a\nexport SFS_ROOT=b\n"])
def test_invalid_settings_fail_before_submission(tmp_path, value):
    path = tmp_path/"settings.sh"
    path.write_text(value)
    with pytest.raises(ValueError):
        read_settings(path)
