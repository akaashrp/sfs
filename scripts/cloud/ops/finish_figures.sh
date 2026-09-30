#!/usr/bin/env bash
# Post-run: set the published rate grid and re-render. Run only after build_results.py has taken in
# the new cells, so no x position is drawn empty.
#
# 8.75, 9.3 and 9.5 leave the grid: the five rate-fill baselines were never measured there and never
# will be, so keeping them would show four series against a gap. 8.9 and 9.0 come in, which is where
# the new cells are.
set -euo pipefail
F=/ocean/projects/cis250162p/aparthas/sfs_paper_results/figures/make_figures.py
python3 - "$F" <<'PY'
import sys, pathlib
p = pathlib.Path(sys.argv[1])
s = p.read_text()
old = 'FIG5_RATES = [3, 4, 5, 6, 7, 8, 8.3, 8.6, 8.75, 9.2, 9.3, 9.5]'
new = 'FIG5_RATES = [3, 4, 5, 6, 7, 8, 8.3, 8.6, 8.9, 9.0, 9.2]'
if new in s:
    print('already set'); raise SystemExit(0)
assert old in s, 'FIG5_RATES not in the expected form'
p.write_text(s.replace(old, new, 1))
print('FIG5_RATES ->', new)
PY
source /opt/packages/anaconda3-2024.10-1/etc/profile.d/conda.sh
conda activate vllm
python "$F"
echo FIGURES_DONE
