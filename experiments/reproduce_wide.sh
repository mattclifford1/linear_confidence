#!/usr/bin/env bash
# Regenerate the wide grid: 32 datasets x 7 models x 10 seeds x 2 calibration
# modes.
#
#   ./reproduce_wide.sh          # reuse any projections already exported
#   FRESH=1 ./reproduce_wide.sh  # re-export everything from scratch
#
# Single environment since the sklearn 1.3.2 pin was removed - `uv sync` from
# the repo root sets it up, including the sibling packages as editable path
# dependencies.
#
# Cold: ~30 min to export (the PneumoniaMNIST gradient-boosting fits dominate)
# plus ~65 min for the deltas methods, of which the published slack method is
# roughly 85%.
set -euo pipefail
cd "$(dirname "$0")"

RUN=${RUN:-uv run}
JOBS=${JOBS:-8}
SEEDS=${SEEDS:-10}

echo "== environment =="
$RUN python -c "import sys, sklearn, numpy, data_loaders, projection_models; \
import deltas.misc.use_two as u; \
print('python', sys.version.split()[0], '| sklearn', sklearn.__version__, \
'| numpy', numpy.__version__, '| USE_TWO', u.USE_TWO)"

echo "== tests =="
$RUN pytest ../tests -q

echo "== export projections =="
$RUN python export_projections.py --seeds "$SEEDS" --jobs "$JOBS" \
  ${FRESH:+--force}

echo "== run deltas methods =="
TQDM_DISABLE=1 $RUN python run_wide.py --seeds "$SEEDS" --jobs "$JOBS"

echo "== analyse =="
$RUN python analyse_wide.py

echo "== done =="
