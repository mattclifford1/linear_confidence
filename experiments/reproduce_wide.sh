#!/usr/bin/env bash
# Regenerate the wide grid: 32 datasets x 7 models x 10 seeds x 2 calibration
# modes.
#
#   ./reproduce_wide.sh          # reuse any projections already exported
#   FRESH=1 ./reproduce_wide.sh  # re-export everything from scratch
#
# Two environments are involved on purpose - see export_projections.py for why
# the sibling packages cannot be imported into the deltas env.
#
# Cold: ~30 min to export (the PneumoniaMNIST gradient-boosting fits dominate)
# plus ~65 min for the deltas methods, of which the published slack method is
# roughly 85%.
set -euo pipefail
cd "$(dirname "$0")"

PY=${PY:-python}
EXPORT_PY=${EXPORT_PY:-/home/matt/Repos/projection_models/.venv/bin/python}
JOBS=${JOBS:-8}
SEEDS=${SEEDS:-10}

if [[ ! -x "$EXPORT_PY" ]]; then
  echo "export interpreter not found: $EXPORT_PY" >&2
  echo "expected the projection_models venv (it has toy_datasets too)" >&2
  exit 1
fi

echo "== export environment =="
"$EXPORT_PY" -c "import sys, sklearn, numpy, data_loaders, projection_models; \
print('python', sys.version.split()[0], '| sklearn', sklearn.__version__, \
'| numpy', numpy.__version__)"

echo "== deltas environment =="
$PY -c "import sys, sklearn, deltas.misc.use_two as u; \
print('python', sys.version.split()[0], '| sklearn', sklearn.__version__, \
'| USE_TWO', u.USE_TWO)"

echo "== tests =="
$PY -m pytest ../tests -q

echo "== export projections =="
"$EXPORT_PY" export_projections.py --seeds "$SEEDS" --jobs "$JOBS" \
  ${FRESH:+--force}

echo "== run deltas methods =="
TQDM_DISABLE=1 $PY run_wide.py --seeds "$SEEDS" --jobs "$JOBS"

echo "== analyse =="
$PY analyse_wide.py

echo "== done =="
