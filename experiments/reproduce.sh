#!/usr/bin/env bash
# Regenerate every result and figure in one go.
#
#   ./reproduce.sh            # uses the cached classifiers (fast)
#   FRESH=1 ./reproduce.sh    # wipes the cache first (slow: retrains MIMIC)
#
# With a warm cache this takes a few minutes; cold it is ~1 hour, essentially
# all of it training the MIMIC-III MLPs.
set -euo pipefail
cd "$(dirname "$0")"

RUN=${RUN:-uv run}
PY="$RUN python"

if [[ "${FRESH:-0}" == "1" ]]; then
  echo "== wiping the cache =="
  $PY -c "import deltas.utils.cache as c; print('cleared', c.clear())"
fi

echo "== config =="
$PY -c "import deltas.misc.use_two as u; print('USE_TWO', u.USE_TWO, '| USE_GLOBAL_R', u.USE_GLOBAL_R)"

echo "== tests =="
$RUN pytest ../tests -q

echo "== experiments =="
$PY run_experiments.py --seeds 10

echo "== combined table =="
$PY combine_tables.py > /dev/null
echo "  -> results/combined_table-groundup.txt"

echo "== significance =="
$PY significance.py | tail -25

echo "== bound validation =="
$PY validate_bounds.py | tail -12

echo "== figures =="
rm -f results/overlap_sweep.joblib      # the sweep depends on the estimators
$PY make_figures.py

echo "== done =="
