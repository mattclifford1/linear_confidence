#!/usr/bin/env bash
# The initial extra-experiments run (about 4 h): CPU and GPU chains in parallel,
# then every method over projections-extra/. Resumable: rerun to finish.
#   nohup ./run_initial.sh > logs/initial.log 2>&1 &
set -u
cd "$(dirname "$0")"
mkdir -p logs
METHODS=("Slacks Deltas" "Min Deltas" "F Deltas" "CP Sum" "CP Minimax" "DKW Sum"
         "DKW Minimax" "Gaussian Predictive" "Gaussian Band" "Logistic Band")
FEATURES_DONE=logs/.initial-features-done
rm -f "$FEATURES_DONE"

# stage <log name> <command...>
stage() { local name=$1; shift; echo "[$(date +%H:%M)] $name: $*"
          "$@" > "logs/initial-$name.log" 2>&1 || echo "[$(date +%H:%M)] FAILED: $name"; }

gpu_chain() {
    stage features-gpu uv run python export_torch.py features
    touch "$FEATURES_DONE"
    stage train uv run python export_torch.py train --seeds 3
}

cpu_chain() {
    stage tabular uv run python extra_grid.py tabular --seeds 10 --jobs 24
    stage gaussian uv run python extra_grid.py gaussian --seeds 5 --jobs 24
    stage n-sweep uv run python extra_grid.py n-sweep --seeds 3 --jobs 12
    stage image-medical uv run python extra_grid.py image-fast --datasets MEDICAL_IMAGE \
        --skip-models LinearSVM --seeds 5 --jobs 12
    until [ -e "$FEATURES_DONE" ]; do sleep 30; done
    stage features uv run python extra_grid.py features --seeds 5 --jobs 16
}

gpu_chain & GPU=$!
cpu_chain
wait $GPU

stage methods uv run python run_wide.py --proj projections-extra --seeds 10 --jobs 24 \
    --timeout 300 --max-n "Slacks Deltas=2000" "Min Deltas=12000" \
    --out initial.csv --methods "${METHODS[@]}"
echo "[$(date +%H:%M)] done"
