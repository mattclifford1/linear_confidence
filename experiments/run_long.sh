#!/usr/bin/env bash
# The long extra-experiments run (about 20 h; see the plan paper's "Long run"):
# every stage to its final seed count, in priority order, then every method.
# Every export is saved after each seed and the methods after each file, so a
# rerun picks up where a stop left off.
#   setsid nohup ./run_long.sh >> logs/long.log 2>&1 &        # start or resume
#   setsid nohup ./run_long.sh L2 L3 >> logs/long.log 2>&1 &  # chosen stages, then methods
#   kill -- -$(cat logs/long.pid)                             # stop everything
set -u
cd "$(dirname "$0")"
mkdir -p logs
echo $$ > logs/long.pid
STAGES=${*:-L1 L2 L3 L4 L5 L6}
METHODS=("Slacks Deltas" "Min Deltas" "F Deltas" "CP Sum" "CP Minimax" "DKW Sum"
         "DKW Minimax" "Gaussian Predictive" "Gaussian Band" "Logistic Band")

# stage <log name> <command...>
stage() { local name=$1; shift; echo "[$(date '+%a %H:%M')] $name: $*"
          "$@" >> "logs/long-$name.log" 2>&1 || echo "[$(date '+%a %H:%M')] FAILED: $name"; }
want() { [[ " $STAGES " == *" $1 "* ]]; }

gpu_chain() {
    stage features-gpu uv run python export_torch.py features
    stage train uv run python export_torch.py train --seeds 10
}

cpu_chain() {
    if want L1; then
        stage gaussian uv run python extra_grid.py gaussian --seeds 10 --jobs 24
        stage n-sweep uv run python extra_grid.py n-sweep --seeds 10 --jobs 12
    fi
    want L2 && stage image-fast uv run python extra_grid.py image-fast \
        --skip-models LinearSVM --seeds 10 --jobs 12
    want L3 && stage image-slow uv run python extra_grid.py image-slow \
        --models SVM-rbf RandomForest MLP MLP-deep --seeds 10 --jobs 12
    want L4 && stage features uv run python extra_grid.py features --seeds 10 --jobs 16
    want L5 && stage image-slow-extra uv run python extra_grid.py image-slow \
        --models GradientBoosting SVM-linear --seeds 3 --jobs 12
    want L6 && stage linearsvm uv run python extra_grid.py image-fast \
        --models LinearSVM --seeds 3 --jobs 12
    stage tabular uv run python extra_grid.py tabular --seeds 10 --jobs 24
}

gpu_chain & GPU=$!
cpu_chain
wait $GPU

stage methods uv run python run_wide.py --proj projections-extra --seeds 10 --jobs 24 \
    --timeout 300 --max-n "Slacks Deltas=2000" "Min Deltas=12000" \
    --out extra.csv --methods "${METHODS[@]}"
echo "[$(date '+%a %H:%M')] done"
