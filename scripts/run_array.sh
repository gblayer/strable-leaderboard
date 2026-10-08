#!/bin/bash
# One SLURM array task per line of tasks.txt: "<model folder> <encoder> <table> <fold>".
#
#   python scripts/make_tasks.py models/tabpfn-2.5 > tasks.txt
#   mkdir -p logs        # SLURM does not create the log folder
#   sbatch -p <partition> --array=0-$(( $(wc -l < tasks.txt) - 1 ))%16 scripts/run_array.sh tasks.txt
#
# Before submitting, on a node with internet: python scripts/prepare.py models/tabpfn-2.5
# (dataset, checkpoints and the model's environment go into $STRABLE_CACHE). Compute nodes run
# offline. Finished tasks are skipped, so the same command resubmits after a preemption.
#SBATCH --job-name=strable
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%A_%a.out
#SBATCH --requeue

set -euo pipefail
: "${STRABLE_CACHE:?set STRABLE_CACHE to the shared cache folder}"
read -r MODEL ENCODER TABLE FOLD < <(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$1")

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
source "$STRABLE_CACHE/envs/$(basename "$MODEL")/bin/activate"
python -m strable_leaderboard.run "$MODEL" --encoder "$ENCODER" --table "$TABLE" --fold "$FOLD" \
  --data-dir "$STRABLE_CACHE/data" --cache-dir "$STRABLE_CACHE" --out runs --n-jobs "$SLURM_CPUS_PER_TASK"
