#!/bin/bash
# The SLURM job array of the runner. On the login node it submits itself; inside SLURM each array
# task runs one line of the tasks file:
#
#   embed <encoder> <table>                    fill the embedding cache (GPU; run before any model)
#   smoke <models/name>                        strable_leaderboard.run --smoke on the reference hardware
#   <models/name> <encoder> <table> <fold>     one run of strable_leaderboard.run
#
#   python -m strable_leaderboard.embed --list --cache-dir cache > embed.txt && bash scripts/run_array.sh embed.txt
#   echo "smoke models/<name>" > smoke.txt && bash scripts/run_array.sh smoke.txt
#   python scripts/make_tasks.py models/<name> > tasks.txt && bash scripts/run_array.sh tasks.txt
#
# A tasks file holds one kind of line, for one model. Hardware comes from the model card: gpu jobs
# get $STRABLE_SBATCH_GPU (one GPU of the reference type), cpu jobs $STRABLE_SBATCH_CPU (no GPU),
# both 32 CPUs and 128G. The cluster's partition and node flags live in private/runner.env
# ($STRABLE_RUNNER_ENV; gitignored: the repository names no cluster), for example
#   STRABLE_SBATCH_GPU="--partition=<gpu partition> --exclude=<other GPU types> --gres=gpu:1"
#   STRABLE_SBATCH_CPU="--partition=<cpu partition>"
# Extra arguments go to sbatch after these, so they win. The job id is
# printed. Logs go to $STRABLE_LOGS (default logs/), created before submission, since SLURM fails
# silently on a missing output folder. Finished tasks are skipped, so the same file resubmits after
# a preemption. strable_leaderboard.run runs the model code through scripts/sandbox.sh (Singularity,
# network cut, no dataset). Rows go to $STRABLE_RUNS (default runs/).
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --requeue
set -euo pipefail
# SLURM runs a copy of this file from its spool folder, so the clone is passed on in STRABLE_ROOT.
export STRABLE_ROOT=${STRABLE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
ROOT=$STRABLE_ROOT
RUNNER_ENV=${STRABLE_RUNNER_ENV:-$ROOT/private/runner.env}
[ -f "$RUNNER_ENV" ] && source "$RUNNER_ENV"
export STRABLE_CACHE=${STRABLE_CACHE:-$ROOT/cache}
CACHE=$STRABLE_CACHE
RUNS=${STRABLE_RUNS:-$ROOT/runs}
PY=${STRABLE_PYTHON:-$CACHE/env/bin/python}

if [ -z "${SLURM_ARRAY_TASK_ID:-}" ]; then
  TASKS=$(realpath "$1"); shift
  n=$(grep -c . "$TASKS" || true)
  [ "$n" -gt 0 ] || { echo "no tasks in $TASKS" >&2; exit 1; }
  read -r a b _ < "$TASKS"
  hw_of() { "$PY" -c 'import sys, yaml; print(yaml.safe_load(open(sys.argv[1]))["hardware"])' "$ROOT/$1/model.yaml"; }
  case "$a" in
    embed) kind=embed; time=12:00:00; hw=gpu ;;
    smoke) kind=smoke; time=00:30:00; hw=$(hw_of "$b") ;;
    *)     kind=run;   time=12:00:00; hw=$(hw_of "$a") ;;
  esac
  case "$hw" in
    gpu) read -ra where <<< "${STRABLE_SBATCH_GPU:---gres=gpu:1}" ;;
    cpu) read -ra where <<< "${STRABLE_SBATCH_CPU:-}" ;;
    *) echo "unknown hardware '$hw'" >&2; exit 1 ;;
  esac
  LOGS=${STRABLE_LOGS:-$ROOT/logs}
  mkdir -p "$LOGS"
  exec sbatch --parsable --array=0-$((n - 1))%"${STRABLE_PARALLEL:-16}" --job-name="strable-$kind" \
    --time="$time" --output="$LOGS/%A_%a.out" --export=ALL "${where[@]}" "$@" "$0" "$TASKS"
fi

cd "$ROOT"
read -r a b c d < <(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$1")
export PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=$ROOT
DEVICE=cpu; { [ -n "${CUDA_VISIBLE_DEVICES:-}" ] || [ -n "${SLURM_JOB_GPUS:-}" ]; } && DEVICE=cuda
echo "task $SLURM_ARRAY_TASK_ID on $(hostname): $a $b $c $d (device $DEVICE)"

case "$a" in
  embed)
    # less fragmentation: the 8B models in float32 leave little room for a batch of long texts
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "$PY" -m strable_leaderboard.embed --encoder "$b" --table "$c" --cache-dir "$CACHE" --device "$DEVICE" ;;
  smoke)
    # The verdict is written here, by this trusted script.
    name=$(basename "$b"); dir=$CACHE/dispatch/smoke; mkdir -p "$dir"
    set +e
    "$PY" -m strable_leaderboard.run "$ROOT/$b" --smoke --data-dir "$CACHE/data" --cache-dir "$CACHE" \
      --device "$DEVICE" --n-jobs "${SLURM_CPUS_PER_TASK:-32}" 2>&1 | tee "$dir/$name.log"
    rc=${PIPESTATUS[0]}
    set -e
    echo "$rc" > "$dir/$name.rc"
    exit "$rc" ;;
  *)
    "$PY" -m strable_leaderboard.run "$ROOT/$a" --encoder "$b" --table "$c" --fold "$d" --data-dir "$CACHE/data" \
      --cache-dir "$CACHE" --out "$RUNS" --device "$DEVICE" --n-jobs "${SLURM_CPUS_PER_TASK:-32}"
    echo "elapsed ${SECONDS}s on $DEVICE" ;;  # the dispatcher's estimate of the time to results reads it
esac
