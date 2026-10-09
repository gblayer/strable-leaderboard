#!/bin/bash
# Run the model's child process inside Singularity, with the network cut.
#
#   scripts/sandbox.sh <model folder> <inputs folder> <outputs folder> <command> [args...]
#
# Called by strable_leaderboard.run, the trusted parent, which writes X_train, y_train and X_test
# into the inputs folder and reads the predictions back from the outputs folder.
# `singularity exec --containall --net --network none` on a small Debian image
# ($STRABLE_CACHE/sandbox/base.sif, pulled once by scripts/prepare.py). The container sees only,
# read-only: the shared env and its interpreter, this model's weights, the package code, the model
# folder and the inputs; read-write: the outputs folder, and an empty /tmp and home on node disk.
# No dataset, no folds, no embeddings, no results, no real home, no credentials, no other model,
# no network. --nv adds the GPU when the job has one. Without Singularity or its image the command
# does not run: there is no unsandboxed fallback.
set -euo pipefail
MODEL=$(realpath "$1"); IN=$(realpath "$2"); OUT=$(realpath "$3"); shift 3
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
CACHE=${STRABLE_CACHE:-$ROOT/cache}
IMG=${STRABLE_IMAGE:-$CACHE/sandbox/base.sif}
command -v singularity > /dev/null || { echo "sandbox: singularity not found; refusing to run unsandboxed" >&2; exit 2; }
[ -f "$IMG" ] || { echo "sandbox: no image $IMG; run scripts/prepare.py" >&2; exit 2; }
export PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTHONPATH=$ROOT

TMP=$(mktemp -d "${SLURM_TMPDIR:-${TMPDIR:-/tmp}}/strable.XXXXXX")
trap 'rm -rf "$TMP"' EXIT
binds=(--bind "$OUT:$OUT")
for p in "$CACHE/env" "$CACHE/python" "$CACHE/weights/$(basename "$MODEL")" "$ROOT/strable_leaderboard" "$MODEL" "$IN"; do
  [ -e "$p" ] && binds+=(--bind "$p:$p:ro")
done
nv=(); { [ -n "${CUDA_VISIBLE_DEVICES:-}" ] || [ -n "${SLURM_JOB_GPUS:-}" ]; } && nv=(--nv)
for v in PYTHONNOUSERSITE HF_HUB_OFFLINE TRANSFORMERS_OFFLINE HF_DATASETS_OFFLINE PYTHONPATH OMP_NUM_THREADS \
         CUDA_VISIBLE_DEVICES; do
  [ -n "${!v:-}" ] && export "SINGULARITYENV_$v=${!v}"
done
export SINGULARITYENV_XDG_CACHE_HOME=/tmp/.cache SINGULARITYENV_HF_HOME=/tmp/hf
# --workdir puts /tmp, /var/tmp and the (empty) home on node disk instead of a 64 MB tmpfs.
singularity exec --containall --cleanenv --net --network none --workdir "$TMP" "${nv[@]}" "${binds[@]}" --pwd "$OUT" \
  "$IMG" "$@"
