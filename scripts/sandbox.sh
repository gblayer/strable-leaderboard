#!/bin/bash
# Run a command from a model folder inside Singularity, with the network cut.
#
#   scripts/sandbox.sh <model folder> <writable folder> <command> [args...]
#
# `singularity exec --containall --net --network none` on a small Debian image
# ($STRABLE_CACHE/sandbox/base.sif, pulled once by scripts/prepare.py). The container sees only,
# read-only: the shared env and its interpreter, the dataset, the embedding cache, this model's
# weights, the package code, the folds and the model folder; read-write: the writable folder, and
# an empty /tmp and home on node disk. No real home, no other model, no results, no credentials, no
# network. --nv adds the GPU when the job has one. Without Singularity or its image the command
# does not run: there is no unsandboxed fallback.
set -euo pipefail
MODEL=$(realpath "$1"); OUT=$2; shift 2
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
CACHE=${STRABLE_CACHE:-$ROOT/cache}
IMG=${STRABLE_IMAGE:-$CACHE/sandbox/base.sif}
command -v singularity > /dev/null || { echo "sandbox: singularity not found; refusing to run unsandboxed" >&2; exit 2; }
[ -f "$IMG" ] || { echo "sandbox: no image $IMG; run scripts/prepare.py" >&2; exit 2; }
mkdir -p "$OUT"; OUT=$(realpath "$OUT")
export PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTHONPATH=$ROOT

TMP=$(mktemp -d "${SLURM_TMPDIR:-${TMPDIR:-/tmp}}/strable.XXXXXX")
trap 'rm -rf "$TMP"' EXIT
binds=(--bind "$OUT:$OUT")
for p in "$CACHE/env" "$CACHE/python" "$CACHE/data" "$CACHE/embeddings" "$CACHE/weights/$(basename "$MODEL")" \
         "$ROOT/strable_leaderboard" "$ROOT/splits" "$ROOT/env" "$MODEL"; do
  [ -e "$p" ] && binds+=(--bind "$p:$p:ro")
done
nv=(); { [ -n "${CUDA_VISIBLE_DEVICES:-}" ] || [ -n "${SLURM_JOB_GPUS:-}" ]; } && nv=(--nv)
for v in PYTHONNOUSERSITE HF_HUB_OFFLINE TRANSFORMERS_OFFLINE HF_DATASETS_OFFLINE PYTHONPATH STRABLE_CACHE \
         OMP_NUM_THREADS CUDA_VISIBLE_DEVICES; do
  [ -n "${!v:-}" ] && export "SINGULARITYENV_$v=${!v}"
done
export SINGULARITYENV_XDG_CACHE_HOME=/tmp/.cache SINGULARITYENV_HF_HOME=/tmp/hf
# --workdir puts /tmp, /var/tmp and the (empty) home on node disk instead of a 64 MB tmpfs.
singularity exec --containall --cleanenv --net --network none --workdir "$TMP" "${nv[@]}" "${binds[@]}" --pwd "$OUT" \
  "$IMG" "$@"
