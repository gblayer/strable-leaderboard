#!/bin/bash
# Run the model's child process inside Singularity, with the network cut (kept for an API model only).
#
#   scripts/sandbox.sh <model folder> <inputs folder> <outputs folder> <command> [args...]
#
# Called by strable_leaderboard.run, the trusted parent, which writes X_train, y_train and X_test
# into the inputs folder and reads the predictions back from the outputs folder.
# `singularity exec --containall --net --network none` on a Debian image with gcc and the C library
# headers ($STRABLE_CACHE/sandbox/base.sif, pulled by scripts/prepare.py; $STRABLE_IMAGE overrides
# it): Triton, which some packages run, compiles a small C launcher with the Python headers of the
# bound interpreter when its first kernel runs. The container sees only,
# read-only: the shared env and its interpreter, this model's weights, the package code, the model
# folder and the inputs; read-write: the outputs folder, and an empty /tmp and home on node disk.
# No dataset, no folds, no embeddings, no results, no real home, no credentials, no other model,
# no network. --nv adds the GPU when the job has one. Without Singularity or its image the command
# does not run: there is no unsandboxed fallback.
#
# An API model (hardware: api) runs on its provider's servers. For its child only, the parent sets
# STRABLE_SANDBOX_NETWORK=1 and the endpoint, key and auth scheme of its key file (STRABLE_API_URL,
# STRABLE_API_KEY, STRABLE_API_AUTH), and the container keeps the host's network. The trade-off of
# this simplest working version: that child can reach any host, not only its endpoint. It still sees
# only the training rows, the test rows without their labels and its own key, and only the maintainer
# adds API models. A possible later step: allow only the endpoint's host (a proxy or a firewall rule).
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
net=(--net --network none); [ "${STRABLE_SANDBOX_NETWORK:-}" = 1 ] && net=()
# The thread caps are set by strable_leaderboard.run to its --n-jobs.
for v in PYTHONNOUSERSITE HF_HUB_OFFLINE TRANSFORMERS_OFFLINE HF_DATASETS_OFFLINE PYTHONPATH OMP_NUM_THREADS \
         MKL_NUM_THREADS OPENBLAS_NUM_THREADS NUMEXPR_NUM_THREADS CUDA_VISIBLE_DEVICES \
         STRABLE_API_URL STRABLE_API_KEY STRABLE_API_AUTH; do
  [ -n "${!v:-}" ] && export "SINGULARITYENV_$v=${!v}"
done
export SINGULARITYENV_XDG_CACHE_HOME=/tmp/.cache SINGULARITYENV_HF_HOME=/tmp/hf
# --workdir puts /tmp, /var/tmp and the (empty) home on node disk instead of a 64 MB tmpfs.
singularity --quiet exec --containall --cleanenv "${net[@]}" --workdir "$TMP" "${nv[@]}" "${binds[@]}" --pwd "$OUT" \
  "$IMG" "$@"
