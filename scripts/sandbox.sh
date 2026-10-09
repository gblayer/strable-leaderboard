#!/bin/bash
# Run a command from a model folder with the network cut and, by default, inside Singularity.
#
#   scripts/sandbox.sh <model folder> <writable folder> <command> [args...]
#
# STRABLE_SANDBOX=singularity (default): `singularity exec --containall --net --network none`
#   on a small Debian image ($STRABLE_CACHE/sandbox/base.sif, pulled once by scripts/prepare.py).
#   The container sees only, read-only: the shared env and its interpreter, the dataset, the
#   embedding cache, this model's weights, the package code, the folds and the model folder;
#   read-write: the writable folder, and an empty /tmp and home on node disk. No real home, no
#   other model, no results, no credentials, no network. --nv adds the GPU when the job has one.
# STRABLE_SANDBOX=unshare: `unshare --user --net --map-root-user`, network cut only; the files
#   stay visible with the user's rights. The fallback if Singularity is unavailable.
set -euo pipefail
MODEL=$(realpath "$1"); OUT=$2; shift 2
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
CACHE=${STRABLE_CACHE:-$ROOT/cache}
mkdir -p "$OUT"; OUT=$(realpath "$OUT")
export PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTHONPATH=$ROOT

case "${STRABLE_SANDBOX:-singularity}" in
  singularity)
    IMG=${STRABLE_IMAGE:-$CACHE/sandbox/base.sif}
    [ -f "$IMG" ] || { echo "sandbox: no image $IMG; run scripts/prepare.py" >&2; exit 2; }
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
    ;;
  unshare)
    cd "$OUT"
    exec unshare --user --net --map-root-user "$@"
    ;;
  *)
    echo "sandbox: STRABLE_SANDBOX must be singularity or unshare" >&2; exit 2 ;;
esac
