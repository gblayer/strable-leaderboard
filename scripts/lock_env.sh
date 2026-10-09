#!/bin/bash
# Regenerate env/requirements.txt, the shared environment, from env/requirements.in with the
# paper's full freeze (env/paper-freeze.txt) as constraints, so every dependency also keeps the
# paper's version. Needs uv. The maintainer runs it after merging a pull request that adds a
# package to env/requirements.in; a constraint conflict means the new package needs another
# version of an existing pin, which only the maintainer can decide.
#
#   bash scripts/lock_env.sh
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
out=$(mktemp)
trap 'rm -f "$out"' EXIT
uv pip compile env/requirements.in -c env/paper-freeze.txt --python-version 3.12 \
  --python-platform x86_64-manylinux_2_28 --no-header --no-annotate --quiet -o "$out"
{
  echo "# The shared environment, every package pinned to the version of the STRABLE paper environment"
  echo "# (salts_env, 2026-03-23, Python 3.12.12). Generated from env/requirements.in; do not edit by hand."
  echo "# scripts/prepare.py installs it into cache/env together with this package (-e .)."
  cat "$out"
} > env/requirements.txt
