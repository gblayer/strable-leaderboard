#!/bin/bash
# Regenerate env/requirements.txt, the shared environment, from env/requirements.in with the
# paper's full freeze (env/paper-freeze.txt) as constraints, so every dependency also keeps the
# paper's version. Needs uv and python3. The maintainer runs it after merging a pull request that
# adds a package to env/requirements.in; a constraint conflict means the new package needs another
# version of an existing pin, which only the maintainer can decide. The uv command lives in
# strable_leaderboard/check_requirements.py, which also reports what a pull request would change.
#
#   bash scripts/lock_env.sh
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
python3 -m strable_leaderboard.check_requirements --lock
