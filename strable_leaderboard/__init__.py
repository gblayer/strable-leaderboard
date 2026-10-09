"""STRABLE leaderboard: evaluation protocol, splits, metrics and the model contract."""

import re

__version__ = "0.0.1"

# The data every result is computed on. Change the revision only with a minor version bump.
HF_DATASET = "inria-soda/STRABLE-benchmark"
HF_REVISION = "335c0a77595d4a9ddd86f46dc9c3b414f63a09a8"  # 2026-06-11

# The repository, shared by the gate, the dispatcher and the checks.
REPO = "gblayer/strable-leaderboard"
MAINTAINER = "gblayer"
APPROVED, NEEDS = "claude-approved", "needs-maintainer"
SLUG = re.compile(r"^[a-z0-9][a-z0-9.-]*$")  # a model folder name
# Folder names of the paper's learners: only the maintainer adds them.
RESERVED = {"ridge", "xgboost", "extratrees", "catboost", "tabpfn-2.5", "tabicl-v2", "realmlp", "tabm", "mambular",
            "contexttab", "tabstar", "tarte"}
