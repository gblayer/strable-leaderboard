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
OVERLAP = "pretraining-overlap"  # the leakage pass found that pre-training may include STRABLE tables; never blocks
# Per model folder, {"leakage": "possible", "pr": <model pull request>}: written by the dispatcher
# from the model pull request's OVERLAP label, read by scripts/build_site_data.py for the page's tag.
FLAGS = "site/flags.json"
# What a results pull request commits: the files scripts/build_site_data.py builds, and FLAGS.
RESULTS_FILES = ("site/results.json", "site/results.parquet", "site/results_per_fold.parquet",
                 "site/results_per_fold.csv", FLAGS)
SLUG = re.compile(r"^[a-z0-9][a-z0-9.-]*$")  # a model folder name
# Example folders for contributors: the dispatcher never runs them.
EXAMPLES = {"ridge"}
# Folders named test-<name>: the maintainer's test runs of the dispatcher, on a few tasks, never published.
TEST_PREFIX = "test-"
# Folder names only the maintainer adds: the paper's learners, and models she adds herself.
RESERVED = {"ridge", "xgboost", "extratrees", "catboost", "tabpfn-2.5", "tabicl-v2", "realmlp", "tabm", "mambular",
            "contexttab", "tabstar", "tarte", "causilo"}
