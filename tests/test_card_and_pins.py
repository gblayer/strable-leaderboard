"""Checks that the smoke test rejects incomplete model cards, versions that differ from the shared environment
and own folds or seeds, and runs the model with Hugging Face downloads off."""

import copy
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from strable_leaderboard.card import card_problems
from strable_leaderboard.check_requirements import freeze_constraints, pin_changes, problems as pin_problems
from strable_leaderboard.smoke import code_problems, static_problems

ROOT = Path(__file__).resolve().parent.parent
GOOD = {
    "name": "TabPFN", "version": "2.5", "end_to_end": False, "tasks": ["regression", "binary"],
    "variant": "default", "hardware": "gpu", "package": "tabpfn==9.0.0",
    "checkpoint": {"repo": "Prior-Labs/TabPFN-v2.5", "revision": "a" * 40}, "licence": "Prior Labs licence",
    "pretraining_data": "synthetic", "code": "https://github.com/PriorLabs/TabPFN",
}


def _folder(tmp_path, card=GOOD, requirements="tabpfn==9.0.0\n", code="class Model: pass\n"):
    folder = tmp_path / "tabpfn-2.5"
    folder.mkdir(parents=True)
    (folder / "model.yaml").write_text(yaml.safe_dump(card))
    (folder / "requirements.txt").write_text(requirements)
    (folder / "model.py").write_text(code)
    return folder


def test_complete_card_passes():
    assert card_problems(GOOD) == []


def test_every_missing_or_invalid_field_is_reported():
    bad = copy.deepcopy(GOOD)
    del bad["licence"]
    bad.update(variant="tuned", access="api", hardware="tpu", tasks=["ranking"], typo_field=1,
               checkpoint={"repo": "x/y", "revision": "main"}, pretraining_data="web tables")
    found = "\n".join(card_problems(bad))
    for word in ["missing field licence", "unknown field typo_field", "unknown field access", "variant", "hardware", "tasks",
                 "40-character commit hash", "pretraining_data must be one of"]:
        assert word in found


def test_pretraining_data_takes_one_of_five_values():
    for value in ["synthetic", "real", "mixed", "none", "not disclosed"]:
        assert card_problems(dict(GOOD, pretraining_data=value)) == []
    for value in [None, "unknown", ["synthetic"]]:
        assert card_problems(dict(GOOD, pretraining_data=value)) == [
            "pretraining_data must be one of ['mixed', 'none', 'not disclosed', 'real', 'synthetic']"]


def test_checkpoint_files_are_optional():
    ckpt = GOOD["checkpoint"]
    assert card_problems(dict(GOOD, checkpoint=dict(ckpt, files=["model.ckpt", "*.json"]))) == []
    for files in [[], "model.ckpt", [""], [1]]:
        assert "checkpoint.files" in "\n".join(card_problems(dict(GOOD, checkpoint=dict(ckpt, files=files))))
    assert card_problems(dict(GOOD, checkpoint=dict(ckpt, branch="main")))


def test_template_placeholders_fail():
    template = yaml.safe_load((ROOT / "models" / "_template" / "model.yaml").read_text())
    found = "\n".join(card_problems(template))
    assert "placeholder 'My Model'" in found and "placeholder 'my-model==1.0.0'" in found


def test_pins_must_equal_the_shared_env(tmp_path):
    assert pin_problems(_folder(tmp_path)) == []
    found = pin_problems(_folder(tmp_path / "b", requirements="tabpfn==6.0.6\nrequests\nnot-in-env==1.0\n"))
    assert any("the shared environment has tabpfn==9.0.0" in p for p in found)
    assert any("'requests'" in p for p in found)  # unpinned
    assert any("not-in-env==1.0: not in the shared environment" in p for p in found)


def test_card_package_must_match_requirements(tmp_path):
    found = pin_problems(_folder(tmp_path, requirements="scikit-learn==1.7.2\n"))
    assert any("model.yaml package tabpfn==9.0.0" in p for p in found)


def test_own_folds_and_seeds_are_flagged(tmp_path):
    code = ("from sklearn.model_selection import train_test_split, KFold\n"
            "m = Est(random_state=42)\nm = Est(random_state=self.seed)  # fine\n"
            "torch.manual_seed(0)\n# KFold in a comment is fine\n")
    found = code_problems(_folder(tmp_path, code=code))
    assert len(found) == 3
    assert "line 1" in found[0] and "line 2" in found[1] and "line 4" in found[2]


def test_example_models_pass_the_static_checks():
    assert static_problems(ROOT / "models" / "ridge") == []
    assert static_problems(ROOT / "models" / "_template") != []


DOWNLOADS = """import numpy as np
from huggingface_hub import hf_hub_download

from strable_leaderboard.model import StrableModel


class Model(StrableModel):
    def fit(self, X, y):
        try:
            hf_hub_download("no-owner/no-model", "model.ckpt")
        except Exception as exc:
            raise RuntimeError(f"download failed: {type(exc.__cause__).__name__}") from exc
        return self

    def predict(self, X):
        return np.zeros(len(X))
"""


def test_the_smoke_test_runs_the_model_with_downloads_off(tmp_path):
    table = tmp_path / "data" / "china-overseas-finance-inventory"  # the regression smoke table, made up
    table.mkdir(parents=True)
    rng = np.random.default_rng(0)
    pd.DataFrame({"x": rng.normal(size=60), "y": rng.normal(size=60)}).to_parquet(table / "data.parquet")
    (table / "config.json").write_text(json.dumps({"target_name": "y", "task": "regression"}))
    card = dict(GOOD, tasks=["regression"], hardware="cpu", package="scikit-learn==1.7.2", checkpoint=None)
    folder = _folder(tmp_path, card=card, requirements="scikit-learn==1.7.2\n", code=DOWNLOADS)
    # huggingface_hub is imported first, as the weight download does before the model runs.
    code = ("import sys, huggingface_hub.file_download; from strable_leaderboard.smoke import main; "
            f"sys.exit(main([{str(folder)!r}, '--data-dir', {str(tmp_path / 'data')!r}]))")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=ROOT)
    assert out.returncode == 1 and "download failed: OfflineModeIsEnabled" in out.stdout, out.stdout + out.stderr


def _env(folder, txt, req_in):
    folder.mkdir(parents=True)
    (folder / "requirements.txt").write_text(txt)
    (folder / "requirements.in").write_text(req_in)
    return folder


def test_a_pull_request_may_only_add_packages(tmp_path):
    base = _env(tmp_path / "base", "numpy==2.3.4\nfilelock==3.20.0\n", "numpy==2.3.4\n")
    added = _env(tmp_path / "added", "numpy==2.3.4\nfilelock==3.20.0\n", "numpy==2.3.4\nkumo==1.0\n")
    assert pin_changes(base, added) == []
    for name, txt, req_in in [("bumped", "numpy==2.3.4\nfilelock==3.20.0\n", "numpy==2.4.0\n"),
                              ("removed", "numpy==2.3.4\n", "numpy==2.3.4\n"),
                              ("dependency", "numpy==2.3.4\nfilelock==3.20.0\n", "numpy==2.3.4\nfilelock==3.21\n"),
                              ("range", "numpy==2.3.4\nfilelock==3.20.0\n", "numpy==2.3.4\nkumo>=1\n")]:
        assert pin_changes(base, _env(tmp_path / name, txt, req_in)), name


def test_a_package_added_to_requirements_in_satisfies_the_model_but_not_the_runner(tmp_path):
    env = _env(tmp_path / "env", "scikit-learn==1.7.2\n", "scikit-learn==1.7.2\nkumo==1.0\n")
    card = dict(GOOD, package="kumo==1.0", checkpoint=None)
    folder = _folder(tmp_path / "m", card=card, requirements="kumo==1.0\n")
    assert pin_problems(folder, (env / "requirements.txt", env / "requirements.in")) == []
    assert pin_problems(folder, (env / "requirements.txt",)) != []


def test_lock_changes_list_additions_and_every_moved_version():
    from strable_leaderboard.check_requirements import lock_changes

    old = {"numpy": "2.3.4", "scikit-learn": "1.7.2", "joblib": "1.5.2"}
    assert lock_changes(old, dict(old, tabulate="0.9.0")) == ["- Added: `tabulate==0.9.0`",
                                                              "- No existing package changes version."]
    lines = lock_changes(old, {"numpy": "2.4.0", "scikit-learn": "1.5.2", "sktime": "0.30.0"})
    assert lines == ["- Added: `sktime==0.30.0`", "- `joblib` 1.5.2 → removed", "- `numpy` 2.3.4 → 2.4.0",
                     "- `scikit-learn` 1.7.2 → 1.5.2 **downgrade**"]


def test_a_requirements_in_pin_overrides_the_freeze(tmp_path):
    req = tmp_path / "requirements.in"
    req.write_text("tabpfn==9.0.0\nnumpy==2.3.4\n")
    kept = freeze_constraints(req, tmp_path / "freeze.txt").read_text().splitlines()
    assert "tabpfn==6.0.6" not in kept and "numpy==2.3.4" in kept and "skrub==0.6.2" in kept
