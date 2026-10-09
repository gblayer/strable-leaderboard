import copy
from pathlib import Path

import yaml

from strable_leaderboard.card import card_problems
from strable_leaderboard.check_requirements import problems as pin_problems
from strable_leaderboard.smoke import code_problems, static_problems

ROOT = Path(__file__).resolve().parent.parent
GOOD = {
    "name": "TabPFN", "version": "2.5", "handles_strings": False, "tasks": ["regression", "binary"],
    "variant": "default", "access": "open", "hardware": "gpu", "package": "tabpfn==6.0.6",
    "checkpoint": {"repo": "Prior-Labs/TabPFN-v2.5", "revision": "a" * 40}, "licence": "Prior Labs licence",
    "trained_on_strable": [], "code": "https://github.com/PriorLabs/TabPFN",
}


def _folder(tmp_path, card=GOOD, requirements="tabpfn==6.0.6\n", code="class Model: pass\n"):
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
               checkpoint={"repo": "x/y", "revision": "main"}, trained_on_strable=["not-a-table"])
    found = "\n".join(card_problems(bad))
    for word in ["missing field licence", "unknown field typo_field", "variant", "access", "hardware", "tasks",
                 "40-character commit hash", "unknown tables"]:
        assert word in found


def test_template_placeholders_fail():
    template = yaml.safe_load((ROOT / "models" / "_template" / "model.yaml").read_text())
    found = "\n".join(card_problems(template))
    assert "placeholder 'My Model'" in found and "placeholder 'my-model==1.0.0'" in found


def test_pins_must_equal_the_shared_env(tmp_path):
    assert pin_problems(_folder(tmp_path)) == []
    found = pin_problems(_folder(tmp_path / "b", requirements="tabpfn==9.0.0\nrequests\nnot-in-env==1.0\n"))
    assert any("the shared environment has tabpfn==6.0.6" in p for p in found)
    assert any("'requests'" in p for p in found)  # unpinned
    assert any("not-in-env==1.0: not in the shared environment" in p for p in found)


def test_card_package_must_match_requirements(tmp_path):
    found = pin_problems(_folder(tmp_path, requirements="scikit-learn==1.7.2\n"))
    assert any("model.yaml package tabpfn==6.0.6" in p for p in found)


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
