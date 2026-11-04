"""The weekly check for new models: which models are new, and the issue it writes. No network."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import model_check as watch  # noqa: E402


def row(model, elo=1500, **extra):
    return {"Model": model, "Elo [⬆️]": str(elo), "Elo 95% CI": "+5/-5", "TypeName": "Foundation Model",
            "License": "MIT", "Organization": "Lab", **extra}


def test_a_model_is_its_name_without_variant_and_found_once():
    found = watch.models([row("[TabDPT (tuned + ensembled)](https://a)", 1600), row("[TabDPT (default)](https://a)", 1400),
                          row("[TabFM+](https://b)", 1700), row("[AutoGluon 1.6 (extreme, 4h)](https://c)"), row("Plain")])
    assert {k: m["name"] for k, m in found.items()} == {
        "tabdpt": "TabDPT", "tabfm": "TabFM", "autogluon16": "AutoGluon 1.6", "plain": "Plain"}
    assert found["tabdpt"]["variants"] == ["TabDPT (tuned + ensembled)", "TabDPT (default)"]
    assert found["tabdpt"]["url"] == "https://a" and found["tabdpt"]["licence"] == "MIT"
    assert (found["tabfm"]["rank"], found["tabdpt"]["rank"], found["tabdpt"]["elo"], found["tabdpt"]["of"]) == (1, 2, 1600, 5)


def test_new_is_a_top_model_not_on_the_board_not_known_not_an_issue(tmp_path):
    names = ("TabPFN-3.6", "TabPFN-3.5", "Kumo-Tabular", "LimiX-2", "Seen", "New-1", "New-2", "Low")
    found = watch.models([row(f"[{n} (default)](https://x)", 2000 - 10 * i) for i, n in enumerate(names)])
    known = tmp_path / "known.txt"
    known.write_text("# comment\nLimiX-2\n")
    skip = watch.board_names() | watch.known_names(known) | watch.issue_names(["New model: Seen", "Other title"])
    assert "tabpfn35" in skip and "kumotabular" in skip  # on the board: models/tabpfn-3.5/, models/kumo-tabular/
    new = watch.new_models(found, skip, limit=2, top=7)
    assert [m["name"] for m in new] == ["TabPFN-3.6", "New-1"] and [m["i"] for m in new] == [0, 1]
    assert [m["name"] for m in watch.new_models(found, skip, limit=5, top=7)] == ["TabPFN-3.6", "New-1", "New-2"]
    assert [m["name"] for m in watch.new_models(found, skip, limit=5, top=8)][-1] == "Low"


def test_issue_text_quotes_the_answer_and_escapes_table_cells():
    model = watch.models([row("[New|Model (default)](https://x)")])["newmodel"]
    answer = {"summary": "A model.\nTwo lines.", "tasks": ["regression", "binary"], "weights": "open",
              "licence": "MIT", "install": "pip install new==1.0", "text_columns": "category codes",
              "mode": "encoders", "hardware": "8 GB", "runnable": "yes", "blockers": [],
              "evidence": [{"quote": "Strings are encoded as categories.", "url": "https://x/readme"}]}
    assert watch.title(model) == "New model: New|Model"
    text = watch.issue(model, answer)
    assert "**New\\|Model**" in text and "(rank 1 of 1 rows by Elo, 1500)" in text
    assert "| What it is | A model. Two lines. |" in text
    assert "with the board's four string encoders" in text and "What stops it" not in text
    assert '- "Strings are encoded as categories." (https://x/readme)' in text
