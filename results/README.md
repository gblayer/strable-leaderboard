# Results

One file per model, `<model>.csv`, one row per encoder, table and outer fold. The columns are in [docs/protocol.md](../docs/protocol.md#result-records); `python -m strable_leaderboard.check_results <file>` validates a file.

`versions.yaml` lists the board's versions, newest first. `python scripts/build_site_data.py` turns this folder into `site/results.json`.
