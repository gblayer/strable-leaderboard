# The page

`index.html` is the leaderboard page, served by GitHub Pages at https://gblayer.github.io/strable-leaderboard/ (`.github/workflows/pages.yml` deploys this folder). The **Demo** button walks through the page.

The page reads `results.json`, built by `scripts/build_site_data.py` from the results folder together with `results.parquet` (all views) and the per-fold results `results_per_fold.parquet` and `results_per_fold.csv`, all offered for download. The Refresh button next to the version chip re-reads `results.json`. The Auto / Light / Dark control sets the theme; `#dark` or `#light` in the address forces one. The visit counter (GoatCounter) is in the head, commented out until the account exists. The protocol behind the numbers: [docs/protocol.md](../docs/protocol.md).
