# The page

`index.html` is the leaderboard page, served by GitHub Pages at https://gblayer.github.io/strable-leaderboard/ (`.github/workflows/pages.yml` deploys this folder). The **Demo** button walks through the page.

The page reads `results.json`, built by `scripts/build_site_data.py` from the private results folder together with `results.parquet` (all views, offered for download). The Refresh button next to the version chip re-reads that file. The visit counter (GoatCounter) is in the head, commented out until the account exists. See [docs/site.md](../docs/site.md).
