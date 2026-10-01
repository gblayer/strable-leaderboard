# The page

`index.html` is the leaderboard page, served by GitHub Pages at https://gblayer.github.io/strable-leaderboard/ (`.github/workflows/pages.yml` deploys this folder). The **Demo** button walks through the page. The top bar with the theme switch and the demo is part of the preview, and all scores and times are placeholders until the launch.

The final page is the same file reading `results.json`, built from `results/` after each merge. The Refresh button next to the version chip re-reads that file. The visit counter (GoatCounter) is in the head, commented out until the account exists. See [docs/site.md](../docs/site.md).
