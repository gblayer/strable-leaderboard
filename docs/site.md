# The site

## Where people open it

The leaderboard is one web page, served free by GitHub Pages from this repository at https://gblayer.github.io/strable-leaderboard/. Pages is switched on (Settings → Pages → Source: GitHub Actions) and `.github/workflows/pages.yml` deploys `site/` on every push that touches it. The repository can stay private: only the published page is public. No domain is needed; a custom one could be added later under Settings → Pages.

**Visits.** GitHub's Insights → Traffic counts visits to the repository, not to the page. For the page, [GoatCounter](https://www.goatcounter.com) is free for non-commercial use, needs no cookies and no banner. The script tag is already in `site/index.html`, commented out: create the account with the code `strable`, then remove the comment marks.

## Hosting: GitHub Pages

One HTML page plus `results.json` (and `results.parquet`, for download). The page computes rankings and the Pareto front in the browser.

| | GitHub Pages | Hugging Face Space |
|---|---|---|
| What is served | Static files from a CDN | Static Space: static files. Gradio Space: a Python server that renders every view |
| First load | Tens of milliseconds for the HTML, then one JSON | Static: similar. Gradio: seconds, and a cold start of up to a minute after the free hardware sleeps |
| Interaction | In the browser, instant | Gradio: a round trip to the server per click |
| Custom domain, if ever wanted | Free | Paid plan |
| Discoverability | Search engines, the paper, the repository | The Hugging Face catalogue and the dataset page |
| Cost | Free | Free tier |

The [Impermanent leaderboard](https://impermanent.timecopilot.dev/) loads fast for this reason: one static page on GitHub Pages. TabArena runs a Gradio Space, which is why it feels slow. A static Space mirror of the same HTML can be added later for the Hugging Face audience, with no extra work.

## Design

`site/index.html` is the page; every number on it comes from `results.json`. The **Demo** button walks through the board.

**Palette:** the Pantone set. Light theme on Mother of Pearl, dark theme on Mystic Navy. Greens stay close to Pine Grove.

| Token | Light | Dark | Pantone |
|---|---|---|---|
| Background | `#E9D4C3` | `#13273F` | Mother of Pearl 12-1006 TCX / Mystic Navy 7546C |
| Ink | `#13273F` | `#E9D4C3` | Mystic Navy / Mother of Pearl |
| Headline, buttons | `#4E0000` | `#9E2F3A` | Red Inferno 4975C |
| Second accent, suggestion panel | `#223631` | `#91ACA5` | Pine Grove 19-5406 TCX |
| Pareto marks | `#8E2A33` | `#D0646E` | Red Inferno tints |

**Type:** Balimo for headlines and text, a paid licence of about $24; the page uses Outfit, a free look-alike. IBM Plex Mono for numbers and code.

**Header:** STRABLE, then STRing-tABLE, /ˈstreɪ.bəl/ and *stray-buhl*. At the top: the version chip, the Version history link, the build date and the Refresh button, which re-reads `results.json` without reloading the page. Navigation: Datasets, Benchmark code, Paper, the Submit a pipeline button to the README's submission steps (`#submit-a-model`), and a Cite button that opens the paper's BibTeX in a dialog with a Copy button. One line: accepted to the NeurIPS 2026 Evaluation and Datasets track, and the board does not list the paper's ~445 pipelines. The art is a spreadsheet with a string column dropping into a table and a selected cell holding "str".

**Ticker:** top score, fastest pipeline on the front, size of the front, version.

**Controls:** Task (Both, Regression · R², Classification · AUC) and Sort (★ Pareto, Score, Time). No filters and no key: the paper defines the numbers, tags on the rows carry the rest.

**Table:** rank, pipeline, score (median R² & AUC), time (total run time per 1K samples), at a fixed height with its own scrollbar, so every pipeline is always in it. Tags for end-to-end, tuned, anonymous, closed, paper and incomplete rows; the incomplete tag carries no counts, and the paper tag's tooltip says that paper rows were timed by the paper's code on mixed hardware, so time comparisons with runner rows are approximate. A star marks the rows on the Pareto Front, the others are grouped as "Not on the Pareto Front", and one red frame surrounds the top three rows of the current order.

**Plot:** as in the paper's Figure 4a, y "Median score (R² & AUC)" (or "Median R²", "Median AUC"), x "Total run time per 1K samples (s), log scale", a "Better" arrow towards the top-left corner, the front as a staircase from the x axis to the right edge, labels on the front, a ring around the top three.

**Using the leaderboard in your paper:** downloads of the current view (Task and Sort), named `strable-leaderboard_<version>_<date>_<view>.<ext>`. The plot as SVG, PNG, JPEG (from the SVG on a canvas) or PDF (jsPDF with svg2pdf.js), in the light palette on white with a line giving the version and date. The ranking as CSV, JSON or XLSX (SheetJS): rank, pipeline, variant, tags, median score, median time, on the Pareto Front, with the version, date, view, column definitions and the citation. `results.parquet` holds all three views at once. The three libraries load from cdnjs or jsdelivr, pinned with SRI hashes, only when their button is clicked.

**Version history**, then a low **suggestion panel** on Pine Grove: one line and a button that opens the repository's new-issue chooser on GitHub. No footer.

**Demo:** a button in the navigation, nine steps that show where things are: the two controls, the table, the plot, Refresh, the version history, the downloads, Cite, and how to add a model. The theme follows the system (or `#dark` / `#light` in the address).

**Data:** the page reads `results.json` next to it when it opens and when Refresh is clicked; `scripts/build_site_data.py` writes it together with `results.parquet`. Until it has loaded, the board and the version chip are empty; if it cannot be read, a short message under the table says so and the downloads stay off.
