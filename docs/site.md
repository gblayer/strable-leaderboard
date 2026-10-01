# The site

## Hosting: GitHub Pages

One HTML page plus `results.json`, served by GitHub Pages at strable.ai. The page computes rankings and the Pareto front in the browser.

| | GitHub Pages | Hugging Face Space |
|---|---|---|
| What is served | Static files from a CDN | Static Space: static files. Gradio Space: a Python server that renders every view |
| First load | Tens of milliseconds for the HTML, then one JSON | Static: similar. Gradio: seconds, and a cold start of up to a minute after the free hardware sleeps |
| Interaction | In the browser, instant | Gradio: a round trip to the server per click |
| Custom domain | Free | Paid plan |
| Discoverability | Search engines, the paper, the repository | The Hugging Face catalogue and the dataset page |
| Cost | Free | Free tier |

The [Impermanent leaderboard](https://impermanent.timecopilot.dev/) loads fast for this reason: one static page on GitHub Pages. TabArena runs a Gradio Space, which is why it feels slow. A static Space mirror of the same HTML can be added later for the Hugging Face audience, with no extra work.

## Design

`site/index.html` is the current design, built from a preview whose numbers are placeholders. The **Demo** button walks through the board.

**Palette:** the Pantone set. Light theme on Mother of Pearl, dark theme on Mystic Navy. Greens stay close to Pine Grove.

| Token | Light | Dark | Pantone |
|---|---|---|---|
| Background | `#E9D4C3` | `#13273F` | Mother of Pearl 12-1006 TCX / Mystic Navy 7546C |
| Ink | `#13273F` | `#E9D4C3` | Mystic Navy / Mother of Pearl |
| Headline, buttons | `#4E0000` | `#9E2F3A` | Red Inferno 4975C |
| Second accent, suggestion panel | `#223631` | `#91ACA5` | Pine Grove 19-5406 TCX |
| Pareto marks | `#8E2A33` | `#D0646E` | Red Inferno tints |

**Type:** Balimo for headlines and text, a paid licence of about $24; the preview uses Outfit, a free look-alike. IBM Plex Mono for numbers and code.

**Header:** STRABLE, then STRing-tABLE, /ˈstreɪ.bəl/ and *stray-buhl*. The version chip and the Version history link at the top. One line: accepted to the NeurIPS 2026 Evaluation and Datasets track, and the board does not list the paper's ~445 pipelines. The art is a spreadsheet with a string column dropping into a table and a selected cell holding "str".

**Ticker:** top score, fastest pipeline on the front, size of the front, version.

**Controls:** Task (Both, Regression · R², Classification · AUC), Time (Fit, Predict) and Sort (★ Pareto, Score, Time). No filters: tags on the rows carry the rest.

**Key:** one line each for score, fit time, predict time, modular, end-to-end and runs-on.

**Table:** rank, pipeline, score, fit, predict, runs on. Tags for end-to-end, anonymous, closed. A star marks Pareto-optimal rows. Under the table: the build date, a Refresh button that re-reads `results.json`, and Show all.

**Plot:** score against time on a log scale, a "Better" arrow towards the top-left corner, the front as a staircase, API models as hollow squares.

**Version history**, then the **suggestion panel** on Pine Grove: a free-text box and an optional email, posted to a Tally form, with a link to open a GitHub issue instead.

**Links:** Submit a pipeline → this repository's README. Paper → arXiv. Footer: repository, benchmark code, data, paper.
