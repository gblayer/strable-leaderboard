# The site: hosting and design

## Hosting

**A static site on GitHub Pages, at strable.ai.** It is one HTML page plus one `results.json`. The page computes rankings and the Pareto front in the browser.

Why not a Hugging Face Space:
- A Space is only a host. A *static* Space serves plain HTML, much like GitHub Pages.
- Leaderboards such as TabArena use a *Gradio* Space instead: a Python server that renders the page. On free hardware it sleeps when idle and needs a cold start, and every interaction goes through the server. That is why they feel slow.
- The [Impermanent leaderboard](https://impermanent.timecopilot.dev/) loads fast for this reason: it is one static HTML file served by GitHub Pages through a CDN, with Chart.js loaded from jsDelivr.
- A custom domain is free on GitHub Pages. On a Space it needs a paid plan.

A static Space mirror can be added later, so the board also shows up on Hugging Face.

## Design preview

`site/index.html` is the current design. Open it in a browser. The **Demo** button walks through the board.

**Palette: Pantone set.** Light theme on Mother of Pearl, dark theme on Mystic Navy.

| Token | Light | Dark | Pantone |
|---|---|---|---|
| Background | `#E9D4C3` | `#13273F` | Mother of Pearl 12-1006 TCX / Mystic Navy 7546C |
| Ink | `#13273F` | `#E9D4C3` | Mystic Navy / Mother of Pearl |
| Headline, buttons | `#4E0000` | `#9E2F3A` | Red Inferno 4975C |
| Second accent | `#223631` | `#91ACA5` | Pine Grove 19-5406 TCX |
| Pareto marks | `#8E2A33` | `#D0646E` | Red Inferno tints |

The suggestion panel is Pine Grove `#223631` in both themes. Greens stay close to Pine Grove.

**Type**
- Headlines and text: Balimo, which needs a paid licence of about $24 from YouWorkForThem. The preview uses Outfit, a free look-alike.
- Numbers and code: IBM Plex Mono.

**Header**
- STRABLE, then "STRing-tABLE", /ˈstreɪ.bəl/ and *stray-buhl*.
- The version chip and a Version history link sit at the top.
- A line says STRABLE was accepted to the NeurIPS 2026 Evaluation and Datasets track.
- The art is a spreadsheet: a string column drops into a table, and the selected cell holds "str".

**Ticker:** the scrolling line under the header: top score, fastest pipeline on the front, size of the front and version. Kept.

**Controls**
- **Task:** Both, the default, Regression · R², or Classification · AUC.
- **Time:** Fit or Predict. It picks the plot's time axis and the Pareto order.
- **Family:** all, foundation model, tree-based, neural network, linear or other.
- **Access:** all, open source, closed source or API.
- **Sort:** Pareto first, the default, Score, or Time.

**Key:** short definitions of score, fit time, predict time, modular, end-to-end and hardware.

**Table columns:** rank, pipeline, score, fit, predict and runs on. Tags mark end-to-end, anonymous and API entries, and a star marks Pareto-optimal rows.

**Plot:** score against time on a log scale.
- A "Better" arrow points to the top-left corner.
- The Pareto front is drawn as a staircase.
- API models appear as hollow squares.
- Hovering a point shows the score, both times, the family, the access and the hardware.

**Version history:** one line per version.

**Suggestion panel:** "Which feature would you like to see on STRABLE?"
- One free-text box and an optional email.
- On strable.ai it posts to a free Tally form that emails each suggestion to gioia.blayer@inria.fr.
- A link opens a GitHub issue instead.

**Links**
- "Submit a pipeline" points to this repository's README.
- The Paper link points to arXiv.
- The footer links this repository, the benchmark code, the data and the paper.

**Demo, six steps:** task, sort, time, modular versus end-to-end, how to read the plot, how to add a model.

All numbers in the preview are placeholders. Only the pipeline names come from the paper.
