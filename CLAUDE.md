# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

An interactive, web-based visualizer of **loss functions and the training problem** —
how data $\mathbf{x}$ flows through a fixed model to predictions $\hat{\mathbf{y}}$, how
those compare against targets $\mathbf{y}$, and how the discrepancy becomes a per-sample
loss. Built for instructor-led presentation in UCI's AI4SCI Deep Learning course, with
secondary use as self-guided student exploration.

Sibling course projects in `../` — `nnintro`, `infoplane`, `mutual`, `ntk`, `polyfit` —
are the closest models for how this one should be built. `../nnintro` is the nearest
relative (same Vite + TypeScript frontend, same `uv` + PyTorch offline pipeline, same
`nnet-svg` network diagram, same GitHub Pages deploy) and is worth reading before
inventing structure here.

## Status: all four phases implemented

`PLAN.md` §5 Phases 1-4 are in place: the data and model pipeline, the app shell, the
five panel components, and the Pages deployment. The remote is
<https://github.com/dkirkby/data-model-loss>, default branch `main`, and the site is
<https://dkirkby.github.io/data-model-loss/>.

Two things in the UI are deliberately unbuilt: the per-sample inspector named in
`PLAN.md` §4C (Panel C says so in its own text), and point-level hit testing in the
corner plot, which the canvas renderer would need a `d3.quadtree` for.

- `PLAN.md` — the design spec: objectives, stack, UI panel-by-panel component specs,
  phased implementation steps, and the manual QA checklist.
- `DATA.md` — the data spec, kept current with the pipeline: both datasets, their
  physics-based synthetic generators, preprocessing, the full artifact schema, and the
  reference losses a clean run reproduces.

**Read both before writing anything.** Their contents are settled decisions, not
defaults — especially the scope boundaries (§1 of `PLAN.md`) and the exact color codes
and panel behaviors (§4).

Where the pipeline departs from `DATA.md`'s original formulas it does so deliberately,
and both the code and `DATA.md` §2.2 now explain why: the ISA lapse rate is applied to
geopotential height rather than to a pressure difference, and the geostrophic terms need
a pressure gradient that a per-sample scalar cannot supply. Do not "simplify" either
back to the one-line form.

## Commands

```bash
uv sync                           # create .venv from uv.lock (Python 3.12)
uv run python/generate_data.py    # regenerate the snapshot, ~40 s, deterministic
uv run python/generate_data.py --epochs 1200   # or --out PATH
uv run pytest                     # 19 pipeline tests, ~1 s
uv run pytest -k stellar          # one group
```

`generate_data.py` resolves its paths from the script, not the cwd, so it runs from
anywhere in the repo. It prints the twelve total losses on completion — compare them
against the table in `DATA.md` §5 to confirm a run is sane.

No linter or formatter is configured, matching the siblings. `scipy` is in `PLAN.md`'s
dependency list but nothing needs it, so it is not declared; `numpy` and `torch` are.

```bash
cd frontend
npm ci                  # clean install; no --legacy-peer-deps needed here
npm run dev             # vite dev server, at /data-model-loss/ (not /)
npm run typecheck       # tsc --noEmit
npm run test            # vitest: 20 tests, ~0.5 s
npm run build           # tsc && vite build -> frontend/dist
```

`npm run build` typechecks `src` **and** `tests` before bundling, so a broken test file
fails the build rather than slipping through.

`.github/workflows/deploy.yml` publishes to Pages on every push to `main`:
`npm ci && npm run test && npm run build`, then `upload-pages-artifact` over
`frontend/dist`. **CI never runs Python** — the site serves the committed JSON snapshot,
so refreshed numbers reach it only by running `uv run python/generate_data.py` locally
and committing the result. The tests run before the build on purpose:
`tests/artifact.test.ts` is what certifies the snapshot, and one the app disagrees with
should fail the job rather than ship.

Pages is configured with `build_type: workflow` (the "GitHub Actions" source). There is
no `configure-pages` step, because `vite.config.ts` pins `base` to `/data-model-loss/`
— if the repo is renamed or moves to a custom domain, that constant is the one thing
that has to change with it.

## Architecture

### One contract between two halves

The Python side and the browser side communicate through exactly one static file —
`frontend/public/data/model_runs.json` (`DATA.md` §4). It carries, per dataset:
metadata (domain, task type, feature/target display names, quantile ranges), the raw
`x`/`y` arrays, and an array of **6 weight sets** — each with its parameters, its
forward-pass `outputs`, its `sample_losses`, and its `total_loss`. Nothing is computed
in the browser that the pipeline could precompute; the frontend is a renderer over this
artifact. Changing the schema means changing both halves plus the docs.

The pipeline in `python/` writes **across** into the frontend's Vite `public/`
directory, so the artifact is served at `/data/model_runs.json` with no copy step. It is
a committed snapshot: Python does not run in CI, so refreshed numbers reach the site only
by regenerating locally and committing the file.

### The two datasets are deliberately isomorphic

Both are $\mathbb{R}^3 \to \mathbb{R}^3$ and both drive the *same* layout and the same
6×6 corner plot, so that regression and classification can be contrasted without the UI
changing shape:

| | Regression | Classification |
| --- | --- | --- |
| Domain | Atmospheric: $p, T, RH \to u, v, w$ | Astrophysical: $u\!-\!g, g\!-\!r, r\!-\!i \to$ one-hot class |
| Output activation | Identity | Softmax |
| Loss | MSE | Categorical cross-entropy |
| Samples | $N=500$ | $N=1500$ (500 per class, balanced) |

Each dataset has a real-data source (ERA5 via `cdsapi`; SDSS/Gaia photometry) **and** a
physics-based synthetic generator. The pipeline tries `python/raw_data/` first and falls
back to the generators automatically, so the project has **zero external API
dependencies** at build time. Keep that property.

**The committed snapshot is synthetic for both datasets** — `metadata.source` is
`"synthetic"`, `python/raw_data/` does not exist, and nothing has ever been downloaded.
The loaders for the real path are tested against fixture CSVs but have never seen an
actual ERA5 or SDSS export, so treat their column contracts as unproven against real
files. `test_committed_snapshot_is_synthetic` is the test to change if that ever
changes, and Panel C surfaces the provenance in the UI (`PLAN.md` §4C) so a generated
dataset is never presented as observation.

The model is fixed at $3 \to 16 \to 16 \to 3$ dense, two hidden layers. The 6 weight
sets are **3 pairs**, not 6 independent points: `trained_opt_N` is `random_N` after 800
epochs, because `TRAIN_CONFIGS` reuses `RANDOM_SEEDS`. Keep that pairing — the weight
selector is meant to show before and after on one optimization. The default selection is
the first pretrained set.

Each pair also differs in learning rate and batch size, so the loss spread across the
three trained sets is **not** attributable to $\eta$ alone; say so if the question comes
up.

### Visual coding is semantic, not decorative

- `#fecc07` (outline) = **dataset-fixed** components: features, targets.
- `#255799` (fill) = **parameter-dependent** components: model, outputs.

This distinction is the pedagogical point of the tool — what changes when $\Theta$
changes versus what cannot. Do not recolor panels for aesthetic reasons.

### The flowchart is the navigation

The top data-flow SVG (`FEATURES → MODEL → OUTPUT ↔ TARGET → LOSS`) is clickable, and
each block flashes the corner-plot subplots it corresponds to: `FEATURES`/`TARGET`/
`OUTPUT` highlight their own blocks, `MODEL` highlights every subplot involving
$\hat{y}$, and the loss region highlights the bottom-right diagonal histograms. The
corner plot's four 3×3 quadrants (feature–feature, output/target, feature–target,
feature–output) exist to make those mappings addressable, and all panels share
$[0.5\%, 99.5\%]$ quantile scaling with axes omitted.

Panels live in draggable/resizable/collapsible `Gridstack.js` tiles, arranged by the
instructor at presentation time; nothing auto-advances.

### How the frontend shell is put together

```
frontend/src/
  main.ts            load artifact -> init workspace -> wire panels -> emit once
  model/types.ts     the TS mirror of DATA.md §4; change both or neither
  model/artifact.ts  fetch + validate; validateArtifact is DOM-free so tests use it
  model/session.ts   the selection (dataset, weight set, loss coloring) + subscribers
  model/highlight.ts transient flowchart-click events; not state, so not on Session
  view/colors.ts     the two semantic colors, imported rather than re-typed
  view/workspace.ts  Gridstack init, drag/resize/collapse
  view/mapping.ts    the corner-plot and weight-matrix index arithmetic, pure
  view/Flowchart.ts  top container; the clickable blocks
  view/CornerPlot.ts Panel A; canvas marks under a DOM overlay
  view/ModelPanel.ts Panel B; nnet-svg plus the weight selector
  view/DatasetPanel.ts Panel C; description and the provenance badge
  view/LossPanel.ts  Panel D; KaTeX
```

**`view/mapping.ts` is where the plot's meaning lives.** Corner-plot indices 3-5 are
dual — targets below the diagonal, model outputs above it — and the weight matrices are
`[out][in]`, matching torch's `Linear.weight`. Both are places where an off-by-one gives
a diagram that looks entirely plausible and is wrong, which is why the functions are pure
and separately tested rather than inlined into the renderers.

Three things are easy to undo by accident:

**Fetches must go through `import.meta.env.BASE_URL`.** The site is served from
`/data-model-loss/` on Pages, so a leading-slash path (`/data/model_runs.json`) works in
dev and 404s in production. `vite.config.ts` sets `base`; `model/artifact.ts` is the only
place that builds the URL.

**Panels never talk to each other.** They subscribe to the `Session` and re-render from
it; `session.emit()` in `main.ts` is what paints the initial state. Switching dataset
resets the weight selection to that dataset's `default_weight_id`, because weight ids are
per-dataset.

**Collapse is hand-rolled.** Gridstack has no collapse; `view/workspace.ts` remembers a
panel's height in `dataset.expandedHeight`, shrinks it to one row, and restores it. The
grid runs in `mode: "float"` (gridstack 14's name for the old `float: true`) so a panel
stays where the instructor puts it.

### Explicitly out of scope

Optimization *trajectories*, train/validation/test splits, and learning kinetics are
excluded by design (`PLAN.md` §1, "Model-Agnostic Paradigm"). The tool shows the
training problem's static mapping, not the act of training. Resist adding loss curves
over epochs or an interactive optimizer.

## Testing

`PLAN.md` §6 is explicit: **do not write or run automated browser tests.** Verification
of UI behavior is a manual checklist the user executes. Pipeline-level unit tests in
Python and frontend data/fixture tests (as `../nnintro` does with vitest golden
fixtures) are the appropriate automated coverage.

`tests/test_pipeline.py` covers the pipeline in three groups: the physics generators
(declared ranges, geostrophic consistency, cleaning, class balance, class separability),
the loss and model definitions (MSE over three outputs, CE $= \ln 3$ at uniform logits,
$|\Theta| = 387$, reproducibility by seed), and the committed artifact itself (schema,
one-hot and softmax validity, trained beating random, random sitting at the uninformed
baseline, regression outputs in physical units). `tests/conftest.py` puts `python/` on
`sys.path` — the pipeline is a script, not a package (`tool.uv.package = false`).

The artifact tests skip rather than fail when the snapshot is missing, so a fresh clone
can run `uv run pytest` before `uv run python/generate_data.py`.

On the frontend, `frontend/tests/artifact.test.ts` pushes the committed snapshot through
the same `validateArtifact` the app uses — the other half of the DATA.md §4 contract —
and `frontend/tests/dependencies.test.ts` asserts gridstack, d3, katex and nnet-svg each
import and expose what Phase 3 reaches for. The artifact suite is pinned to vitest's
`node` environment (the project default is jsdom) because it reads the snapshot off disk
through `import.meta.url`, which jsdom reports as an `http://` URL. `frontend/tests/mapping.test.ts`
covers the index arithmetic against the real snapshot: which cells each flowchart block
highlights, that indices 3-5 resolve per triangle, and that all 352 edges and 35 biases
of the network diagram resolve to numbers summing to |Θ| = 387.
