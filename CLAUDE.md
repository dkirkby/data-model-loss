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

## Status: Phase 1 done, frontend not started

`PLAN.md` §5 Phase 1 — the data and model pipeline — is implemented and its snapshot is
built. Phases 2-4 (the Vite/TypeScript app, the panels, the Pages workflow) do not exist
yet. The repository is **not yet a git repo**; the intended remote is
<https://github.com/dkirkby/data-model-loss>.

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

Phase 2 begins with `npm create vite@latest . -- --template vanilla-ts` inside
`frontend/`, then `npm install gridstack d3 nnet-svg katex` (nnet-svg pinned to 0.1.1).
Follow `../nnintro` for the steady-state frontend commands: `npm ci` / `npm run dev` /
`npm run typecheck` / `npm run test` (vitest) / `npm run build`.

Deployment is GitHub Actions → GitHub Pages on push to `main` (`PLAN.md` §5 Phase 4).
In the siblings, **CI never runs Python**: the site serves the committed JSON snapshot,
so refreshed numbers reach the site only by regenerating locally and committing the
artifact.

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
sets are 3 random initializations (different seeds) + 3 converged SGD runs (different
hyperparameters); the default selection is the first *pretrained* set.

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
