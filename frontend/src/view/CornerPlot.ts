/**
 * The 6x6 corner plot (PLAN.md section 4A).
 *
 * Six variables along each edge: x1 x2 x3 y1 y2 y3. Indices 3-5 are dual -- the
 * lower triangle reads them as targets y, the upper triangle as model outputs
 * yhat -- which is what gives the matrix its four blocks:
 *
 *            x1 x2 x3 | y1 y2 y3
 *      x1  [ features | x vs yhat ]   top-left: feature space
 *      x2  [   both   |  (upper)  ]   top-right: features vs outputs
 *      x3  [ triangles|           ]
 *      ----+----------+-----------
 *      y1  [ x vs y   | yhat vs yhat ] bottom-left: features vs targets
 *      y2  [ (lower)  | y vs y       ] bottom-right: outputs above the
 *      y3  [          |              ] diagonal, targets below it
 *
 * Diagonals 1-3 are feature histograms; diagonals 4-6 overlay the target
 * histogram (outlined #fecc07) on the prediction histogram (filled #255799),
 * which is where a bad Theta is most visible.
 *
 * The marks go to a single <canvas>: at N = 1500 the scatter blocks hold ~45,000
 * points, which is too many SVG nodes to re-render responsively when the weight
 * set changes. D3 still supplies the scales, the binning and the color ramp, and
 * the cell frames and flash highlights are DOM elements over the canvas, so
 * interaction stays declarative.
 */

import { bin, scaleLinear, type ScaleLinear } from "d3";

import type { Session } from "../model/session.ts";
import type { HighlightTarget } from "../model/highlight.ts";
import type { Range } from "../model/types.ts";
import { DATASET_COLOR, MODEL_COLOR } from "./colors.ts";
import { lossRamp } from "./lossramp.ts";
import {
  axisDomains,
  cellsFor,
  columnValues,
  FEATURES,
  involvesOutput,
  isUpper,
  SIZE,
  stepOutlinePoints,
  type HistogramBin,
} from "./mapping.ts";

const GAP = 3;
/** Room along the left and bottom edges for the variable names. */
const PAD = { left: 54, bottom: 30, top: 4, right: 4 };
/** The loss color bar: its width, its gap from the matrix, and room for labels. */
const BAR = { width: 13, gap: 10, labels: 52 };
const HIST_BINS = 22;
const FLASH_MS = 1300;

/**
 * Pure #fecc07 is close to invisible as a 1px mark on white, so dataset-only
 * scatter uses a darkened tone of the same hue. The exact spec color stays where
 * it reads: outlines, histogram strokes and the flowchart.
 */
const DATASET_MARK = "#8a6d05";

export interface CornerPlot {
  update(session: Session): void;
  flash(target: HighlightTarget): void;
  destroy(): void;
}

export function createCornerPlot(host: HTMLElement): CornerPlot {
  host.replaceChildren();

  const shell = document.createElement("div");
  shell.className = "corner-host";

  // PLAN.md 4A: the loss-coloring toggle lives with the plot it controls.
  const controls = document.createElement("div");
  controls.className = "controls corner-controls";

  /** A checkbox with an optional color swatch naming what it governs. */
  function checkbox(text: string, swatch?: string): [HTMLLabelElement, HTMLInputElement] {
    const label = document.createElement("label");
    label.className = "control";
    const input = document.createElement("input");
    input.type = "checkbox";
    label.append(input);
    if (swatch) {
      const chip = document.createElement("span");
      chip.className = `swatch swatch-${swatch}`;
      label.append(chip);
    }
    const span = document.createElement("span");
    span.textContent = text;
    label.append(span);
    return [label, input];
  }

  const [dataLabel, dataToggle] = checkbox("Show data", "dataset");
  const [predictionLabel, predictionToggle] = checkbox("Show predictions", "model");
  const [toggleLabel, toggle] = checkbox("Points colored by loss value");

  controls.append(dataLabel, predictionLabel, toggleLabel);
  shell.append(controls);

  const frame = document.createElement("div");
  frame.className = "corner";

  const canvas = document.createElement("canvas");
  canvas.className = "corner-canvas";
  frame.append(canvas);

  const overlay = document.createElement("div");
  overlay.className = "corner-overlay";

  // One frame element per cell, for the flash highlight.
  const cells: HTMLElement[][] = [];
  for (let row = 0; row < SIZE; row += 1) {
    const rowCells: HTMLElement[] = [];
    for (let col = 0; col < SIZE; col += 1) {
      const cell = document.createElement("div");
      cell.className = "corner-cell";
      if (row === col) cell.classList.add("is-diagonal");
      overlay.append(cell);
      rowCells.push(cell);
    }
    cells.push(rowCells);
  }

  // The feature block is outlined in the dataset color (PLAN.md section 4A).
  const featureBlock = document.createElement("div");
  featureBlock.className = "corner-block";
  overlay.append(featureBlock);
  frame.append(overlay);

  const labels = document.createElement("div");
  labels.className = "corner-labels";
  frame.append(labels);

  shell.append(frame);
  host.append(shell);

  let current: Session | null = null;

  /** Loss spans decades, so plain toFixed would print the low end as 0.000. */
  function format(value: number): string {
    if (value === 0) return "0";
    if (value < 0.001 || value >= 1000) return value.toExponential(1);
    return value.toPrecision(3);
  }

  function layout(
    showColorbar: boolean,
  ): { cell: number; plot: number; originX: number; originY: number } | null {
    // The gutter is only taken when the bar is shown, so the matrix uses the
    // full panel width the rest of the time.
    const right = showColorbar ? BAR.gap + BAR.width + BAR.labels : PAD.right;
    const width = frame.clientWidth - PAD.left - right;
    const height = frame.clientHeight - PAD.top - PAD.bottom;
    const plot = Math.min(width, height);
    if (!Number.isFinite(plot) || plot < 60) return null;
    const cell = (plot - GAP * (SIZE - 1)) / SIZE;
    return { cell, plot, originX: PAD.left, originY: PAD.top };
  }

  function placeOverlay(box: { cell: number; plot: number; originX: number; originY: number }): void {
    overlay.style.left = `${box.originX}px`;
    overlay.style.top = `${box.originY}px`;
    overlay.style.width = `${box.plot}px`;
    overlay.style.height = `${box.plot}px`;
    overlay.style.gridTemplateColumns = `repeat(${SIZE}, 1fr)`;
    overlay.style.gridTemplateRows = `repeat(${SIZE}, 1fr)`;
    overlay.style.gap = `${GAP}px`;

    const blockSpan = box.cell * FEATURES + GAP * (FEATURES - 1);
    featureBlock.style.left = "0px";
    featureBlock.style.top = "0px";
    featureBlock.style.width = `${blockSpan}px`;
    featureBlock.style.height = `${blockSpan}px`;
  }

  function renderLabels(
    session: Session,
    box: { cell: number; plot: number; originX: number; originY: number },
  ): void {
    const names = [
      ...session.dataset.metadata.feature_names,
      ...session.dataset.metadata.target_names,
    ];
    labels.replaceChildren();
    for (let i = 0; i < SIZE; i += 1) {
      const offset = i * (box.cell + GAP) + box.cell / 2;

      const rowLabel = document.createElement("span");
      rowLabel.className = "corner-label corner-label-row";
      rowLabel.textContent = names[i] ?? "";
      rowLabel.title = names[i] ?? "";
      rowLabel.style.top = `${box.originY + offset}px`;
      rowLabel.style.width = `${PAD.left - 6}px`;
      labels.append(rowLabel);

      const colLabel = document.createElement("span");
      colLabel.className = "corner-label corner-label-col";
      colLabel.textContent = names[i] ?? "";
      colLabel.title = names[i] ?? "";
      colLabel.style.left = `${box.originX + offset}px`;
      colLabel.style.top = `${box.originY + box.plot + 6}px`;
      labels.append(colLabel);
    }
  }

  /**
   * The loss color bar, down the right-hand side of the matrix.
   *
   * Drawn on the canvas rather than as a CSS gradient so its stops come from
   * the same `colorAt` the points do: a hand-written CSS approximation of
   * viridis drifts from the marks it is supposed to explain.
   */
  function drawColorbar(
    ctx: CanvasRenderingContext2D,
    ramp: ReturnType<typeof lossRamp>,
    box: { plot: number; originX: number; originY: number },
  ): void {
    const left = box.originX + box.plot + BAR.gap;
    const top = box.originY;
    const bottom = box.originY + box.plot;

    // High loss at the top, low at the bottom.
    const gradient = ctx.createLinearGradient(0, bottom, 0, top);
    const STOPS = 48;
    for (let i = 0; i <= STOPS; i += 1) {
      gradient.addColorStop(i / STOPS, ramp.colorAt(i / STOPS));
    }
    ctx.fillStyle = gradient;
    ctx.fillRect(left, top, BAR.width, bottom - top);
    ctx.strokeStyle = "#c9ccd1";
    ctx.lineWidth = 1;
    ctx.strokeRect(left, top, BAR.width, bottom - top);

    ctx.fillStyle = "#5c5c5c";
    ctx.strokeStyle = "#5c5c5c";
    ctx.font = "10px system-ui, -apple-system, sans-serif";
    ctx.textAlign = "left";
    ctx.textBaseline = "middle";

    for (const value of ramp.ticks(5)) {
      const y = bottom - ramp.position(value) * (bottom - top);
      ctx.beginPath();
      ctx.moveTo(left + BAR.width, y);
      ctx.lineTo(left + BAR.width + 3, y);
      ctx.stroke();
      ctx.fillText(format(value), left + BAR.width + 6, y);
    }

    // The top of the scale is a cut, not a maximum: say so.
    ctx.textBaseline = "bottom";
    ctx.fillText("L (log)", left, top - 3);
  }

  function draw(session: Session): void {
    const box = layout(session.colorByLoss);
    if (!box) return;
    placeOverlay(box);
    renderLabels(session, box);

    const dpr = window.devicePixelRatio || 1;
    const width = frame.clientWidth;
    const height = frame.clientHeight;
    canvas.width = Math.round(width * dpr);
    canvas.height = Math.round(height * dpr);
    canvas.style.width = `${width}px`;
    canvas.style.height = `${height}px`;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, width, height);

    const { metadata, data } = session.dataset;
    const weights = session.weightSet;
    const { quantiles } = metadata;

    const domains: Range[] = axisDomains(quantiles);
    const source = { x: data.x, y: data.y, outputs: weights.outputs };
    const column = (i: number, upper: boolean): number[] => columnValues(source, i, upper);

    const losses = weights.sample_losses;
    const ramp = lossRamp(losses);
    const pointColors = session.colorByLoss ? losses.map((value) => ramp.color(value)) : null;
    if (session.colorByLoss) drawColorbar(ctx, ramp, box);

    for (let row = 0; row < SIZE; row += 1) {
      for (let col = 0; col < SIZE; col += 1) {
        const left = box.originX + col * (box.cell + GAP);
        const top = box.originY + row * (box.cell + GAP);

        ctx.save();
        ctx.beginPath();
        ctx.rect(left, top, box.cell, box.cell);
        ctx.clip();

        if (row === col) {
          drawDiagonal(ctx, {
            left,
            top,
            size: box.cell,
            index: row,
            domain: domains[row] ?? [0, 1],
            dataValues: session.showData ? column(row, false) : null,
            outputValues:
              row >= FEATURES && session.showPredictions ? column(row, true) : null,
          });
        } else {
          const upper = isUpper(row, col);
          // A hidden series leaves its subplots empty rather than redrawing
          // them in the other series' color.
          const showsOutput = involvesOutput(row, col);
          if (showsOutput ? !session.showPredictions : !session.showData) {
            ctx.restore();
            continue;
          }
          const xs = column(col, upper);
          const ys = column(row, upper);
          drawScatter(ctx, {
            left,
            top,
            size: box.cell,
            xs,
            ys,
            xDomain: domains[col] ?? [0, 1],
            yDomain: domains[row] ?? [0, 1],
            baseColor: showsOutput ? MODEL_COLOR : DATASET_MARK,
            pointColors,
          });
        }
        ctx.restore();
      }
    }
  }

  function drawScatter(
    ctx: CanvasRenderingContext2D,
    spec: {
      left: number;
      top: number;
      size: number;
      xs: number[];
      ys: number[];
      xDomain: Range;
      yDomain: Range;
      baseColor: string;
      pointColors: string[] | null;
    },
  ): void {
    const x = scaleLinear().domain(spec.xDomain).range([spec.left + 2, spec.left + spec.size - 2]);
    const y = scaleLinear().domain(spec.yDomain).range([spec.top + spec.size - 2, spec.top + 2]);
    const radius = spec.size > 90 ? 1.5 : 1.1;

    if (spec.pointColors) {
      // Per-point color: no batching possible, so one path each.
      for (let i = 0; i < spec.xs.length; i += 1) {
        const px = spec.xs[i];
        const py = spec.ys[i];
        if (px === undefined || py === undefined || !Number.isFinite(px) || !Number.isFinite(py)) continue;
        ctx.fillStyle = spec.pointColors[i] ?? spec.baseColor;
        ctx.globalAlpha = 0.75;
        ctx.beginPath();
        ctx.arc(x(px), y(py), radius, 0, Math.PI * 2);
        ctx.fill();
      }
      ctx.globalAlpha = 1;
      return;
    }

    ctx.fillStyle = spec.baseColor;
    ctx.globalAlpha = 0.45;
    ctx.beginPath();
    for (let i = 0; i < spec.xs.length; i += 1) {
      const px = spec.xs[i];
      const py = spec.ys[i];
      if (px === undefined || py === undefined || !Number.isFinite(px) || !Number.isFinite(py)) continue;
      const cx = x(px);
      const cy = y(py);
      ctx.moveTo(cx + radius, cy);
      ctx.arc(cx, cy, radius, 0, Math.PI * 2);
    }
    ctx.fill();
    ctx.globalAlpha = 1;
  }

  function drawDiagonal(
    ctx: CanvasRenderingContext2D,
    spec: {
      left: number;
      top: number;
      size: number;
      index: number;
      domain: Range;
      dataValues: number[] | null;
      outputValues: number[] | null;
    },
  ): void {
    const x = scaleLinear().domain(spec.domain).range([spec.left + 2, spec.left + spec.size - 2]);
    const thresholds = x.ticks(HIST_BINS);
    const binner = bin<number, number>().domain(spec.domain).thresholds(thresholds);

    // Values outside the quantile domain are folded into the end bins rather
    // than discarded, so a histogram accounts for every sample.
    const fold = (values: number[]): number[] =>
      values
        .filter(Number.isFinite)
        .map((value) => Math.min(Math.max(value, spec.domain[0]), spec.domain[1]));

    const dataBins = spec.dataValues ? binner(fold(spec.dataValues)) : null;
    const outputBins = spec.outputValues ? binner(fold(spec.outputValues)) : null;
    if (!dataBins && !outputBins) return;

    // Only the visible series set the vertical scale, so hiding one lets the
    // other use the full height of the cell.
    const peak = Math.max(
      1,
      ...(dataBins ? dataBins.map((b) => b.length) : []),
      ...(outputBins ? outputBins.map((b) => b.length) : []),
    );
    // Headroom: the tallest bar stops short of the cell edge so it reads as a
    // peak rather than as a bar running off the top of the subplot.
    const base = spec.top + spec.size - 2;
    const height = scaleLinear()
      .domain([0, peak])
      .range([0, (spec.size - 4) * 0.86]);

    // Predictions first, as a solid fill; the dataset outline goes over them so
    // the comparison stays readable where they overlap.
    if (outputBins) {
      ctx.fillStyle = MODEL_COLOR;
      ctx.globalAlpha = 0.75;
      drawBars(ctx, outputBins, x, height, base);
      ctx.globalAlpha = 1;
    }

    if (!dataBins) return;
    ctx.strokeStyle = DATASET_COLOR;
    ctx.lineWidth = 1.6;
    drawStepOutline(ctx, dataBins, x, height, base);
  }

  /** Solid bars: the prediction histogram. */
  function drawBars(
    ctx: CanvasRenderingContext2D,
    bins: HistogramBin[],
    x: ScaleLinear<number, number>,
    height: ScaleLinear<number, number>,
    base: number,
  ): void {
    ctx.beginPath();
    for (const item of bins) {
      if (item.x0 === undefined || item.x1 === undefined) continue;
      const left = x(item.x0);
      const barHeight = height(item.length);
      ctx.rect(left, base - barHeight, Math.max(0.5, x(item.x1) - left), barHeight);
    }
    ctx.fill();
  }

  /**
   * A single unfilled step silhouette, matplotlib's histtype="step".
   *
   * One continuous path: up from the axis at the left edge, horizontal across
   * each bin, a vertical only at the boundary *between* adjacent bins, and back
   * down to the axis at the right edge. The earlier version dropped a line to
   * the axis after every bin, which drew a comb and read as a bevel or a drop
   * shadow against the filled bars underneath.
   */
  function drawStepOutline(
    ctx: CanvasRenderingContext2D,
    bins: HistogramBin[],
    x: ScaleLinear<number, number>,
    height: ScaleLinear<number, number>,
    base: number,
  ): void {
    const points = stepOutlinePoints(bins, x, height, base);
    if (points.length === 0) return;

    ctx.save();
    ctx.lineJoin = "miter";
    ctx.lineCap = "butt";
    ctx.beginPath();
    const [first, ...rest] = points;
    ctx.moveTo(first![0], first![1]);
    for (const [px, py] of rest) ctx.lineTo(px, py);
    ctx.stroke();
    ctx.restore();
  }

  const observer = new ResizeObserver(() => {
    if (current) draw(current);
  });

  observer.observe(frame);

  let wired = false;

  return {
    update(session: Session): void {
      current = session;
      if (!wired) {
        toggle.addEventListener("change", () => session.setColorByLoss(toggle.checked));
        dataToggle.addEventListener("change", () => session.setShowData(dataToggle.checked));
        predictionToggle.addEventListener("change", () =>
          session.setShowPredictions(predictionToggle.checked),
        );
        wired = true;
      }
      toggle.checked = session.colorByLoss;
      dataToggle.checked = session.showData;
      predictionToggle.checked = session.showPredictions;
      draw(session);
    },
    flash(target: HighlightTarget): void {
      const matches = cellsFor(target);
      for (let row = 0; row < SIZE; row += 1) {
        for (let col = 0; col < SIZE; col += 1) {
          const cell = cells[row]?.[col];
          if (!cell || !matches(row, col)) continue;
          cell.classList.remove("is-flashing");
          // Force a reflow so a repeated click restarts the animation.
          void cell.offsetWidth;
          cell.classList.add("is-flashing");
          window.setTimeout(() => cell.classList.remove("is-flashing"), FLASH_MS);
        }
      }
    },
    destroy(): void {
      observer.disconnect();
      host.replaceChildren();
    },
  };
}
