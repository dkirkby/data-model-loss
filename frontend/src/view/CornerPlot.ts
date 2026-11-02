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

import { bin, interpolateViridis, quantile, scaleLinear, type ScaleLinear } from "d3";

import type { Session } from "../model/session.ts";
import type { HighlightTarget } from "../model/highlight.ts";
import type { Range } from "../model/types.ts";
import { DATASET_COLOR, MODEL_COLOR } from "./colors.ts";
import {
  axisDomains,
  cellsFor,
  columnValues,
  FEATURES,
  involvesOutput,
  isUpper,
  SIZE,
} from "./mapping.ts";

const GAP = 3;
/** Room along the left and bottom edges for the variable names. */
const PAD = { left: 54, bottom: 30, top: 4, right: 4 };
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

  host.append(frame);

  let current: Session | null = null;

  function layout(): { cell: number; plot: number; originX: number; originY: number } | null {
    const width = frame.clientWidth - PAD.left - PAD.right;
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

  function draw(session: Session): void {
    const box = layout();
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

    // The loss ramp is cut at the 99th percentile: a handful of badly fit
    // samples would otherwise flatten the whole scale.
    const losses = weights.sample_losses;
    const lossHigh = quantile(losses, 0.99) ?? 1;
    const lossLow = Math.min(...losses);
    const lossColor = scaleLinear<number>()
      .domain([lossLow, lossHigh > lossLow ? lossHigh : lossLow + 1])
      .range([0, 1])
      .clamp(true);

    const pointColors = session.colorByLoss
      ? losses.map((value) => interpolateViridis(lossColor(value)))
      : null;

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
            dataValues: column(row, false),
            outputValues: row >= FEATURES ? column(row, true) : null,
          });
        } else {
          const upper = isUpper(row, col);
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
            baseColor: involvesOutput(row, col) ? MODEL_COLOR : DATASET_MARK,
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
      dataValues: number[];
      outputValues: number[] | null;
    },
  ): void {
    const x = scaleLinear().domain(spec.domain).range([spec.left + 2, spec.left + spec.size - 2]);
    const thresholds = x.ticks(HIST_BINS);
    const binner = bin<number, number>().domain(spec.domain).thresholds(thresholds);

    const dataBins = binner(spec.dataValues.filter(Number.isFinite));
    const outputBins = spec.outputValues
      ? binner(spec.outputValues.filter(Number.isFinite))
      : null;

    const peak = Math.max(
      1,
      ...dataBins.map((b) => b.length),
      ...(outputBins ? outputBins.map((b) => b.length) : []),
    );
    const height = scaleLinear()
      .domain([0, peak])
      .range([0, spec.size - 6]);

    const base = spec.top + spec.size - 2;

    // Predictions first, as a solid fill; the targets are then stroked over
    // them so the comparison stays readable where they overlap.
    if (outputBins) {
      ctx.fillStyle = MODEL_COLOR;
      ctx.globalAlpha = 0.75;
      drawBars(ctx, outputBins, x, height, base, "fill");
      ctx.globalAlpha = 1;
    }

    if (spec.index < FEATURES) {
      ctx.fillStyle = DATASET_COLOR;
      ctx.globalAlpha = 0.55;
      drawBars(ctx, dataBins, x, height, base, "fill");
      ctx.globalAlpha = 1;
    }
    ctx.strokeStyle = spec.index < FEATURES ? DATASET_MARK : DATASET_COLOR;
    ctx.lineWidth = spec.index < FEATURES ? 1 : 1.6;
    drawBars(ctx, dataBins, x, height, base, "stroke");
  }

  function drawBars(
    ctx: CanvasRenderingContext2D,
    bins: { x0?: number; x1?: number; length: number }[],
    x: ScaleLinear<number, number>,
    height: ScaleLinear<number, number>,
    base: number,
    mode: "fill" | "stroke",
  ): void {
    ctx.beginPath();
    for (const item of bins) {
      if (item.x0 === undefined || item.x1 === undefined) continue;
      const left = x(item.x0);
      const right = x(item.x1);
      const barHeight = height(item.length);
      if (mode === "fill") {
        ctx.rect(left, base - barHeight, Math.max(0.5, right - left), barHeight);
      } else {
        // An outline that traces the bar tops, rather than 22 separate boxes.
        ctx.moveTo(left, base - barHeight);
        ctx.lineTo(right, base - barHeight);
        ctx.lineTo(right, base);
      }
    }
    if (mode === "fill") ctx.fill();
    else ctx.stroke();
  }

  const observer = new ResizeObserver(() => {
    if (current) draw(current);
  });
  observer.observe(frame);

  return {
    update(session: Session): void {
      current = session;
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
