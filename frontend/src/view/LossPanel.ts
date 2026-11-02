/**
 * Panel D: the loss definition, rendered with KaTeX (PLAN.md 4D).
 *
 * The formula follows the active dataset -- MSE for the regression task,
 * categorical cross-entropy for the classifier -- and the running total for the
 * selected Theta sits underneath it, so the equation and the number a student is
 * looking at are never out of step.
 */

import { bin, scaleLinear } from "d3";
import katex from "katex";

import type { Session } from "../model/session.ts";
import type { LossName } from "../model/types.ts";
import { lossRamp } from "./lossramp.ts";

const SVG_NS = "http://www.w3.org/2000/svg";
// `top` reserves a band above the bars for the mean label; at top = 8 the label
// baseline fell outside the viewBox and the text was cut off.
const HIST = { width: 420, height: 112, bins: 30, left: 4, right: 4, top: 26, bottom: 18 };
/** Baseline of the mean label, inside the reserved band. */
const MEAN_LABEL_Y = 11;

function svg<K extends keyof SVGElementTagNameMap>(
  tag: K,
  attrs: Record<string, string | number> = {},
): SVGElementTagNameMap[K] {
  const el = document.createElementNS(SVG_NS, tag);
  for (const [key, value] of Object.entries(attrs)) el.setAttribute(key, String(value));
  return el;
}

/**
 * The distribution of per-sample losses for the selected Theta, bar-colored with
 * the same ramp the corner plot uses when "color by loss" is on, so a dark point
 * there and a dark bar here mean the same thing.
 *
 * The axis stops at the ramp's 99th-percentile cut and the final bin collects
 * everything above it. Cross-entropy in particular has an unbounded tail, and a
 * single confidently-wrong sample would otherwise squash the whole figure into
 * the leftmost pixel.
 */
function renderHistogram(host: HTMLElement, losses: number[], mean: number): void {
  host.replaceChildren();
  if (losses.length === 0) return;

  const ramp = lossRamp(losses);
  const [low, high] = ramp.domain;

  const root = svg("svg", {
    viewBox: `0 0 ${HIST.width} ${HIST.height}`,
    class: "loss-hist",
    role: "img",
    "aria-label": `Distribution of per-sample loss, mean ${mean.toFixed(4)}`,
  });

  const x = scaleLinear()
    .domain([low, high])
    .range([HIST.left, HIST.width - HIST.right]);
  const clamped = losses.map((value) => Math.min(value, high));
  const bins = bin<number, number>()
    .domain([low, high])
    .thresholds(x.ticks(HIST.bins))(clamped);

  const peak = Math.max(1, ...bins.map((item) => item.length));
  const y = scaleLinear()
    .domain([0, peak])
    .range([0, HIST.height - HIST.top - HIST.bottom]);
  const base = HIST.height - HIST.bottom;

  for (const item of bins) {
    if (item.x0 === undefined || item.x1 === undefined) continue;
    const left = x(item.x0);
    const width = Math.max(1, x(item.x1) - left - 0.5);
    const height = y(item.length);
    root.append(
      svg("rect", {
        x: left,
        y: base - height,
        width,
        height,
        fill: ramp.color((item.x0 + item.x1) / 2),
      }),
    );
  }

  root.append(
    svg("line", {
      x1: HIST.left,
      x2: HIST.width - HIST.right,
      y1: base,
      y2: base,
      stroke: "#c9ccd1",
      "stroke-width": 1,
    }),
  );

  // Where the reported total sits inside the spread it summarizes.
  const meanX = x(Math.min(mean, high));
  root.append(
    svg("line", {
      x1: meanX,
      x2: meanX,
      y1: HIST.top - 10,
      y2: base,
      stroke: "#1a1a1a",
      "stroke-width": 1.2,
      "stroke-dasharray": "3 2",
    }),
  );
  const meanLabel = svg("text", {
    x: meanX,
    y: MEAN_LABEL_Y,
    "text-anchor": meanX > HIST.width * 0.75 ? "end" : "start",
    "font-size": 9,
    fill: "#1a1a1a",
  });
  meanLabel.textContent = `mean ${mean.toFixed(4)}`;
  root.append(meanLabel);

  const lowLabel = svg("text", { x: HIST.left, y: HIST.height - 5, "font-size": 9, fill: "#5c5c5c" });
  lowLabel.textContent = `L = ${low.toFixed(high < 1 ? 3 : 2)}`;
  root.append(lowLabel);

  const highLabel = svg("text", {
    x: HIST.width - HIST.right,
    y: HIST.height - 5,
    "text-anchor": "end",
    "font-size": 9,
    fill: "#5c5c5c",
  });
  highLabel.textContent = `\u2265 ${high.toFixed(high < 1 ? 3 : 2)} (99th pct)`;
  root.append(highLabel);

  host.append(root);
}

interface LossCopy {
  title: string;
  formula: string;
  /** What the formula means, in the language of the active task. */
  explain: string;
  /** The unit or space the reported number lives in. */
  units: string;
}

/**
 * The dataset-level loss, the number the weight selector reports.
 *
 * Written as a mean rather than a bare sum because that is what the pipeline
 * exports: `total_loss` is `sample_losses.mean()`. A plain sum would be N times
 * larger and would not match the figure printed underneath it, and it would make
 * the two datasets incomparable, N being 500 against 1500.
 */
const TOTAL_FORMULA =
  "L(\\Theta) = \\frac{1}{N}\\sum_{i=1}^{N} L(\\hat{\\mathbf{y}}_i, \\mathbf{y}_i; \\Theta)";

const COPY: Record<LossName, LossCopy> = {
  mse: {
    title: "Mean Squared Error",
    formula:
      "L_{\\mathrm{MSE}}(\\hat{\\mathbf{y}}_i, \\mathbf{y}_i; \\Theta) = " +
      "\\frac{1}{3}\\sum_{l=1}^{3}\\left(\\hat{y}_{il} - y_{il}\\right)^{2}",
    explain:
      "One number per sample: the squared miss on each of the three wind components, " +
      "averaged. Squaring makes large misses dominate, and it is symmetric — " +
      "predicting too high costs the same as too low.",
    units:
      "Computed on standardized targets, so u, v and w weigh equally rather than " +
      "the two in m/s drowning out the one in cm/s. A value near 1 is what you get " +
      "for predicting the mean; 0 would be exact.",
  },
  cross_entropy: {
    title: "Categorical Cross-Entropy",
    formula:
      "L_{\\mathrm{CE}}(\\hat{\\mathbf{y}}_i, \\mathbf{y}_i; \\Theta) = " +
      "-\\sum_{l=1}^{3} y_{il}\\,\\log \\hat{y}_{il}, \\qquad " +
      "\\hat{y}_{il} = \\frac{e^{z_{il}}}{\\sum_{k} e^{z_{ik}}}",
    explain:
      "The targets are one-hot, so the sum keeps exactly one term: minus the log of " +
      "the probability assigned to the true class. Confident and right costs almost " +
      "nothing; confident and wrong is unbounded.",
    units:
      "A model that knows nothing spreads its probability evenly and scores " +
      "log 3 ≈ 1.0986. Lower than that means it has learned something.",
  },
};

export interface LossPanel {
  update(session: Session): void;
  destroy(): void;
}

export function createLossPanel(host: HTMLElement): LossPanel {
  host.replaceChildren();

  const title = document.createElement("p");
  title.className = "loss-title";

  const perSampleLabel = document.createElement("p");
  perSampleLabel.className = "control-label";
  perSampleLabel.textContent = "Per sample";

  const formula = document.createElement("div");
  formula.className = "loss-formula";

  const totalLabel = document.createElement("p");
  totalLabel.className = "control-label";
  totalLabel.textContent = "Over the dataset";

  const totalFormula = document.createElement("div");
  totalFormula.className = "loss-formula";

  const total = document.createElement("p");
  total.className = "badge";

  const histTitle = document.createElement("p");
  histTitle.className = "control-label";
  histTitle.textContent = "Per-sample loss distribution";

  const histogram = document.createElement("div");
  histogram.className = "loss-hist-host";

  const explain = document.createElement("p");
  explain.className = "loss-explain";

  const units = document.createElement("p");
  units.className = "loss-units";

  host.append(
    title,
    perSampleLabel,
    formula,
    totalLabel,
    totalFormula,
    total,
    histTitle,
    histogram,
    explain,
    units,
  );

  return {
    update(session: Session): void {
      const copy = COPY[session.dataset.metadata.loss_name];
      title.textContent = copy.title;
      // KaTeX writes its own markup; throwOnError is off so a formula typo
      // degrades to visible source rather than taking the panel down.
      const options = { displayMode: true, throwOnError: false, output: "html" } as const;
      katex.render(copy.formula, formula, options);
      katex.render(TOTAL_FORMULA, totalFormula, options);
      const weights = session.weightSet;
      total.textContent =
        `mean over ${session.dataset.metadata.num_samples} samples = ` +
        `${weights.total_loss.toFixed(6)} — ${weights.name}`;
      renderHistogram(histogram, weights.sample_losses, weights.total_loss);
      explain.textContent = copy.explain;
      units.textContent = copy.units;
    },
    destroy(): void {
      host.replaceChildren();
    },
  };
}
