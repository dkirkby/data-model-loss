/**
 * The central data flow chart (PLAN.md, section 4 "Top Container").
 *
 *   FEATURES  ->  MODEL  ->  OUTPUT  <->  TARGET
 *                                    `-- the loss
 *
 * The two dataset-fixed blocks are outlined in #fecc07 and the two
 * parameter-dependent ones filled with #255799, which is the whole point of the
 * figure: the left and right edges are given, the middle is what Theta decides.
 *
 * Every block is also a button. Clicking one emits a HighlightTarget, and the
 * corner plot flashes the subplots that show that quantity.
 */

import { DATASET_COLOR, LOSS_COLOR, MODEL_COLOR } from "./colors.ts";
import type { HighlightBus, HighlightTarget } from "../model/highlight.ts";

const SVG_NS = "http://www.w3.org/2000/svg";

// Internal coordinate system; the SVG scales to its container. The aspect ratio
// is what decides how much of a taller panel the figure actually uses: with the
// old 960x190 the extra height just became whitespace on either side of a
// letterboxed drawing.
const VIEW = { width: 960, height: 560 };
// FEATURES is wider than the other two data blocks; OUTPUT and TARGET share one
// narrower width so the pair the loss compares reads as a matched pair.
const BLOCK = { top: 68, height: 440, wide: 132, narrow: 80, modelWidth: 150 };

/** One tick row per drawn sample. Representative, not one per real sample. */
const SAMPLE_ROWS = 26;

const COLUMNS = {
  features: 58,
  model: 322,
  output: 600,
  target: 800,
} as const;

function svg<K extends keyof SVGElementTagNameMap>(
  tag: K,
  attrs: Record<string, string | number> = {},
): SVGElementTagNameMap[K] {
  const el = document.createElementNS(SVG_NS, tag);
  for (const [key, value] of Object.entries(attrs)) el.setAttribute(key, String(value));
  return el;
}

/** An SVG text label with a subscript, e.g. x_ij. */
function mathLabel(x: number, y: number, base: string, sub: string, fill: string): SVGTextElement {
  const text = svg("text", {
    x,
    y,
    fill,
    "text-anchor": "middle",
    "font-size": 24,
    "font-style": "italic",
    "font-family": "Georgia, 'Times New Roman', serif",
  });
  text.append(document.createTextNode(base));
  const subscript = svg("tspan", { "baseline-shift": "sub", "font-size": 16, "font-style": "normal" });
  subscript.textContent = sub;
  text.append(subscript);
  return text;
}

function columnTitle(x: number, label: string): SVGTextElement {
  const text = svg("text", {
    x,
    y: 40,
    "text-anchor": "middle",
    "font-size": 15,
    "font-weight": 600,
    "letter-spacing": 1.2,
    fill: "#44464a",
  });
  text.textContent = label;
  return text;
}

/** Evenly spaced row centers inside a tall block: the samples i = 1..N. */
function rowCenters(): number[] {
  const step = BLOCK.height / (SAMPLE_ROWS + 1);
  return Array.from({ length: SAMPLE_ROWS }, (_, i) => BLOCK.top + step * (i + 1));
}

export interface Flowchart {
  destroy(): void;
}

export function createFlowchart(host: HTMLElement, bus: HighlightBus, sampleCount: number): Flowchart {
  host.replaceChildren();

  const root = svg("svg", {
    viewBox: `0 0 ${VIEW.width} ${VIEW.height}`,
    class: "flowchart",
    role: "group",
    "aria-label": "Data flow from features through the model to outputs, compared against targets",
  });

  // Arrowheads: one grey for the forward path, one dark green for the loss link.
  const defs = svg("defs");
  for (const [id, color] of [["fc-arrow", "#6b6f76"], ["fc-arrow-loss", LOSS_COLOR]] as const) {
    const marker = svg("marker", {
      id,
      viewBox: "0 0 10 10",
      refX: 9,
      refY: 5,
      markerWidth: 6,
      markerHeight: 6,
      orient: "auto-start-reverse",
    });
    marker.append(svg("path", { d: "M 0 1 L 9 5 L 0 9 z", fill: color }));
    defs.append(marker);
  }
  root.append(defs);

  const rows = rowCenters();

  /** A clickable block: the rect, its titles, and the keyboard affordances. */
  function block(
    target: HighlightTarget,
    x: number,
    width: number,
    options: {
      filled: boolean;
      /** Draw the dataset-color outline even on a filled block. */
      outlined?: boolean;
      rounded?: boolean;
      title: string;
      base: string;
      sub: string;
    },
  ): SVGGElement {
    const group = svg("g", {
      class: `fc-block fc-block-${target}`,
      role: "button",
      tabindex: 0,
      "aria-label": `${options.title}: highlight the matching corner-plot subplots`,
    });
    const rect = svg("rect", {
      x,
      y: BLOCK.top,
      width,
      height: BLOCK.height,
      // Sharp corners on the data blocks; only MODEL is rounded.
      rx: options.rounded ? 16 : 0,
      fill: options.filled ? MODEL_COLOR : "#ffffff",
      stroke: options.filled && !options.outlined ? MODEL_COLOR : DATASET_COLOR,
      "stroke-width": options.filled && !options.outlined ? 1 : 2.5,
    });
    group.append(rect);

    if (!options.filled) {
      // The sample rows, drawn only on the outlined blocks so the height reads
      // as "one row per sample" without obscuring the filled ones.
      for (const y of rows) {
        group.append(
          svg("line", {
            x1: x + 10,
            x2: x + width - 10,
            y1: y,
            y2: y,
            stroke: DATASET_COLOR,
            "stroke-width": 1,
            opacity: 0.55,
          }),
        );
      }
    }

    group.append(columnTitle(x + width / 2, options.title));
    group.append(
      mathLabel(
        x + width / 2,
        BLOCK.top + BLOCK.height / 2 + 9,
        options.base,
        options.sub,
        options.filled ? "#ffffff" : "#1a1a1a",
      ),
    );

    const activate = () => bus.emit(target);
    group.addEventListener("click", () => {
      activate();
      // Clicking should highlight and nothing else. Without this the block keeps
      // focus and some browsers leave the focus ring drawn around it; keyboard
      // activation goes through keydown below and keeps its ring.
      group.blur();
    });
    group.addEventListener("keydown", (event) => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        activate();
      }
    });
    return group;
  }

  /** Thin per-sample arrows between two columns. */
  function connect(xFrom: number, xTo: number): SVGGElement {
    const group = svg("g", { class: "fc-flow" });
    for (const y of rows) {
      group.append(
        svg("line", {
          x1: xFrom,
          x2: xTo,
          y1: y,
          y2: y,
          stroke: "#9aa0a6",
          "stroke-width": 0.8,
          "marker-end": "url(#fc-arrow)",
        }),
      );
    }
    return group;
  }

  const featuresX = COLUMNS.features;
  const modelX = COLUMNS.model;
  const outputX = COLUMNS.output;
  const targetX = COLUMNS.target;

  root.append(connect(featuresX + BLOCK.wide, modelX));
  root.append(connect(modelX + BLOCK.modelWidth, outputX));

  // The loss link: a dark green double-headed arrow per aligned sample pair,
  // plus a vertical spine carrying the rotated label.
  const lossGroup = svg("g", {
    class: "fc-block fc-block-loss",
    role: "button",
    tabindex: 0,
    "aria-label": "Loss function: highlight the output and target histograms",
  });
  const gapFrom = outputX + BLOCK.narrow;
  const gapTo = targetX;
  const gapMid = (gapFrom + gapTo) / 2;
  lossGroup.append(
    svg("rect", {
      x: gapFrom,
      y: BLOCK.top,
      width: gapTo - gapFrom,
      height: BLOCK.height,
      fill: "transparent",
    }),
  );
  for (const y of rows) {
    lossGroup.append(
      svg("line", {
        x1: gapFrom + 2,
        x2: gapTo - 2,
        y1: y,
        y2: y,
        stroke: LOSS_COLOR,
        "stroke-width": 1,
        opacity: 0.75,
        "marker-start": "url(#fc-arrow-loss)",
        "marker-end": "url(#fc-arrow-loss)",
      }),
    );
  }
  lossGroup.append(
    svg("line", {
      x1: gapMid,
      x2: gapMid,
      y1: BLOCK.top - 6,
      y2: BLOCK.top + BLOCK.height + 6,
      stroke: LOSS_COLOR,
      "stroke-width": 2,
    }),
  );
  const lossLabel = svg("text", {
    transform: `rotate(-90 ${gapMid} ${BLOCK.top + BLOCK.height / 2})`,
    x: gapMid,
    y: BLOCK.top + BLOCK.height / 2,
    "text-anchor": "middle",
    "font-size": 15,
    "font-weight": 600,
    "letter-spacing": 1.5,
    fill: LOSS_COLOR,
    dy: -9,
  });
  lossLabel.textContent = "LOSS FUNCTION";
  lossGroup.append(lossLabel);
  const lossMath = svg("text", {
    transform: `rotate(-90 ${gapMid} ${BLOCK.top + BLOCK.height / 2})`,
    x: gapMid,
    y: BLOCK.top + BLOCK.height / 2,
    "text-anchor": "middle",
    "font-size": 17,
    "font-style": "italic",
    "font-family": "Georgia, 'Times New Roman', serif",
    fill: LOSS_COLOR,
    dy: 15,
  });
  lossMath.textContent = "L(ŷ, y; Θ)";
  lossGroup.append(lossMath);

  const activateLoss = () => bus.emit("loss");
  lossGroup.addEventListener("click", () => {
    activateLoss();
    lossGroup.blur();
  });
  lossGroup.addEventListener("keydown", (event) => {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      activateLoss();
    }
  });

  root.append(
    block("features", featuresX, BLOCK.wide, {
      filled: false,
      title: "FEATURES",
      base: "x",
      sub: "ij",
    }),
  );
  root.append(
    block("model", modelX, BLOCK.modelWidth, {
      filled: true,
      rounded: true,
      title: "MODEL",
      base: "f",
      sub: "l (xᵢ; Θ)",
    }),
  );
  root.append(
    block("output", outputX, BLOCK.narrow, {
      filled: true,
      // Model-blue fill for a parameter-dependent quantity, but outlined like
      // the data blocks: it is the per-sample array the loss compares against
      // TARGET, and the outline makes that pairing visible.
      outlined: true,
      title: "OUTPUT",
      base: "ŷ",
      sub: "il",
    }),
  );
  root.append(lossGroup);
  root.append(
    block("target", targetX, BLOCK.narrow, {
      filled: false,
      title: "TARGET",
      base: "y",
      sub: "il",
    }),
  );

  // The sample-count annotation, so the block height has a stated meaning.
  const samples = svg("text", {
    x: featuresX - 8,
    y: BLOCK.top + BLOCK.height / 2,
    "text-anchor": "end",
    "font-size": 13,
    fill: "#6b6f76",
  });
  samples.textContent = `i = 1 … ${sampleCount}`;
  root.append(samples);

  const hint = svg("text", { x: VIEW.width / 2, y: VIEW.height - 10, "text-anchor": "middle", "font-size": 13, fill: "#8a8f96" });
  hint.textContent = "click a block to highlight the subplots that show it";
  root.append(hint);

  host.append(root);

  return {
    destroy(): void {
      host.replaceChildren();
    },
  };
}
