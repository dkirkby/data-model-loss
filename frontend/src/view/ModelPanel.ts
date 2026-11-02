/**
 * Panel B: the architecture diagram and the weight selector (PLAN.md 4B).
 *
 * nnet-svg draws the 3 -> 16 -> 16 -> 3 network; this module supplies the
 * numbers. Edge color carries the sign of w and edge width its magnitude, node
 * fill the sign of b, so switching weight sets is visible at a glance even
 * though 387 numbers are far too many to read.
 *
 * Numeric labels are a toggle, off by default. PLAN.md asks for edges labeled
 * with w and nodes with b, but the middle layer pair alone has 16x16 = 256
 * edges: printed all at once they are unreadable. Every value is always
 * available on hover through the SVG <title>, and the toggle turns the printed
 * labels on when a layer is being discussed.
 */

import { renderDenseNetworkSvg } from "nnet-svg";

import type { Session } from "../model/session.ts";
import { MODEL_COLOR } from "./colors.ts";
import { biasFor, weightFor } from "./mapping.ts";

const ARCH = [3, 16, 16, 3];
/** Show every node: the default (12) would truncate the 16-wide hidden layers. */
const MAX_DISPLAYED_NODES = 16;

const POSITIVE = MODEL_COLOR;
const NEGATIVE = "#b3402c";

export interface ModelPanel {
  update(session: Session): void;
  destroy(): void;
}

function maxAbs(values: number[]): number {
  let max = 0;
  for (const value of values) max = Math.max(max, Math.abs(value));
  return max > 0 ? max : 1;
}

export function createModelPanel(host: HTMLElement): ModelPanel {
  host.replaceChildren();
  let showLabels = false;
  let current: Session | null = null;

  const badge = document.createElement("p");
  badge.className = "badge";

  const controls = document.createElement("div");
  controls.className = "controls";
  const labelToggle = document.createElement("label");
  labelToggle.className = "control";
  const labelCheckbox = document.createElement("input");
  labelCheckbox.type = "checkbox";
  labelCheckbox.addEventListener("change", () => {
    showLabels = labelCheckbox.checked;
    if (current) render(current);
  });
  const labelText = document.createElement("span");
  labelText.textContent = "Show w and b labels";
  labelToggle.append(labelCheckbox, labelText);
  controls.append(labelToggle);

  const legend = document.createElement("p");
  legend.className = "model-legend";
  legend.innerHTML =
    '<span class="swatch swatch-negative"></span> w, b &lt; 0' +
    '&nbsp;&nbsp;<span class="swatch swatch-positive"></span> w, b &gt; 0' +
    "&nbsp;&nbsp;thickness = |w|";

  const diagram = document.createElement("div");
  diagram.className = "model-diagram";

  const selector = document.createElement("fieldset");
  selector.className = "weight-selector";

  host.append(badge, controls, legend, diagram, selector);

  function renderSelector(session: Session): void {
    selector.replaceChildren();
    const legendEl = document.createElement("legend");
    legendEl.textContent = "Weights Θ";
    selector.append(legendEl);

    for (const entry of session.dataset.weights) {
      const row = document.createElement("label");
      row.className = "control";
      const radio = document.createElement("input");
      radio.type = "radio";
      radio.name = "weight-set";
      radio.value = entry.id;
      radio.checked = entry.id === session.weightSet.id;
      radio.addEventListener("change", () => session.selectWeights(entry.id));

      const name = document.createElement("span");
      name.textContent = entry.name;
      const loss = document.createElement("span");
      loss.className = "loss-chip";
      loss.textContent = `L = ${entry.total_loss.toFixed(4)}`;

      row.append(radio, name, loss);
      selector.append(row);
    }
  }

  function render(session: Session): void {
    const { parameters } = session.weightSet;
    const layers = parameters.layers;

    badge.textContent = `|Θ| = ${parameters.num_params} parameters`;

    const allWeights = layers.flatMap((layer) => layer.weights.flat());
    const allBiases = layers.flatMap((layer) => layer.biases);
    const weightScale = maxAbs(allWeights);
    const biasScale = maxAbs(allBiases);

    renderDenseNetworkSvg(diagram, {
      layers: ARCH,
      maxDisplayedNodes: MAX_DISPLAYED_NODES,
      orientation: "horizontal",
      responsive: true,
      title: `Dense network ${ARCH.join(" → ")} with weight set ${session.weightSet.name}`,
      edgeValue: (edge) => weightFor(layers, edge) ?? null,
      nodeValue: (node) => biasFor(layers, node) ?? null,
      edgeAttrs: (edge) => {
        const w = edge.value ?? 0;
        const strength = Math.abs(w) / weightScale;
        return {
          stroke: w < 0 ? NEGATIVE : POSITIVE,
          "stroke-width": 0.35 + 2.2 * strength,
          "stroke-opacity": 0.2 + 0.7 * strength,
        };
      },
      nodeAttrs: (node) => {
        if (node.layerIndex === 0) {
          return { fill: "#ffffff", stroke: "#8a8f96" };
        }
        const b = node.value ?? 0;
        const strength = Math.abs(b) / biasScale;
        return {
          fill: b < 0 ? NEGATIVE : POSITIVE,
          "fill-opacity": 0.15 + 0.7 * strength,
          stroke: b < 0 ? NEGATIVE : POSITIVE,
        };
      },
      edgeLabel: (edge) =>
        showLabels && edge.value !== undefined ? edge.value.toFixed(2) : null,
      nodeLabel: (node) => (showLabels && node.value !== undefined ? node.value.toFixed(2) : null),
      edgeLabelAttrs: () => ({ "font-size": 4, fill: "#44464a" }),
      nodeLabelAttrs: () => ({ "font-size": 5, fill: "#1a1a1a" }),
      edgeTitle: (edge) =>
        edge.value === undefined
          ? null
          : `w[${edge.sourceLayerIndex}][${edge.targetNodeIndex}][${edge.sourceNodeIndex}] = ${edge.value.toFixed(4)}`,
      nodeTitle: (node) =>
        node.value === undefined
          ? `input ${node.nodeIndex + 1}`
          : `b[${node.layerIndex - 1}][${node.nodeIndex}] = ${node.value.toFixed(4)}`,
    });

    renderSelector(session);
  }

  return {
    update(session: Session): void {
      current = session;
      labelCheckbox.checked = showLabels;
      render(session);
    },
    destroy(): void {
      host.replaceChildren();
    },
  };
}
