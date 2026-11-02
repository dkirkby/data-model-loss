/**
 * Panel bodies for the Phase 2 shell.
 *
 * The controls are real and wired to the Session: dataset selector (Panel C),
 * weight selector (Panel B), and the loss-coloring toggle (Panel A). The
 * visualizations are placeholders that state what Phase 3 mounts in their place
 * -- Flowchart.ts, CornerPlot.ts, ModelPanel.ts and LossPanel.ts, per PLAN.md
 * section 5 Phase 3.
 */

import type { Session } from "../model/session.ts";
import { DATASET_KEYS, DATASET_LABELS, SOURCE_LABELS, type DatasetKey } from "../model/types.ts";

function element<K extends keyof HTMLElementTagNameMap>(
  tag: K,
  className?: string,
  text?: string,
): HTMLElementTagNameMap[K] {
  const el = document.createElement(tag);
  if (className) el.className = className;
  if (text !== undefined) el.textContent = text;
  return el;
}

function clear(host: HTMLElement): void {
  host.replaceChildren();
}

/** A stand-in for a Phase 3 component, so the shell reads honestly. */
function placeholder(host: HTMLElement, component: string, describes: string): void {
  const box = element("div", "placeholder");
  box.append(
    element("p", "placeholder-component", component),
    element("p", "placeholder-note", describes),
  );
  host.append(box);
}

// -- Top container: the data flow chart -----------------------------------

export function renderFlowchart(host: HTMLElement): void {
  clear(host);
  placeholder(
    host,
    "Flowchart.ts",
    "FEATURES → MODEL → OUTPUT ↔ TARGET, with the loss link. " +
      "Clicking a block will highlight the corner-plot subplots it maps to.",
  );
}

// -- Panel A: the 6x6 corner plot -----------------------------------------

export function renderCornerPlot(host: HTMLElement, session: Session): void {
  clear(host);

  const controls = element("div", "controls");
  const label = element("label", "control");
  const toggle = element("input");
  toggle.type = "checkbox";
  toggle.checked = session.colorByLoss;
  toggle.addEventListener("change", () => session.setColorByLoss(toggle.checked));
  label.append(toggle, element("span", undefined, "Points colored by loss value"));
  controls.append(label);
  host.append(controls);

  const { metadata } = session.dataset;
  placeholder(
    host,
    "CornerPlot.ts",
    `6×6 matrix over ${metadata.feature_names.join(", ")} and ` +
      `${metadata.target_names.join(", ")}, scaled to the [0.5%, 99.5%] quantiles ` +
      `in metadata.quantiles. Loss coloring is ${session.colorByLoss ? "on" : "off"}.`,
  );
}

// -- Panel B: model architecture and the weight selector ------------------

export function renderModelPanel(host: HTMLElement, session: Session): void {
  clear(host);

  const { weights } = session.dataset;
  const active = session.weightSet;

  const badge = element("p", "badge", `|Θ| = ${active.parameters.num_params} parameters`);
  host.append(badge);

  const fieldset = element("fieldset", "weight-selector");
  fieldset.append(element("legend", undefined, "Weights Θ"));
  for (const entry of weights) {
    const row = element("label", "control");
    const radio = element("input");
    radio.type = "radio";
    radio.name = "weight-set";
    radio.value = entry.id;
    radio.checked = entry.id === active.id;
    radio.addEventListener("change", () => session.selectWeights(entry.id));
    row.append(
      radio,
      element("span", undefined, entry.name),
      element("span", "loss-chip", `L = ${entry.total_loss.toFixed(4)}`),
    );
    fieldset.append(row);
  }
  host.append(fieldset);

  const layers = active.parameters.layers
    .map((layer) => `${layer.in_features}→${layer.out_features}`)
    .join(", ");
  placeholder(host, "ModelPanel.ts", `nnet-svg diagram of the dense layers (${layers}).`);
}

// -- Panel C: dataset description and inspector ---------------------------

export function renderDatasetPanel(host: HTMLElement, session: Session): void {
  clear(host);
  const { metadata } = session.dataset;

  const selector = element("label", "control control-block");
  selector.append(element("span", "control-label", "Dataset"));
  const select = element("select");
  for (const key of DATASET_KEYS) {
    const option = element("option", undefined, DATASET_LABELS[key]);
    option.value = key;
    option.selected = key === session.datasetKey;
    select.append(option);
  }
  select.addEventListener("change", () => session.selectDataset(select.value as DatasetKey));
  selector.append(select);
  host.append(selector);

  host.append(
    element("p", "dataset-domain", `${metadata.domain} — ${metadata.task_type}, N = ${metadata.num_samples}`),
  );

  // The provenance badge is the one element of this panel that is not optional
  // (PLAN.md section 4C): a generated dataset must never read as observation.
  const provenance = element("p", `provenance provenance-${metadata.source}`);
  provenance.append(
    element("strong", undefined, "Data source: "),
    element("span", undefined, SOURCE_LABELS[metadata.source]),
  );
  host.append(provenance);

  const variables = element("dl", "variables");
  variables.append(
    element("dt", undefined, "Features x"),
    element("dd", undefined, metadata.feature_names.join(", ")),
    element("dt", undefined, "Targets y"),
    element("dd", undefined, metadata.target_names.join(", ")),
  );
  host.append(variables);

  placeholder(host, "Inspector (Phase 3)", "Per-sample readout of x, y, yhat and L.");
}

// -- Panel D: the loss function definition --------------------------------

const LOSS_TITLES = {
  mse: "Mean Squared Error",
  cross_entropy: "Categorical Cross-Entropy",
} as const;

export function renderLossPanel(host: HTMLElement, session: Session): void {
  clear(host);
  const { metadata } = session.dataset;
  const active = session.weightSet;

  host.append(
    element("p", "loss-title", LOSS_TITLES[metadata.loss_name]),
    element("p", "badge", `total loss = ${active.total_loss.toFixed(6)}  (${active.name})`),
  );
  placeholder(host, "LossPanel.ts", "KaTeX rendering of the loss definition and its explanation.");
}
