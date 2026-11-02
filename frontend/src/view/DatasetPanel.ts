/**
 * Panel C: what the active dataset is, and where it came from (PLAN.md 4C).
 *
 * The provenance badge is the one element here that is not optional: the
 * committed snapshot is synthetic, and a student must never read it as
 * reanalysis or a photometric catalog.
 */

import type { Session } from "../model/session.ts";
import { DATASET_KEYS, DATASET_LABELS, SOURCE_LABELS, type DatasetKey } from "../model/types.ts";

export interface DatasetPanel {
  update(session: Session): void;
  destroy(): void;
}

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

export function createDatasetPanel(host: HTMLElement): DatasetPanel {
  host.replaceChildren();

  const selectorLabel = element("label", "control control-block");
  selectorLabel.append(element("span", "control-label", "Dataset"));
  const select = document.createElement("select");
  for (const key of DATASET_KEYS) {
    const option = document.createElement("option");
    option.value = key;
    option.textContent = DATASET_LABELS[key];
    select.append(option);
  }
  selectorLabel.append(select);

  const domain = element("p", "dataset-domain");
  const provenance = element("p", "provenance");
  const variables = element("dl", "variables");
  const inspector = element("p", "loss-units");

  host.append(selectorLabel, domain, provenance, variables, inspector);

  let wired = false;

  return {
    update(session: Session): void {
      if (!wired) {
        select.addEventListener("change", () =>
          session.selectDataset(select.value as DatasetKey),
        );
        wired = true;
      }
      select.value = session.datasetKey;

      const { metadata } = session.dataset;
      domain.textContent =
        `${metadata.domain} — ${metadata.task_type}, N = ${metadata.num_samples}`;

      provenance.className = `provenance provenance-${metadata.source}`;
      provenance.replaceChildren(
        element("strong", undefined, "Data source: "),
        element("span", undefined, SOURCE_LABELS[metadata.source]),
      );

      variables.replaceChildren(
        element("dt", undefined, "Features x"),
        element("dd", undefined, metadata.feature_names.join(", ")),
        element("dt", undefined, "Targets y"),
        element("dd", undefined, metadata.target_names.join(", ")),
      );

      inspector.textContent =
        `Showing ${session.weightSet.name}. ` +
        "Per-sample inspection of x, y, ŷ and L is not built yet.";
    },
    destroy(): void {
      host.replaceChildren();
    },
  };
}
