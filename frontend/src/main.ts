/**
 * Application entry point: load the artifact, raise the panel canvas, wire the
 * panels to the session.
 *
 * Phase 3 replaces the placeholder bodies in view/panels.ts with the real
 * components. The shape here -- one Session, panels that subscribe to it and
 * re-render -- is what those components plug into.
 */

import "gridstack/dist/gridstack.min.css";
import "katex/dist/katex.min.css";
import "./style.css";

import { loadArtifact } from "./model/artifact.ts";
import { Session } from "./model/session.ts";
import { initWorkspace } from "./view/workspace.ts";
import {
  renderCornerPlot,
  renderDatasetPanel,
  renderFlowchart,
  renderLossPanel,
  renderModelPanel,
} from "./view/panels.ts";

function setStatus(message: string, isError = false): void {
  const status = document.querySelector<HTMLElement>("#status");
  if (!status) return;
  status.textContent = message;
  status.classList.toggle("status-error", isError);
}

async function start(): Promise<void> {
  const artifact = await loadArtifact();
  const workspace = initWorkspace();
  const session = new Session(artifact, "atmospheric_regression");

  // The flow chart does not depend on the selection, so it renders once.
  renderFlowchart(workspace.body("flowchart"));

  session.subscribe((s) => renderCornerPlot(workspace.body("corner"), s));
  session.subscribe((s) => renderModelPanel(workspace.body("model"), s));
  session.subscribe((s) => renderDatasetPanel(workspace.body("dataset"), s));
  session.subscribe((s) => renderLossPanel(workspace.body("loss"), s));
  session.emit();

  const { metadata } = session.dataset;
  setStatus(
    `Loaded ${Object.keys(artifact.datasets).length} datasets — ` +
      `showing ${metadata.domain}, ${metadata.num_samples} samples, ` +
      `${session.dataset.weights.length} weight sets.`,
  );
}

start().catch((error: unknown) => {
  const message = error instanceof Error ? error.message : String(error);
  setStatus(`Could not start: ${message}`, true);
  console.error(error);
});
