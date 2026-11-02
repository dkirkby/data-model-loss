/**
 * Application entry point: load the artifact, raise the panel canvas, wire the
 * five components to the session and to each other.
 *
 * The only cross-panel coupling is the HighlightBus: a click on a flowchart
 * block flashes the corner-plot subplots that show that quantity. Everything
 * else flows one way, from the Session outward.
 */

import "gridstack/dist/gridstack.min.css";
import "katex/dist/katex.min.css";
import "./style.css";

import { loadArtifact } from "./model/artifact.ts";
import { HighlightBus } from "./model/highlight.ts";
import { Session } from "./model/session.ts";
import { initWorkspace } from "./view/workspace.ts";
import { createCornerPlot } from "./view/CornerPlot.ts";
import { createDatasetPanel } from "./view/DatasetPanel.ts";
import { createFlowchart } from "./view/Flowchart.ts";
import { createLossPanel } from "./view/LossPanel.ts";
import { createModelPanel } from "./view/ModelPanel.ts";

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
  const highlights = new HighlightBus();

  const cornerPlot = createCornerPlot(workspace.body("corner"));
  const modelPanel = createModelPanel(workspace.body("model"));
  const datasetPanel = createDatasetPanel(workspace.body("dataset"));
  const lossPanel = createLossPanel(workspace.body("loss"));

  // The flow chart is structural: it depends on the sample count but not on the
  // selection, so it is built once per dataset rather than on every change.
  let flowchartFor: string | null = null;
  function ensureFlowchart(): void {
    if (flowchartFor === session.datasetKey) return;
    createFlowchart(workspace.body("flowchart"), highlights, session.dataset.metadata.num_samples);
    flowchartFor = session.datasetKey;
  }

  session.subscribe(() => ensureFlowchart());
  session.subscribe((s) => cornerPlot.update(s));
  session.subscribe((s) => modelPanel.update(s));
  session.subscribe((s) => datasetPanel.update(s));
  session.subscribe((s) => lossPanel.update(s));
  session.emit();

  highlights.subscribe((target) => cornerPlot.flash(target));

  // A panel resize changes the corner plot's geometry; its own ResizeObserver
  // catches that, but Gridstack also needs a nudge once the grid settles.
  workspace.grid.on("resizestop", () => cornerPlot.update(session));

  setStatus(
    `${session.dataset.metadata.domain}: ${session.dataset.metadata.num_samples} samples, ` +
      `${session.dataset.weights.length} weight sets. ` +
      "Click a flow-chart block to highlight its subplots.",
  );
}

start().catch((error: unknown) => {
  const message = error instanceof Error ? error.message : String(error);
  setStatus(`Could not start: ${message}`, true);
  console.error(error);
});
