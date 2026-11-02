/**
 * The flowchart-to-corner-plot link (PLAN.md section 4A, "Visual Highlighting").
 *
 * These are transient events, not state: clicking a flowchart block flashes the
 * subplots it maps to and nothing is remembered afterwards. That is why they do
 * not live on the Session.
 */

/** The clickable regions of the flow chart. */
export type HighlightTarget = "features" | "model" | "output" | "target" | "loss";

export type HighlightListener = (target: HighlightTarget) => void;

export class HighlightBus {
  private readonly listeners: HighlightListener[] = [];

  subscribe(listener: HighlightListener): void {
    this.listeners.push(listener);
  }

  emit(target: HighlightTarget): void {
    for (const listener of this.listeners) listener(target);
  }
}
