/**
 * Panel D: the loss definition, rendered with KaTeX (PLAN.md 4D).
 *
 * The formula follows the active dataset -- MSE for the regression task,
 * categorical cross-entropy for the classifier -- and the running total for the
 * selected Theta sits underneath it, so the equation and the number a student is
 * looking at are never out of step.
 */

import katex from "katex";

import type { Session } from "../model/session.ts";
import type { LossName } from "../model/types.ts";

interface LossCopy {
  title: string;
  formula: string;
  /** What the formula means, in the language of the active task. */
  explain: string;
  /** The unit or space the reported number lives in. */
  units: string;
}

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

  const formula = document.createElement("div");
  formula.className = "loss-formula";

  const total = document.createElement("p");
  total.className = "badge";

  const explain = document.createElement("p");
  explain.className = "loss-explain";

  const units = document.createElement("p");
  units.className = "loss-units";

  host.append(title, formula, total, explain, units);

  return {
    update(session: Session): void {
      const copy = COPY[session.dataset.metadata.loss_name];
      title.textContent = copy.title;
      // KaTeX writes its own markup; throwOnError is off so a formula typo
      // degrades to visible source rather than taking the panel down.
      katex.render(copy.formula, formula, {
        displayMode: true,
        throwOnError: false,
        output: "html",
      });
      const weights = session.weightSet;
      total.textContent =
        `mean over ${session.dataset.metadata.num_samples} samples = ` +
        `${weights.total_loss.toFixed(6)} — ${weights.name}`;
      explain.textContent = copy.explain;
      units.textContent = copy.units;
    },
    destroy(): void {
      host.replaceChildren();
    },
  };
}
