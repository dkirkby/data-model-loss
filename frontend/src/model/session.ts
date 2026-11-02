/**
 * What the instructor has selected right now, and who needs to know when it
 * changes. Panels subscribe; nothing reaches into another panel directly.
 */

import type { Artifact, Dataset, DatasetKey, WeightSet } from "./types.ts";
import { weightSetById } from "./artifact.ts";

export interface SessionState {
  datasetKey: DatasetKey;
  weightId: string;
  /** PLAN.md section 4A: "points colored by loss value". */
  colorByLoss: boolean;
}

export type SessionListener = (session: Session) => void;

export class Session {
  private readonly listeners: SessionListener[] = [];
  private readonly artifact: Artifact;
  private state: SessionState;

  constructor(artifact: Artifact, datasetKey: DatasetKey) {
    this.artifact = artifact;
    this.state = {
      datasetKey,
      weightId: artifact.datasets[datasetKey].metadata.default_weight_id,
      colorByLoss: false,
    };
  }

  get dataset(): Dataset {
    return this.artifact.datasets[this.state.datasetKey];
  }

  get weightSet(): WeightSet {
    return weightSetById(this.dataset, this.state.weightId);
  }

  get datasetKey(): DatasetKey {
    return this.state.datasetKey;
  }

  get colorByLoss(): boolean {
    return this.state.colorByLoss;
  }

  /** Switching dataset resets the weight selection to that dataset's default. */
  selectDataset(datasetKey: DatasetKey): void {
    if (datasetKey === this.state.datasetKey) return;
    this.state = {
      ...this.state,
      datasetKey,
      weightId: this.artifact.datasets[datasetKey].metadata.default_weight_id,
    };
    this.emit();
  }

  selectWeights(weightId: string): void {
    if (weightId === this.state.weightId) return;
    weightSetById(this.dataset, weightId); // throws if it is not in this dataset
    this.state = { ...this.state, weightId };
    this.emit();
  }

  setColorByLoss(colorByLoss: boolean): void {
    if (colorByLoss === this.state.colorByLoss) return;
    this.state = { ...this.state, colorByLoss };
    this.emit();
  }

  subscribe(listener: SessionListener): void {
    this.listeners.push(listener);
  }

  /** Call once after wiring, so every panel renders its initial state. */
  emit(): void {
    for (const listener of this.listeners) listener(this);
  }
}
