/**
 * Loading and validating the static artifact.
 *
 * `validateArtifact` is deliberately free of DOM and fetch so the test suite can
 * run it against the committed snapshot straight off disk: it is what certifies
 * that the Python pipeline and this app still agree.
 */

import type { Artifact, Dataset, DatasetKey, Range } from "./types.ts";
import { DATASET_KEYS } from "./types.ts";

export class ArtifactError extends Error {}

function fail(message: string): never {
  throw new ArtifactError(message);
}

function asRecord(value: unknown, where: string): Record<string, unknown> {
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    fail(`${where}: expected an object`);
  }
  return value as Record<string, unknown>;
}

function asNumberMatrix(value: unknown, where: string, columns: number): number[][] {
  if (!Array.isArray(value)) fail(`${where}: expected an array of rows`);
  value.forEach((row, i) => {
    if (!Array.isArray(row) || row.length !== columns) {
      fail(`${where}[${i}]: expected ${columns} numbers`);
    }
    if (row.some((v) => typeof v !== "number" || !Number.isFinite(v))) {
      fail(`${where}[${i}]: contains a non-finite value`);
    }
  });
  return value as number[][];
}

function asRanges(value: unknown, where: string): Range[] {
  if (!Array.isArray(value)) fail(`${where}: expected an array of ranges`);
  value.forEach((range, i) => {
    if (!Array.isArray(range) || range.length !== 2) fail(`${where}[${i}]: expected [lo, hi]`);
    if (range[0] > range[1]) fail(`${where}[${i}]: lo exceeds hi`);
  });
  return value as Range[];
}

function validateDataset(value: unknown, key: string): Dataset {
  const dataset = asRecord(value, key);
  const metadata = asRecord(dataset.metadata, `${key}.metadata`);
  const data = asRecord(dataset.data, `${key}.data`);

  const n = metadata.num_samples;
  if (typeof n !== "number" || n <= 0) fail(`${key}.metadata.num_samples: expected a count`);

  for (const field of ["feature_names", "target_names"] as const) {
    const names = metadata[field];
    if (!Array.isArray(names) || names.length !== 3) {
      fail(`${key}.metadata.${field}: expected 3 names`);
    }
  }

  const x = asNumberMatrix(data.x, `${key}.data.x`, 3);
  const y = asNumberMatrix(data.y, `${key}.data.y`, 3);
  if (x.length !== n || y.length !== n) {
    fail(`${key}.data: ${x.length} feature rows and ${y.length} target rows for num_samples=${n}`);
  }

  const quantiles = asRecord(metadata.quantiles, `${key}.metadata.quantiles`);
  for (const field of ["features", "targets", "outputs"] as const) {
    asRanges(quantiles[field], `${key}.metadata.quantiles.${field}`);
  }

  if (!Array.isArray(dataset.weights) || dataset.weights.length === 0) {
    fail(`${key}.weights: expected at least one weight set`);
  }
  dataset.weights.forEach((entry, i) => {
    const weightSet = asRecord(entry, `${key}.weights[${i}]`);
    const parameters = asRecord(weightSet.parameters, `${key}.weights[${i}].parameters`);
    if (typeof parameters.num_params !== "number") {
      fail(`${key}.weights[${i}].parameters.num_params: expected a number`);
    }
    asNumberMatrix(weightSet.outputs, `${key}.weights[${i}].outputs`, 3);
    const losses = weightSet.sample_losses;
    if (!Array.isArray(losses) || losses.length !== n) {
      fail(`${key}.weights[${i}].sample_losses: expected ${n} values`);
    }
    if ((weightSet.outputs as number[][]).length !== n) {
      fail(`${key}.weights[${i}].outputs: expected ${n} rows`);
    }
  });

  const ids = dataset.weights.map((entry) => (entry as { id: string }).id);
  if (!ids.includes(metadata.default_weight_id as string)) {
    fail(`${key}.metadata.default_weight_id: "${metadata.default_weight_id}" is not a weight set id`);
  }

  return dataset as unknown as Dataset;
}

/** Throws ArtifactError with a located message rather than letting a panel fail later. */
export function validateArtifact(value: unknown): Artifact {
  const root = asRecord(value, "artifact");
  const datasets = asRecord(root.datasets, "artifact.datasets");

  const missing = DATASET_KEYS.filter((key) => !(key in datasets));
  if (missing.length > 0) fail(`artifact.datasets: missing ${missing.join(", ")}`);

  for (const key of DATASET_KEYS) {
    validateDataset(datasets[key], `datasets.${key}`);
  }
  return root as unknown as Artifact;
}

/**
 * Fetch the snapshot. BASE_URL, not a leading slash: the deployed site lives
 * under /data-model-loss/ on GitHub Pages.
 */
export async function loadArtifact(): Promise<Artifact> {
  const url = `${import.meta.env.BASE_URL}data/model_runs.json`;
  const response = await fetch(url);
  if (!response.ok) {
    throw new ArtifactError(`${url}: ${response.status} ${response.statusText}`);
  }
  return validateArtifact(await response.json());
}

/** The weight set a dataset opens on, per metadata.default_weight_id. */
export function defaultWeightId(dataset: Dataset): string {
  return dataset.metadata.default_weight_id;
}

export function weightSetById(dataset: Dataset, id: string) {
  const found = dataset.weights.find((entry) => entry.id === id);
  if (!found) fail(`no weight set "${id}" in this dataset`);
  return found;
}

export function datasetKeys(): readonly DatasetKey[] {
  return DATASET_KEYS;
}
