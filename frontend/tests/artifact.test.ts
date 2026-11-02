// @vitest-environment node
//
// Node, not the project's jsdom default: this suite reads the snapshot off disk
// via import.meta.url, which jsdom reports as an http:// URL.

/**
 * The committed snapshot read straight off disk and pushed through the same
 * validator the app uses. This is the frontend half of the contract in
 * DATA.md section 4: if the pipeline changes shape, this fails before a panel
 * does.
 */

import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";

import { ArtifactError, validateArtifact, weightSetById } from "../src/model/artifact.ts";
import { DATASET_KEYS, SOURCE_LABELS } from "../src/model/types.ts";

const SNAPSHOT_PATH = fileURLToPath(new URL("../public/data/model_runs.json", import.meta.url));
const snapshot = JSON.parse(readFileSync(SNAPSHOT_PATH, "utf8")) as unknown;

describe("the committed snapshot", () => {
  const artifact = validateArtifact(snapshot);

  it("holds both datasets", () => {
    expect(Object.keys(artifact.datasets).sort()).toEqual([...DATASET_KEYS].sort());
  });

  it.each([...DATASET_KEYS])("%s: rows line up with num_samples", (key) => {
    const dataset = artifact.datasets[key];
    const n = dataset.metadata.num_samples;
    expect(dataset.data.x).toHaveLength(n);
    expect(dataset.data.y).toHaveLength(n);
    for (const entry of dataset.weights) {
      expect(entry.outputs).toHaveLength(n);
      expect(entry.sample_losses).toHaveLength(n);
    }
  });

  it.each([...DATASET_KEYS])("%s: carries six weight sets of 387 parameters", (key) => {
    const dataset = artifact.datasets[key];
    expect(dataset.weights.map((entry) => entry.id)).toEqual([
      "random_1",
      "random_2",
      "random_3",
      "trained_opt_1",
      "trained_opt_2",
      "trained_opt_3",
    ]);
    for (const entry of dataset.weights) {
      expect(entry.parameters.num_params).toBe(387);
      expect(entry.parameters.layers.map((l) => [l.in_features, l.out_features])).toEqual([
        [3, 16],
        [16, 16],
        [16, 3],
      ]);
    }
  });

  it.each([...DATASET_KEYS])("%s: default_weight_id resolves and is pretrained", (key) => {
    const dataset = artifact.datasets[key];
    const entry = weightSetById(dataset, dataset.metadata.default_weight_id);
    expect(entry.is_trained).toBe(true);
  });

  it.each([...DATASET_KEYS])("%s: has a source the UI can label", (key) => {
    const { source } = artifact.datasets[key].metadata;
    expect(SOURCE_LABELS[source]).toBeTruthy();
    // The committed snapshot is generated, not observed. If that changes, so
    // does this expectation -- deliberately.
    expect(source).toBe("synthetic");
  });

  it("names three features and three targets per dataset", () => {
    for (const key of DATASET_KEYS) {
      const { feature_names, target_names } = artifact.datasets[key].metadata;
      expect(feature_names).toHaveLength(3);
      expect(target_names).toHaveLength(3);
    }
  });

  it("gives every quantile range as an ordered pair", () => {
    for (const key of DATASET_KEYS) {
      const { quantiles } = artifact.datasets[key].metadata;
      for (const ranges of [quantiles.features, quantiles.targets, quantiles.outputs]) {
        expect(ranges).toHaveLength(3);
        for (const [lo, hi] of ranges) expect(lo).toBeLessThanOrEqual(hi);
      }
    }
  });

  it("pairs each loss name with its task type", () => {
    const { atmospheric_regression, stellar_classification } = artifact.datasets;
    expect(atmospheric_regression.metadata.task_type).toBe("regression");
    expect(atmospheric_regression.metadata.loss_name).toBe("mse");
    expect(stellar_classification.metadata.task_type).toBe("classification");
    expect(stellar_classification.metadata.loss_name).toBe("cross_entropy");
  });
});

describe("validateArtifact", () => {
  it("locates a missing dataset", () => {
    expect(() => validateArtifact({ datasets: {} })).toThrow(ArtifactError);
    expect(() => validateArtifact({ datasets: {} })).toThrow(/missing/);
  });

  it("rejects a row count that disagrees with num_samples", () => {
    const broken = structuredClone(snapshot) as {
      datasets: { atmospheric_regression: { data: { x: number[][] } } };
    };
    broken.datasets.atmospheric_regression.data.x.pop();
    expect(() => validateArtifact(broken)).toThrow(/feature rows/);
  });

  it("rejects a default_weight_id that names nothing", () => {
    const broken = structuredClone(snapshot) as {
      datasets: { stellar_classification: { metadata: { default_weight_id: string } } };
    };
    broken.datasets.stellar_classification.metadata.default_weight_id = "nope";
    expect(() => validateArtifact(broken)).toThrow(/is not a weight set id/);
  });

  it("rejects a non-finite value in the data", () => {
    const broken = structuredClone(snapshot) as {
      datasets: { atmospheric_regression: { data: { y: number[][] } } };
    };
    broken.datasets.atmospheric_regression.data.y[0]![0] = Number.NaN;
    expect(() => validateArtifact(broken)).toThrow(/non-finite/);
  });
});
