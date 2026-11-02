// @vitest-environment node

import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { beforeEach, describe, expect, it } from "vitest";

import { validateArtifact } from "../src/model/artifact.ts";
import { Session } from "../src/model/session.ts";

const SNAPSHOT_PATH = fileURLToPath(new URL("../public/data/model_runs.json", import.meta.url));
const artifact = validateArtifact(JSON.parse(readFileSync(SNAPSHOT_PATH, "utf8")));

describe("Session", () => {
  let session: Session;
  let notifications: number;

  beforeEach(() => {
    session = new Session(artifact, "atmospheric_regression");
    notifications = 0;
    session.subscribe(() => {
      notifications += 1;
    });
  });

  it("opens on the dataset's default weight set", () => {
    expect(session.weightSet.id).toBe(
      artifact.datasets.atmospheric_regression.metadata.default_weight_id,
    );
    expect(session.weightSet.is_trained).toBe(true);
  });

  it("shows both series and no loss coloring to start", () => {
    expect(session.showData).toBe(true);
    expect(session.showPredictions).toBe(true);
    expect(session.colorByLoss).toBe(false);
  });

  it("resets the weight selection when the dataset changes", () => {
    session.selectWeights("random_3");
    expect(session.weightSet.id).toBe("random_3");

    session.selectDataset("stellar_classification");
    // Weight ids are per-dataset, so carrying "random_3" across would be a
    // different set of numbers under the same name.
    expect(session.weightSet.id).toBe(
      artifact.datasets.stellar_classification.metadata.default_weight_id,
    );
  });

  it("rejects a weight id from the other dataset", () => {
    expect(() => session.selectWeights("not_a_set")).toThrow();
  });

  it("notifies once per real change and not at all for a no-op", () => {
    session.setShowData(false);
    session.setShowPredictions(false);
    session.setColorByLoss(true);
    expect(notifications).toBe(3);

    session.setShowData(false);
    session.setColorByLoss(true);
    session.selectDataset("atmospheric_regression");
    expect(notifications).toBe(3);
  });

  it("keeps the visibility toggles across a dataset switch", () => {
    session.setShowPredictions(false);
    session.selectDataset("stellar_classification");
    expect(session.showPredictions).toBe(false);
  });
});
