// @vitest-environment node
//
// Pure index arithmetic: no DOM needed, and these are the rules that decide
// what every subplot and every edge of the diagram actually shows.

import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";

import { validateArtifact } from "../src/model/artifact.ts";
import {
  axisDomains,
  biasFor,
  cellsFor,
  columnValues,
  FEATURES,
  involvesOutput,
  isUpper,
  SIZE,
  weightFor,
} from "../src/view/mapping.ts";

const SNAPSHOT_PATH = fileURLToPath(new URL("../public/data/model_runs.json", import.meta.url));
const artifact = validateArtifact(JSON.parse(readFileSync(SNAPSHOT_PATH, "utf8")));

/** Every cell of the matrix, as [row, col]. */
const allCells: [number, number][] = [];
for (let row = 0; row < SIZE; row += 1) {
  for (let col = 0; col < SIZE; col += 1) allCells.push([row, col]);
}

function matching(target: Parameters<typeof cellsFor>[0]): string[] {
  const predicate = cellsFor(target);
  return allCells.filter(([r, c]) => predicate(r, c)).map(([r, c]) => `${r},${c}`);
}

describe("the four blocks of the matrix", () => {
  it("reads indices 3-5 as targets below the diagonal and outputs above it", () => {
    // Bottom-right lower triangle: y vs y.
    expect(isUpper(4, 3)).toBe(false);
    // Bottom-right upper triangle: yhat vs yhat.
    expect(isUpper(3, 4)).toBe(true);
    // Top-right is always upper, so features vs outputs.
    expect(isUpper(0, 5)).toBe(true);
    // Bottom-left is always lower, so features vs targets.
    expect(isUpper(5, 0)).toBe(false);
  });

  it("marks exactly the yhat subplots as parameter-dependent", () => {
    // Top-right block and the bottom-right upper triangle, nothing else.
    expect(involvesOutput(0, 3)).toBe(true);
    expect(involvesOutput(3, 4)).toBe(true);
    // Feature space does not move when Theta changes.
    expect(involvesOutput(0, 1)).toBe(false);
    // Nor does features-vs-targets or targets-vs-targets.
    expect(involvesOutput(4, 0)).toBe(false);
    expect(involvesOutput(5, 3)).toBe(false);
  });
});

describe("cellsFor", () => {
  it("highlights the 3x3 feature block for FEATURES", () => {
    expect(matching("features")).toEqual([
      "0,0", "0,1", "0,2",
      "1,0", "1,1", "1,2",
      "2,0", "2,1", "2,2",
    ]);
  });

  it("highlights the target rows at or below the diagonal for TARGET", () => {
    expect(matching("target")).toEqual([
      "3,0", "3,1", "3,2", "3,3",
      "4,0", "4,1", "4,2", "4,3", "4,4",
      "5,0", "5,1", "5,2", "5,3", "5,4", "5,5",
    ]);
  });

  it("highlights the same yhat subplots for MODEL and OUTPUT", () => {
    expect(matching("model")).toEqual(matching("output"));
    expect(matching("output")).toEqual([
      "0,3", "0,4", "0,5",
      "1,3", "1,4", "1,5",
      "2,3", "2,4", "2,5",
      "3,3", "3,4", "3,5",
      "4,4", "4,5",
      "5,5",
    ]);
  });

  it("highlights only the output/target diagonal for the loss link", () => {
    expect(matching("loss")).toEqual(["3,3", "4,4", "5,5"]);
  });

  it("never highlights a target cell and an output cell for the same click", () => {
    // The two sets meet only on the diagonal, where both are drawn.
    const targets = new Set(matching("target"));
    const outputs = matching("output").filter((cell) => targets.has(cell));
    expect(outputs).toEqual(["3,3", "4,4", "5,5"]);
  });
});

describe("axisDomains", () => {
  const { quantiles } = artifact.datasets.atmospheric_regression.metadata;
  const domains = axisDomains(quantiles);

  it("gives one domain per variable", () => {
    expect(domains).toHaveLength(SIZE);
  });

  it("passes feature quantiles through unchanged", () => {
    expect(domains.slice(0, FEATURES)).toEqual(quantiles.features);
  });

  it("unions targets with outputs so the diagonal histograms share an axis", () => {
    for (let i = 0; i < FEATURES; i += 1) {
      const target = quantiles.targets[i]!;
      const output = quantiles.outputs[i]!;
      const shared = domains[FEATURES + i]!;
      expect(shared[0]).toBeLessThanOrEqual(Math.min(target[0], output[0]));
      expect(shared[1]).toBeGreaterThanOrEqual(Math.max(target[1], output[1]));
    }
  });
});

describe("columnValues", () => {
  const dataset = artifact.datasets.atmospheric_regression;
  const weights = dataset.weights[3]!;
  const source = { x: dataset.data.x, y: dataset.data.y, outputs: weights.outputs };

  it("returns features for indices 0-2, whichever triangle", () => {
    for (const upper of [true, false]) {
      expect(columnValues(source, 1, upper)).toEqual(dataset.data.x.map((row) => row[1]));
    }
  });

  it("returns targets below the diagonal and outputs above it for indices 3-5", () => {
    expect(columnValues(source, 4, false)).toEqual(dataset.data.y.map((row) => row[1]));
    expect(columnValues(source, 4, true)).toEqual(weights.outputs.map((row) => row[1]));
  });

  it("produces one value per sample", () => {
    expect(columnValues(source, 5, true)).toHaveLength(dataset.metadata.num_samples);
  });
});

describe("weightFor and biasFor", () => {
  const layers = artifact.datasets.atmospheric_regression.weights[0]!.parameters.layers;

  it("indexes the [out][in] matrices with the target node first", () => {
    // The 16x16 middle layer, where both the value and its transpose exist, so
    // reading the matrix the wrong way round would go unnoticed.
    const value = weightFor(layers, {
      kind: "edge",
      sourceLayerIndex: 1,
      sourceNodeIndex: 2,
      targetLayerIndex: 2,
      targetNodeIndex: 7,
      edgeIndex: 0,
    });
    const transposed = layers[1]!.weights[2]![7];
    expect(value).toBe(layers[1]!.weights[7]![2]);
    expect(typeof transposed).toBe("number");
    expect(value).not.toBe(transposed);
  });

  it("covers every edge of the three layer pairs", () => {
    const pairs: [number, number, number][] = [
      [0, 3, 16],
      [1, 16, 16],
      [2, 16, 3],
    ];
    let seen = 0;
    for (const [layer, inputs, outputs] of pairs) {
      for (let i = 0; i < inputs; i += 1) {
        for (let j = 0; j < outputs; j += 1) {
          const value = weightFor(layers, {
            kind: "edge",
            sourceLayerIndex: layer,
            sourceNodeIndex: i,
            targetLayerIndex: layer + 1,
            targetNodeIndex: j,
            edgeIndex: seen,
          });
          expect(typeof value).toBe("number");
          seen += 1;
        }
      }
    }
    expect(seen).toBe(3 * 16 + 16 * 16 + 16 * 3);
  });

  it("gives the input layer no bias and every other node one", () => {
    expect(biasFor(layers, { kind: "node", layerIndex: 0, nodeIndex: 1 })).toBeUndefined();
    for (const [layerIndex, count] of [
      [1, 16],
      [2, 16],
      [3, 3],
    ] as const) {
      for (let nodeIndex = 0; nodeIndex < count; nodeIndex += 1) {
        expect(typeof biasFor(layers, { kind: "node", layerIndex, nodeIndex })).toBe("number");
      }
    }
  });

  it("accounts for all 387 parameters between weights and biases", () => {
    const weightCount = layers.reduce((sum, l) => sum + l.weights.flat().length, 0);
    const biasCount = layers.reduce((sum, l) => sum + l.biases.length, 0);
    expect(weightCount + biasCount).toBe(387);
  });
});
