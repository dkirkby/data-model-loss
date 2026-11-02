/**
 * The index arithmetic behind the corner plot and the network diagram.
 *
 * Kept pure and separate because these are the places a silent off-by-one
 * produces a plot or a diagram that looks entirely plausible and is wrong: the
 * weight matrices are [out][in], and corner-plot columns 3-5 mean targets below
 * the diagonal and model outputs above it.
 */

import type { EdgeRef, NodeRef } from "nnet-svg";

import type { DatasetMetadata, LayerParameters, Range } from "../model/types.ts";
import type { HighlightTarget } from "../model/highlight.ts";

/** Variables along each edge of the matrix: x1 x2 x3 y1 y2 y3. */
export const SIZE = 6;
/** Indices below this are features; at or above it, targets/outputs. */
export const FEATURES = 3;

/**
 * Whether a cell sits in the upper triangle, which is what decides if indices
 * 3-5 read as model outputs rather than targets.
 */
export function isUpper(row: number, col: number): boolean {
  return row < col;
}

/**
 * Whether a cell shows a model output on either axis, and so changes when Theta
 * changes. Drives the mark color: dataset-only cells are amber, these are blue.
 */
export function involvesOutput(row: number, col: number): boolean {
  return isUpper(row, col) && (row >= FEATURES || col >= FEATURES);
}

/**
 * Shared axis limits per index. Indices 3-5 must hold targets *and* outputs on
 * one scale, so their range is the union: a histogram pair on the diagonal is
 * only comparable if both are drawn against the same axis.
 */
export function axisDomains(quantiles: DatasetMetadata["quantiles"]): Range[] {
  const domains: Range[] = [];
  for (let i = 0; i < FEATURES; i += 1) {
    domains.push(quantiles.features[i] ?? [0, 1]);
  }
  for (let i = 0; i < FEATURES; i += 1) {
    const target = quantiles.targets[i] ?? [0, 1];
    const output = quantiles.outputs[i] ?? [0, 1];
    domains.push([Math.min(target[0], output[0]), Math.max(target[1], output[1])]);
  }
  return domains;
}

export interface ColumnSource {
  x: number[][];
  y: number[][];
  outputs: number[][];
}

/** The values plotted along one axis of a cell. */
export function columnValues(source: ColumnSource, index: number, upper: boolean): number[] {
  if (index < FEATURES) {
    return source.x.map((row) => row[index] ?? Number.NaN);
  }
  const rows = upper ? source.outputs : source.y;
  const component = index - FEATURES;
  return rows.map((row) => row[component] ?? Number.NaN);
}

/**
 * Which cells a flowchart block highlights (PLAN.md 4A, "Visual Highlighting").
 * FEATURES is the feature block; TARGET everything at or below the diagonal in
 * the target rows; MODEL and OUTPUT everything at or above it in the output
 * columns; the loss link just the four-to-six diagonal histograms.
 */
export function cellsFor(target: HighlightTarget): (row: number, col: number) => boolean {
  switch (target) {
    case "features":
      return (row, col) => row < FEATURES && col < FEATURES;
    case "target":
      return (row, col) => row >= FEATURES && col <= row;
    case "model":
    case "output":
      return (row, col) => col >= FEATURES && row <= col;
    case "loss":
      return (row, col) => row === col && row >= FEATURES;
  }
}

/**
 * w for the edge from node i of layer L to node j of layer L+1. The exported
 * matrices are row-major [out_features][in_features], matching torch's
 * Linear.weight, so the target index comes first.
 */
export function weightFor(layers: LayerParameters[], edge: EdgeRef): number | undefined {
  return layers[edge.sourceLayerIndex]?.weights[edge.targetNodeIndex]?.[edge.sourceNodeIndex];
}

/** b for a node. Layer 0 is the input and carries no bias. */
export function biasFor(layers: LayerParameters[], node: NodeRef): number | undefined {
  if (node.layerIndex === 0) return undefined;
  return layers[node.layerIndex - 1]?.biases[node.nodeIndex];
}


/** The shape d3.bin returns, reduced to what the outline needs. */
export interface HistogramBin {
  x0?: number;
  x1?: number;
  length: number;
}

/**
 * The polyline of an unfilled step histogram, matplotlib's histtype="step".
 *
 * Up from the axis at the left edge, horizontal across each bin, a vertical only
 * at the boundary *between* adjacent bins, and back down to the axis at the
 * right edge. Nothing drops to the axis mid-figure: a line after every bin draws
 * a comb, which reads as a bevel or a drop shadow rather than an outline.
 */
export function stepOutlinePoints(
  bins: HistogramBin[],
  x: (value: number) => number,
  height: (count: number) => number,
  base: number,
): [number, number][] {
  const points: [number, number][] = [];
  let lastRight: number | null = null;

  for (const item of bins) {
    if (item.x0 === undefined || item.x1 === undefined) continue;
    const left = x(item.x0);
    const right = x(item.x1);
    const top = base - height(item.length);
    if (points.length === 0) points.push([left, base]);
    points.push([left, top], [right, top]);
    lastRight = right;
  }

  // An empty final bin already ends on the axis; closing again would emit a
  // duplicate point.
  const last = points[points.length - 1];
  if (lastRight !== null && !(last && last[0] === lastRight && last[1] === base)) {
    points.push([lastRight, base]);
  }
  return points;
}
