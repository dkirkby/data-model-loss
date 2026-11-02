/**
 * The per-sample loss color ramp, shared by the corner plot's "color by loss"
 * mode and the loss histogram in Panel D. One definition, so a point and the bar
 * it falls into are always the same color.
 *
 * The scale is logarithmic. Loss distributions are heavily right-skewed -- a
 * trained model puts most samples very close to zero and a few far out -- so on
 * a linear ramp almost every point lands in the same dark band and the coloring
 * says nothing. In log space the well-fit bulk spreads out, which is where the
 * interesting structure is.
 */

import { interpolateViridis, quantile, scaleLog } from "d3";

/** Spread beyond this many decades and the low end stops being informative. */
const MAX_DECADES = 4;

export interface LossRamp {
  /** Viridis over log(L), clamped to the domain below. */
  color(value: number): string;
  /** The color at a fraction of the way along the ramp; for drawing a color bar. */
  colorAt(fraction: number): string;
  /** Where a loss sits along the ramp, 0 to 1. */
  position(value: number): number;
  /** Round values to label a color bar with, at most `count` of them. */
  ticks(count: number): number[];
  /** [low, high] -- high is the 99th percentile, low a positive floor. */
  domain: [number, number];
  /** Always true; kept explicit so callers can label their axes honestly. */
  logarithmic: true;
}

function ramp(low: number, high: number): LossRamp {
  const scale = scaleLog().domain([low, high]).range([0, 1]).clamp(true);
  // log(0) is -Infinity, and a zero loss is entirely possible, so the floor is
  // applied before the scale rather than relying on clamping.
  const position = (value: number): number => scale(Math.max(value, low));
  return {
    domain: [low, high],
    logarithmic: true,
    position,
    colorAt: interpolateViridis,
    color(value: number): string {
      return interpolateViridis(position(value));
    },
    ticks(count: number): number[] {
      // A log scale hands back every power of ten in range and then some, so
      // thin them to roughly the requested count, keeping the ends.
      const candidates = scale.ticks(count).filter((value) => value >= low && value <= high);
      if (candidates.length <= count) return candidates;
      const stride = Math.ceil(candidates.length / count);
      return candidates.filter((_, index) => index % stride === 0);
    },
  };
}

export function lossRamp(losses: number[]): LossRamp {
  if (losses.length === 0) return ramp(1e-6, 1);

  const cut = quantile(losses, 0.99) ?? Math.max(...losses);
  const positive = losses.filter((value) => value > 0);
  const smallest = positive.length > 0 ? Math.min(...positive) : 0;

  const high = cut > 0 ? cut : 1;
  // The floor is the smallest loss actually present, but never more than
  // MAX_DECADES below the top: one sample at 1e-12 should not flatten the rest.
  const floor = high / 10 ** MAX_DECADES;
  const low = smallest > 0 ? Math.max(smallest, floor) : floor;

  return ramp(low, low < high ? high : low * 10);
}
