// @vitest-environment node

import { describe, expect, it } from "vitest";

import { lossRamp } from "../src/view/lossramp.ts";

describe("lossRamp", () => {
  it("runs from the smallest positive loss to the 99th percentile", () => {
    const losses = Array.from({ length: 100 }, (_, i) => i); // 0 .. 99
    const { domain, logarithmic } = lossRamp(losses);
    expect(logarithmic).toBe(true);
    // A zero loss cannot be the low end of a log scale; 1 is the smallest
    // positive value present.
    expect(domain[0]).toBe(1);
    // Not the maximum: the top percentile is cut off deliberately.
    expect(domain[1]).toBeLessThan(99);
    expect(domain[1]).toBeGreaterThan(90);
  });

  it("never spreads the ramp over more than four decades", () => {
    // One sample at 1e-12 must not push the low end twelve decades down.
    const losses = [1e-12, ...Array.from({ length: 99 }, () => 1)];
    const { domain } = lossRamp(losses);
    expect(domain[1] / domain[0]).toBeLessThanOrEqual(10 ** 4 + 1e-9);
    expect(domain[0]).toBeGreaterThan(1e-12);
  });

  it("colors a zero loss as the low end rather than failing on log(0)", () => {
    const ramp = lossRamp([0, 0.01, 0.1, 1]);
    expect(ramp.color(0)).toBe(ramp.color(ramp.domain[0]));
    expect(ramp.color(0)).toMatch(/^#[0-9a-f]{6}$/i);
  });

  it("spreads a right-skewed bulk that a linear ramp would collapse", () => {
    // Typical of a trained model: most samples tiny, a few far out.
    const losses = [...Array.from({ length: 98 }, (_, i) => 0.001 * (i + 1)), 5, 40];
    const ramp = lossRamp(losses);
    // Two samples an order of magnitude apart near the bottom must not share
    // a color, which is exactly what fails on a linear scale.
    expect(ramp.color(0.002)).not.toBe(ramp.color(0.02));
  });

  it("clamps above the cut instead of extending the scale", () => {
    // One sample 1000x worse than the rest must not flatten the ramp.
    const losses = [...Array.from({ length: 99 }, () => 0.1), 100];
    const ramp = lossRamp(losses);
    expect(ramp.domain[1]).toBeLessThan(100);
    expect(ramp.color(100)).toBe(ramp.color(ramp.domain[1]));
    expect(ramp.color(1e9)).toBe(ramp.color(ramp.domain[1]));
  });

  it("maps low loss and high loss to different colors", () => {
    const ramp = lossRamp([0, 0.25, 0.5, 0.75, 1]);
    expect(ramp.color(0)).not.toBe(ramp.color(1));
    expect(ramp.color(0)).toMatch(/^#[0-9a-f]{6}$/i);
  });

  it("survives a degenerate distribution where every loss is equal", () => {
    const ramp = lossRamp([0.5, 0.5, 0.5]);
    expect(ramp.domain[0]).toBeLessThanOrEqual(0.5);
    expect(ramp.domain[1]).toBeGreaterThan(ramp.domain[0]);
    expect(ramp.color(0.5)).toMatch(/^#[0-9a-f]{6}$/i);
  });

  it("handles an empty list without throwing", () => {
    const ramp = lossRamp([]);
    expect(ramp.domain[0]).toBeGreaterThan(0);
    expect(ramp.domain[1]).toBeGreaterThan(ramp.domain[0]);
    expect(ramp.color(0)).toMatch(/^#[0-9a-f]{6}$/i);
  });
});

describe("lossRamp color bar support", () => {
  const ramp = lossRamp([0.001, 0.01, 0.1, 1, 10]);

  it("places the domain ends at the ends of the bar", () => {
    expect(ramp.position(ramp.domain[0])).toBeCloseTo(0, 10);
    expect(ramp.position(ramp.domain[1])).toBeCloseTo(1, 10);
  });

  it("clamps outside the domain instead of running off the bar", () => {
    expect(ramp.position(0)).toBe(0);
    expect(ramp.position(1e9)).toBe(1);
  });

  it("puts the geometric midpoint halfway along, because the scale is log", () => {
    const [low, high] = ramp.domain;
    const geometric = Math.sqrt(low * high);
    expect(ramp.position(geometric)).toBeCloseTo(0.5, 6);
    // The arithmetic midpoint sits far up the bar, which is the whole point of
    // using a log scale for a right-skewed quantity.
    expect(ramp.position((low + high) / 2)).toBeGreaterThan(0.8);
  });

  it("colors the bar from the same function that colors the points", () => {
    for (const value of [ramp.domain[0], 0.05, ramp.domain[1]]) {
      expect(ramp.color(value)).toBe(ramp.colorAt(ramp.position(value)));
    }
  });

  it("gives tick values inside the domain, no more than asked for", () => {
    const ticks = ramp.ticks(5);
    expect(ticks.length).toBeGreaterThan(0);
    expect(ticks.length).toBeLessThanOrEqual(5);
    for (const tick of ticks) {
      expect(tick).toBeGreaterThanOrEqual(ramp.domain[0]);
      expect(tick).toBeLessThanOrEqual(ramp.domain[1]);
    }
  });

  it("still gives ticks when the domain spans well under a decade", () => {
    const narrow = lossRamp([0.5, 0.52, 0.55, 0.6]);
    expect(narrow.ticks(5).length).toBeGreaterThan(0);
  });
});
