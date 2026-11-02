/**
 * Phase 2 step 2 is "install dependencies". This asserts each one actually
 * imports and exposes what the Phase 3 panels will reach for, so a broken or
 * mis-versioned package surfaces here rather than mid-build.
 */

import { describe, expect, it } from "vitest";

describe("the four visualization dependencies", () => {
  it("gridstack exposes GridStack.init for the panel canvas", async () => {
    const { GridStack } = await import("gridstack");
    expect(typeof GridStack.init).toBe("function");
  });

  it("d3 exposes the scales and selections the corner plot needs", async () => {
    const d3 = await import("d3");
    expect(typeof d3.scaleLinear).toBe("function");
    expect(typeof d3.select).toBe("function");
    expect(typeof d3.bin).toBe("function");
  });

  it("katex renders the loss equations", async () => {
    const katex = await import("katex");
    const html = katex.default.renderToString("L_{\\mathrm{MSE}}");
    expect(html).toContain("katex");
  });

  it("nnet-svg 0.1.1 exposes its dense-network renderers", async () => {
    const nnet = await import("nnet-svg");
    expect(typeof nnet.layoutDenseNetwork).toBe("function");
    expect(typeof nnet.createDenseNetworkSvg).toBe("function");
    expect(typeof nnet.renderDenseNetworkSvg).toBe("function");
  });
});
