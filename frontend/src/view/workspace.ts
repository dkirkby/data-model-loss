/**
 * The Gridstack panel canvas.
 *
 * PLAN.md section 1: panel layout, visibility and collapse states are the
 * instructor's to manage, and nothing auto-advances. So the grid runs in
 * mode:'float' (gridstack 14's name for the old float:true) -- a panel stays
 * where it is put instead of being pulled toward the top -- and every panel can
 * be dragged, resized and collapsed independently.
 *
 * Collapse is not a Gridstack feature. It is implemented here by remembering a
 * panel's height, shrinking it to a single row (its header), and restoring the
 * remembered height on expand.
 */

import { GridStack, type GridItemHTMLElement } from "gridstack";

const COLLAPSED_ROWS = 1;

export interface Workspace {
  grid: GridStack;
  /** The body element of a panel, where a Phase 3 component mounts. */
  body(panelId: string): HTMLElement;
}

function requireElement(selector: string): HTMLElement {
  const el = document.querySelector<HTMLElement>(selector);
  if (!el) throw new Error(`${selector}: not found in index.html`);
  return el;
}

function toggleCollapse(grid: GridStack, item: GridItemHTMLElement, button: HTMLElement): void {
  const collapsed = item.classList.toggle("is-collapsed");
  if (collapsed) {
    const current = item.gridstackNode?.h ?? COLLAPSED_ROWS;
    item.dataset.expandedHeight = String(current);
    grid.update(item, { h: COLLAPSED_ROWS, noResize: true });
  } else {
    const restored = Number(item.dataset.expandedHeight ?? COLLAPSED_ROWS);
    grid.update(item, { h: restored, noResize: false });
  }
  button.setAttribute("aria-expanded", String(!collapsed));
  button.textContent = collapsed ? "+" : "–";
  button.title = collapsed ? "Expand panel" : "Collapse panel";
}

export function initWorkspace(): Workspace {
  const grid = GridStack.init(
    {
      // The panels are declared in index.html with gs-* attributes; Gridstack
      // adopts them on init.
      cellHeight: 56,
      margin: 8,
      mode: "float",
      handle: ".panel-header",
      resizable: { handles: "se, sw, e, s, w" },
      // Collapse to a single column on a narrow window so the tool stays usable
      // on a projector or a laptop screen rather than overflowing sideways.
      columnOpts: {
        breakpointForWindow: true,
        breakpoints: [{ w: 1100, c: 1 }],
      },
    },
    "#workspace",
  );
  if (!grid) throw new Error("#workspace: Gridstack failed to initialize");

  for (const button of document.querySelectorAll<HTMLElement>(".panel-collapse")) {
    const item = button.closest<GridItemHTMLElement>(".grid-stack-item");
    if (!item) continue;
    button.addEventListener("click", () => toggleCollapse(grid, item, button));
  }

  return {
    grid,
    body(panelId: string): HTMLElement {
      return requireElement(`#panel-${panelId} .panel-body`);
    },
  };
}
