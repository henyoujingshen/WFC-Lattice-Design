// Copyright (c) 2026 Tang Hubocheng. All rights reserved.
// Original project code: no use or redistribution without written permission.
// See the repository LICENSE for scope, exceptions and third-party rights.
/* Bind 5×5 stencil board: click toggles coarse(1)/fine(0), sync to #wfc_stencil_signal */
async () => {
  const SIZE = 5;

  function wfcGetTextEntry(wrap) {
    return (
      wrap.querySelector("textarea[data-testid='textbox']") ||
      wrap.querySelector("textarea") ||
      wrap.querySelector("input[data-testid='textbox']") ||
      wrap.querySelector("input.scroll-hide") ||
      wrap.querySelector("input[type='text']") ||
      wrap.querySelector("input")
    );
  }

  function parseGrid(host) {
    try {
      const b64 = host.getAttribute("data-grid-b64");
      if (b64) {
        const obj = JSON.parse(atob(b64));
        if (obj && obj.grid) return obj.grid;
      }
    } catch (e) {}
    return Array.from({ length: SIZE }, () => Array(SIZE).fill(0));
  }

  function pushToGradio(grid) {
    const wrap = document.getElementById("wfc_stencil_signal");
    if (!wrap) return;
    const ta = wfcGetTextEntry(wrap);
    if (!ta) return;
    const payload = JSON.stringify({ grid: grid });
    ta.value = payload;
    ta.dispatchEvent(new InputEvent("input", { bubbles: true, cancelable: true }));
    ta.dispatchEvent(new Event("change", { bubbles: true }));
  }

  function updateStats(host, grid) {
    const el = host.querySelector("#wfc-stencil-stats");
    if (!el) return;
    let n = 0;
    for (let r = 0; r < SIZE; r++) {
      for (let c = 0; c < SIZE; c++) n += grid[r][c] ? 1 : 0;
    }
    el.innerHTML =
      "已涂 <b>" + n + "</b> / " + SIZE * SIZE + " 粗单元";
  }

  function renderCells(host, grid) {
    const cells = host.querySelectorAll(".wfc-stencil-cell");
    cells.forEach(function (btn) {
      const r = parseInt(btn.getAttribute("data-r"), 10);
      const c = parseInt(btn.getAttribute("data-c"), 10);
      const on = grid[r] && grid[r][c];
      btn.classList.toggle("coarse", !!on);
      btn.classList.toggle("fine", !on);
    });
    host.setAttribute(
      "data-grid-b64",
      btoa(unescape(encodeURIComponent(JSON.stringify({ grid: grid }))))
    );
    updateStats(host, grid);
  }

  function attachStencilGrid(host) {
    if (!host || host.dataset.wfcStencilBound === "1") return;
    host.dataset.wfcStencilBound = "1";
    let grid = parseGrid(host);
    renderCells(host, grid);

    host.addEventListener("click", function (ev) {
      const btn = ev.target.closest(".wfc-stencil-cell");
      if (!btn || !host.contains(btn)) return;
      ev.preventDefault();
      const r = parseInt(btn.getAttribute("data-r"), 10);
      const c = parseInt(btn.getAttribute("data-c"), 10);
      grid[r][c] = grid[r][c] ? 0 : 1;
      renderCells(host, grid);
      pushToGradio(grid);
    });
  }

  function scan() {
    const host = document.getElementById("wfc-stencil-host");
    if (host) attachStencilGrid(host);
  }

  if (!window.__wfcStencilObserver) {
    window.__wfcStencilObserver = new MutationObserver(function () {
      scan();
    });
    window.__wfcStencilObserver.observe(document.body, {
      childList: true,
      subtree: true,
      attributes: true,
      attributeFilter: ["data-grid-b64"],
    });
  }
  scan();

  return [];
}
