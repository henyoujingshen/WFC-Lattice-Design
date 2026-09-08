// Copyright (c) 2026 Tang Hubocheng. All rights reserved.
// Original project code: no use or redistribution without written permission.
// See the repository LICENSE for scope, exceptions and third-party rights.
/* Gradio wraps this as: await (SOURCE)(...__fn_args) — SOURCE must be an async callable, NOT an IIFE. */
async () => {
  function attachWfcCanvas(canvas) {
    if (!canvas || canvas.dataset.wfcBound === "1") return;
    canvas.dataset.wfcBound = "1";

    const W = canvas.width,
      H = canvas.height;
    let splats = [];

    function parseSplats() {
      try {
        const b64 = canvas.getAttribute("data-splats-b64");
        if (b64) {
          const txt = atob(b64);
          splats = JSON.parse(txt);
        }
      } catch (e) {
        splats = [];
      }
    }

    function readDraftFromDom() {
      return {
        u1: parseFloat(canvas.getAttribute("data-u1") || "0"),
        v1: parseFloat(canvas.getAttribute("data-v1") || "0"),
        u2: parseFloat(canvas.getAttribute("data-u2") || "0"),
        v2: parseFloat(canvas.getAttribute("data-v2") || "0"),
        ang: parseFloat(canvas.getAttribute("data-ang") || "0"),
      };
    }

    let draft = readDraftFromDom();
    let drag = null;

    function clamp01(x) {
      return Math.max(0, Math.min(1, x));
    }

    function uvFromEvent(ev) {
      const rect = canvas.getBoundingClientRect();
      const lx = (ev.clientX - rect.left) / Math.max(rect.width, 1e-6);
      const ly = (ev.clientY - rect.top) / Math.max(rect.height, 1e-6);
      return { u: clamp01(lx), v: clamp01(ly) };
    }

    function drawEllipsePath(ctx, muu, muv, su, sv, angDeg) {
      const cx = muu * W,
        cy = muv * H;
      const rx = su * W,
        ry = sv * H;
      const rad = (angDeg * Math.PI) / 180;
      ctx.beginPath();
      ctx.ellipse(cx, cy, Math.max(rx, 0.5), Math.max(ry, 0.5), rad, 0, Math.PI * 2);
    }

    function draw() {
      const ctx = canvas.getContext("2d");
      if (!ctx) return;
      ctx.clearRect(0, 0, W, H);
      ctx.fillStyle = "#fafafa";
      ctx.fillRect(0, 0, W, H);
      ctx.strokeStyle = "#333";
      ctx.lineWidth = 2;
      ctx.strokeRect(0.5, 0.5, W - 1, H - 1);

      parseSplats();
      draft = readDraftFromDom();
      for (let i = 0; i < splats.length; i++) {
        const s = splats[i];
        const col = s.color === "L" ? "rgba(31,119,180,0.35)" : "rgba(214,39,40,0.35)";
        const line = s.color === "L" ? "#1f77b4" : "#d62728";
        const mu = s.mu || [0.5, 0.5];
        const sc = s.scale || [0.08, 0.08];
        const ang = s.angle || 0;
        ctx.fillStyle = col;
        ctx.strokeStyle = line;
        ctx.lineWidth = 2;
        drawEllipsePath(ctx, mu[0], mu[1], sc[0], sc[1], ang);
        ctx.fill();
        ctx.stroke();
        ctx.fillStyle = line;
        ctx.beginPath();
        ctx.arc(mu[0] * W, mu[1] * H, 4, 0, Math.PI * 2);
        ctx.fill();
      }

      let u1 = draft.u1,
        v1 = draft.v1,
        u2 = draft.u2,
        v2 = draft.v2;
      if (drag) {
        u1 = Math.min(drag.u0, drag.u1);
        u2 = Math.max(drag.u0, drag.u1);
        v1 = Math.min(drag.v0, drag.v1);
        v2 = Math.max(drag.v0, drag.v1);
      }
      const su = Math.max(Math.abs(u2 - u1) / 2, 0.01);
      const sv = Math.max(Math.abs(v2 - v1) / 2, 0.01);
      const cx = (u1 + u2) / 2,
        cy = (v1 + v2) / 2;

      ctx.setLineDash([6, 4]);
      ctx.strokeStyle = "#2ca02c";
      ctx.lineWidth = 2;
      ctx.strokeRect(u1 * W, v1 * H, (u2 - u1) * W, (v2 - v1) * H);
      ctx.setLineDash([]);
      ctx.strokeStyle = "#2ca02c";
      ctx.lineWidth = 2;
      ctx.fillStyle = "rgba(44,160,44,0.12)";
      drawEllipsePath(ctx, cx, cy, su, sv, draft.ang);
      ctx.fill();
      drawEllipsePath(ctx, cx, cy, su, sv, draft.ang);
      ctx.stroke();
    }

    canvas._wfcDraw = draw;

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

    function pushToGradio(u1, v1, u2, v2) {
      const wrap = document.getElementById("wfc_drag_signal");
      if (!wrap) return;
      const ta = wfcGetTextEntry(wrap);
      if (!ta) return;
      const payload = JSON.stringify({ u1: u1, v1: v1, u2: u2, v2: v2 });
      ta.value = payload;
      ta.dispatchEvent(new InputEvent("input", { bubbles: true, cancelable: true }));
      ta.dispatchEvent(new Event("change", { bubbles: true }));
    }

    canvas.addEventListener("mousedown", function (ev) {
      if (ev.button !== 0) return;
      const uv = uvFromEvent(ev);
      drag = { u0: uv.u, v0: uv.v, u1: uv.u, v1: uv.v };
      draw();
    });
    canvas.addEventListener("mousemove", function (ev) {
      if (!drag) return;
      const uv = uvFromEvent(ev);
      drag.u1 = uv.u;
      drag.v1 = uv.v;
      draw();
    });
    canvas.addEventListener("mouseup", function (ev) {
      if (!drag) return;
      const uv = uvFromEvent(ev);
      let u1 = Math.min(drag.u0, uv.u),
        u2 = Math.max(drag.u0, uv.u);
      let v1 = Math.min(drag.v0, uv.v),
        v2 = Math.max(drag.v0, uv.v);
      drag = null;
      if (Math.abs(u2 - u1) < 0.01 || Math.abs(v2 - v1) < 0.01) {
        draw();
        return;
      }
      draft = { u1: u1, v1: v1, u2: u2, v2: v2, ang: draft.ang };
      canvas.setAttribute("data-u1", String(u1));
      canvas.setAttribute("data-v1", String(v1));
      canvas.setAttribute("data-u2", String(u2));
      canvas.setAttribute("data-v2", String(v2));
      draw();
      pushToGradio(u1, v1, u2, v2);
    });
    canvas.addEventListener("mouseleave", function () {
      if (drag) {
        drag = null;
        draw();
      }
    });

    draw();
  }

  function scan() {
    const host = document.getElementById("wfc-splat-host");
    const canvas = host
      ? host.querySelector("canvas#wfc-splat-canvas")
      : document.querySelector("canvas#wfc-splat-canvas");
    if (canvas) attachWfcCanvas(canvas);
  }

  function onDomMutation(mutations) {
    for (let i = 0; i < mutations.length; i++) {
      const m = mutations[i];
      if (m.type === "attributes" && m.target && m.target.id === "wfc-splat-canvas" && typeof m.target._wfcDraw === "function") {
        m.target._wfcDraw();
      }
    }
    scan();
  }

  if (!window.__wfcCanvasObserver) {
    window.__wfcCanvasObserver = new MutationObserver(onDomMutation);
    window.__wfcCanvasObserver.observe(document.body, {
      childList: true,
      subtree: true,
      attributes: true,
      attributeFilter: ["data-u1", "data-v1", "data-u2", "data-v2", "data-ang", "data-splats-b64"],
    });
  }
  scan();

  return [];
}
