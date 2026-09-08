# Copyright (c) 2026 Tang Hubocheng. All rights reserved.
# Original project code: no use or redistribution without written permission.
# See the repository LICENSE for scope, exceptions and third-party rights.
"""Self-contained HTML+JS canvas: drag axis-aligned box in [0,1]^2, emit corners to Gradio via hidden Textbox."""

from __future__ import annotations

import base64
import json

from wfc_toolbox.viz import SPLAT_PREVIEW_PIXEL_W


def build_splat_canvas_html(
    n_rows: int,
    n_cols: int,
    splats_json: str,
    draft_u1: float,
    draft_v1: float,
    draft_u2: float,
    draft_v2: float,
    draft_angle_deg: float,
    pixel_w: int | None = None,
) -> str:
    if pixel_w is None:
        pixel_w = SPLAT_PREVIEW_PIXEL_W
    r = max(4, min(40, int(n_rows)))
    c = max(4, min(40, int(n_cols)))
    pixel_h = max(220, int(pixel_w * r / max(1, c)))
    try:
        sj = (splats_json or "").strip() or "[]"
        json.loads(sj)
        enc = sj.encode("utf-8")
    except (json.JSONDecodeError, UnicodeEncodeError):
        enc = b"[]"
    b64 = base64.standard_b64encode(enc).decode("ascii")
    du1, dv1, du2, dv2 = float(draft_u1), float(draft_v1), float(draft_u2), float(draft_v2)
    dang = float(draft_angle_deg)

    return f"""<div id="wfc-splat-host" class="wfc-splat-wrap" style="max-width:100%;display:flex;flex-direction:column;align-items:center;">
<canvas id="wfc-splat-canvas" width="{pixel_w}" height="{pixel_h}"
  style="width:{pixel_w}px;height:{pixel_h}px;max-width:100%;cursor:crosshair;border:1px solid #444;border-radius:4px;touch-action:none;"
  data-splats-b64="{b64}"
  data-u1="{du1:.6f}" data-v1="{dv1:.6f}" data-u2="{du2:.6f}" data-v2="{dv2:.6f}" data-ang="{dang:.4f}"
></canvas>
<p style="font-size:13px;color:#555;margin:6px 0 0 0;text-align:center;">在画布上按下拖动：轴对齐外接框 → 与旋转角组合成椭球；松开后同步到服务器（用于「添加椭球」）。</p>
</div>"""
