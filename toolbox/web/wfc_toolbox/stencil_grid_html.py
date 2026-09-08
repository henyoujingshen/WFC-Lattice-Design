# Copyright (c) 2026 Tang Hubocheng. All rights reserved.
# Original project code: no use or redistribution without written permission.
# See the repository LICENSE for scope, exceptions and third-party rights.
"""Interactive 5×5 coarse/fine paint board for Gradio."""

from __future__ import annotations

import base64
import json

from wfc_toolbox.pattern_stencil import STENCIL_SIZE, empty_grid, parse_stencil_grid_json

CELL_PX = 52


def build_stencil_grid_html(grid_json: str | None = None) -> str:
    grid = parse_stencil_grid_json(grid_json)
    payload = json.dumps({"grid": grid}, ensure_ascii=False).encode("utf-8")
    b64 = base64.standard_b64encode(payload).decode("ascii")
    n_coarse = sum(sum(row) for row in grid)

    cells_html = []
    for r in range(STENCIL_SIZE):
        for c in range(STENCIL_SIZE):
            on = grid[r][c]
            cls = "wfc-stencil-cell coarse" if on else "wfc-stencil-cell fine"
            cells_html.append(
                f'<button type="button" class="{cls}" data-r="{r}" data-c="{c}" '
                f'title="行{r+1} 列{c+1}：点击切换粗/细" aria-label="r{r}c{c}"></button>'
            )

    grid_inner = "\n".join(
        f'<div class="wfc-stencil-row">{"".join(cells_html[r * STENCIL_SIZE : (r + 1) * STENCIL_SIZE])}</div>'
        for r in range(STENCIL_SIZE)
    )

    return f"""<div id="wfc-stencil-host" class="wfc-stencil-wrap" data-grid-b64="{b64}">
<style>
.wfc-stencil-wrap {{
  font-family: 'Segoe UI', system-ui, sans-serif;
  max-width: 340px;
  margin: 0 auto;
}}
.wfc-stencil-wrap h4 {{
  margin: 0 0 8px 0;
  font-size: 14px;
  color: #334155;
  font-weight: 700;
}}
.wfc-stencil-legend {{
  display: flex;
  gap: 14px;
  font-size: 12px;
  color: #64748b;
  margin-bottom: 10px;
}}
.wfc-stencil-legend span::before {{
  content: '';
  display: inline-block;
  width: 14px;
  height: 14px;
  border-radius: 4px;
  margin-right: 6px;
  vertical-align: middle;
  border: 1px solid #cbd5e1;
}}
.wfc-stencil-legend .lg-coarse::before {{ background: #c62828; border-color: #b71c1c; }}
.wfc-stencil-legend .lg-fine::before {{ background: #eceff1; }}
.wfc-stencil-board {{
  display: inline-block;
  padding: 8px;
  background: #f8fafc;
  border: 2px solid #e2e8f0;
  border-radius: 12px;
}}
.wfc-stencil-row {{ display: flex; gap: 4px; margin-bottom: 4px; }}
.wfc-stencil-row:last-child {{ margin-bottom: 0; }}
.wfc-stencil-cell {{
  width: {CELL_PX}px;
  height: {CELL_PX}px;
  padding: 0;
  border-radius: 8px;
  cursor: pointer;
  border: 2px solid #cfd8dc;
  transition: transform 0.08s, box-shadow 0.12s;
}}
.wfc-stencil-cell:hover {{
  transform: scale(1.04);
  box-shadow: 0 2px 8px rgba(0,0,0,0.12);
}}
.wfc-stencil-cell.fine {{ background: #eceff1; }}
.wfc-stencil-cell.coarse {{ background: #e53935; border-color: #b71c1c; }}
.wfc-stencil-stats {{
  margin-top: 8px;
  font-size: 12px;
  color: #475569;
}}
</style>
<h4>5×5 粗/细单元模板（点击切换）</h4>
<div class="wfc-stencil-legend">
  <span class="lg-fine">细单元（未涂）</span>
  <span class="lg-coarse">粗单元（已涂）</span>
</div>
<div class="wfc-stencil-board">{grid_inner}</div>
<p class="wfc-stencil-stats" id="wfc-stencil-stats">已涂 <b>{n_coarse}</b> / {STENCIL_SIZE * STENCIL_SIZE} 粗单元</p>
<p style="font-size:12px;color:#64748b;margin:6px 0 0 0;">整组粗单元保持画板中的相对位置；大网格上何处出现由上方场预览的 <b>Gate τ</b>（黄层）决定。</p>
</div>"""
