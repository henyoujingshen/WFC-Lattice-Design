# Copyright (c) 2026 Tang Hubocheng. All rights reserved.
# Original project code: no use or redistribution without written permission.
# See the repository LICENSE for scope, exceptions and third-party rights.
"""
Gradio web app: Gaussian ellipse painting → field maps → WFC collapse → STL lattice export.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path


def _resource_path(relative_path: str) -> Path:
    """Resolve resource path for both dev and PyInstaller frozen mode."""
    if getattr(sys, "frozen", False):
        return Path(sys._MEIPASS) / relative_path
    return Path(__file__).resolve().parent / relative_path

import gradio as gr

import gradio_client.utils as _gcu

_json_schema_to_python_type_orig = _gcu._json_schema_to_python_type


def _json_schema_to_python_type_safe(schema, defs):
    if isinstance(schema, bool) or not isinstance(schema, dict):
        return "Any"
    return _json_schema_to_python_type_orig(schema, defs)


_gcu._json_schema_to_python_type = _json_schema_to_python_type_safe

from wfc_toolbox.ct_core import (
    calculate_ratio,
    compute_pattern_gate_map,
    generate_field_maps,
)
from wfc_toolbox.pattern_stencil import (
    compile_stencil,
    empty_grid,
    parse_stencil_grid_json,
    preset_bridging_horiz,
    preset_bridging_vert,
    preset_deflection_snake,
    stencil_from_grid,
)
from wfc_toolbox.stencil_grid_html import build_stencil_grid_html
from wfc_toolbox.pattern_utils import apply_splat_hyperparams
from wfc_toolbox.unit_cells import (
    UNIT_CELL_PRESETS,
    default_cell_b_states,
    preset_id_from_label,
    tile_choices_for_pattern,
)
from wfc_toolbox.viz import (
    collapsed_grid_lattice_image,
    fields_and_result_to_image,
    lattice_pipeline_diagram_image,
    splats_canvas_to_image,
    unit_cell_preview_image,
)
from wfc_toolbox.splat_canvas_html import build_splat_canvas_html
from wfc_toolbox.stl_export import export_lattice_stl
from wfc_toolbox.wfc_guided import run_wfc_guided

_CELL_LABEL_CHOICES = [UNIT_CELL_PRESETS[k].label for k in ("octet_ct", "isotropic_octet", "chiral")]
_CELL_B_LABEL_CHOICES = _CELL_LABEL_CHOICES + ["Custom JSON"]
_DEFAULT_CELL_B_JSON = (
    '[[[0,0],[0.5,0.5]],[[1,0],[0.5,0.5]],[[1,1],[0.5,0.5]],[[0,1],[0.5,0.5]],'
    '[[0,0],[1,0]],[[1,0],[1,1]],[[1,1],[0,1]],[[0,1],[0,0]]]'
)
_DEFAULT_STENCIL_JSON = json.dumps({"grid": empty_grid()}, ensure_ascii=False)


def _stencil_json_from_signal(signal: str, fallback: str) -> str:
    if signal and str(signal).strip():
        return str(signal).strip()
    return fallback or _DEFAULT_STENCIL_JSON


def _resolve_protos_rules(stencil_grid_json: str):
    st = stencil_from_grid(parse_stencil_grid_json(stencil_grid_json))
    protos, rules, heads, summary, motif = compile_stencil(st)
    return protos, rules, heads, summary, motif


CANVAS_BIND_JS = _resource_path("wfc_toolbox/splat_canvas_bind.js").read_text(encoding="utf-8")
STENCIL_BIND_JS = _resource_path("wfc_toolbox/stencil_grid_bind.js").read_text(encoding="utf-8")


def _splats_from_state(splats_json: str) -> list[dict]:
    if not splats_json or not str(splats_json).strip():
        return []
    try:
        data = json.loads(splats_json)
        return data if isinstance(data, list) else []
    except json.JSONDecodeError:
        return []


DEFAULT_DRAFT_RECT_JSON = '{"u1":0.12,"v1":0.18,"u2":0.38,"v2":0.48}'


def _parse_draft_rect(draft_json: str) -> tuple[float, float, float, float]:
    try:
        d = json.loads(draft_json or "{}")
        u1, v1, u2, v2 = float(d["u1"]), float(d["v1"]), float(d["u2"]), float(d["v2"])
        return u1, v1, u2, v2
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        return 0.12, 0.18, 0.38, 0.48


def _splats_to_state(splats: list[dict]) -> str:
    return json.dumps(splats, ensure_ascii=False)


def refresh_preview(
    u1,
    v1,
    u2,
    v2,
    angle_deg,
    splats_json: str,
    n_rows: int,
    n_cols: int,
    opacity_regulator: float,
    scale_multiplier: float,
    gate_tau: float,
    chaos_scale: float,
    chaos_power: float,
):
    sj = splats_json if splats_json else "[]"
    html = build_splat_canvas_html(
        int(n_rows),
        int(n_cols),
        sj,
        float(u1),
        float(v1),
        float(u2),
        float(v2),
        float(angle_deg),
    )
    ar, sm = float(opacity_regulator), float(scale_multiplier)
    ri, ci = int(n_rows), int(n_cols)
    gate_map = None
    splats = _splats_from_state(sj)
    if splats:
        splats_eff = apply_splat_hyperparams(splats, sm, ar)
        _, _, chaos, weak, overlap = generate_field_maps(
            ri, ci, splats_eff, chaos_scale=chaos_scale, chaos_power=chaos_power
        )
        gate_map = compute_pattern_gate_map(
            chaos, weak_map=weak, overlap_map=overlap, tau=float(gate_tau)
        )
    mat = splats_canvas_to_image(
        splats,
        title=f"Splat field {ri}x{ci} | alpha={ar:.2f} S_mul={sm:.2f} gate_tau={gate_tau:.2f}",
        aspect_cols=ci,
        aspect_rows=ri,
        opacity_regulator=ar,
        scale_multiplier=sm,
        gate_map=gate_map,
    )
    return html, mat


def refresh_from_draft(
    draft_json: str,
    angle_deg,
    splats_json: str,
    n_rows: int,
    n_cols: int,
    opacity_regulator: float,
    scale_multiplier: float,
    gate_tau: float,
    chaos_scale: float,
    chaos_power: float,
):
    u1, v1, u2, v2 = _parse_draft_rect(draft_json)
    return refresh_preview(
        u1,
        v1,
        u2,
        v2,
        angle_deg,
        splats_json,
        n_rows,
        n_cols,
        opacity_regulator,
        scale_multiplier,
        gate_tau,
        chaos_scale,
        chaos_power,
    )


def add_splat(
    splats_json: str,
    draft_json: str,
    channel: str,
    angle_deg: float,
    opacity: float,
    su_floor: float,
    n_rows: int,
    n_cols: int,
    opacity_regulator: float,
    scale_multiplier: float,
    gate_tau: float,
    chaos_scale: float,
    chaos_power: float,
):
    splats = [dict(x) for x in _splats_from_state(splats_json)]
    u1, v1, u2, v2 = _parse_draft_rect(draft_json)
    u1, u2 = min(u1, u2), max(u1, u2)
    v1, v2 = min(v1, v2), max(v1, v2)
    cx = (u1 + u2) / 2.0
    cy = (v1 + v2) / 2.0
    su = max(abs(u2 - u1) / 2.0, su_floor)
    sv = max(abs(v2 - v1) / 2.0, su_floor)
    splats.append(
        {
            "color": "L" if channel.startswith("L") else "Hd",
            "mu": [cx, cy],
            "scale": [su, sv],
            "angle": float(angle_deg),
            "opacity": float(opacity),
        }
    )
    sj = _splats_to_state(splats)
    html, mat = refresh_preview(
        u1,
        v1,
        u2,
        v2,
        angle_deg,
        sj,
        n_rows,
        n_cols,
        opacity_regulator,
        scale_multiplier,
        gate_tau,
        chaos_scale,
        chaos_power,
    )
    return html, sj, mat, f"Splat #{len(splats)} added ({channel})."


def clear_splats(
    draft_json: str,
    angle_deg: float,
    n_rows: int,
    n_cols: int,
    opacity_regulator: float,
    scale_multiplier: float,
    gate_tau: float,
    chaos_scale: float,
    chaos_power: float,
):
    u1, v1, u2, v2 = _parse_draft_rect(draft_json)
    html, mat = refresh_preview(
        u1,
        v1,
        u2,
        v2,
        angle_deg,
        "[]",
        n_rows,
        n_cols,
        opacity_regulator,
        scale_multiplier,
        gate_tau,
        chaos_scale,
        chaos_power,
    )
    return html, "[]", mat, DEFAULT_DRAFT_RECT_JSON, "All splats cleared."


def default_splats_demo(
    draft_json: str,
    angle_deg: float,
    n_rows: int,
    n_cols: int,
    opacity_regulator: float,
    scale_multiplier: float,
    gate_tau: float,
    chaos_scale: float,
    chaos_power: float,
):
    demo = [
        {"color": "L", "mu": [0.22, 0.35], "scale": [0.22, 0.42], "angle": 0.0, "opacity": 80.0},
        {"color": "L", "mu": [0.22, 0.72], "scale": [0.22, 0.42], "angle": 0.0, "opacity": 80.0},
        {"color": "Hd", "mu": [0.82, 0.35], "scale": [0.16, 0.42], "angle": 0.0, "opacity": 80.0},
        {"color": "Hd", "mu": [0.82, 0.72], "scale": [0.16, 0.42], "angle": 0.0, "opacity": 80.0},
    ]
    sj = _splats_to_state(demo)
    u1, v1, u2, v2 = _parse_draft_rect(draft_json)
    html, mat = refresh_preview(
        u1,
        v1,
        u2,
        v2,
        angle_deg,
        sj,
        n_rows,
        n_cols,
        opacity_regulator,
        scale_multiplier,
        gate_tau,
        chaos_scale,
        chaos_power,
    )
    return html, sj, mat, "Demo splats loaded."


def handle_drag_rect(
    sig,
    splats_json,
    n_rows,
    n_cols,
    angle_deg,
    opacity_regulator,
    scale_multiplier,
    gate_tau,
    chaos_scale,
    chaos_power,
):
    if not sig or not str(sig).strip().startswith("{"):
        return gr.skip()
    try:
        d = json.loads(sig)
        nu1, nv1, nu2, nv2 = float(d["u1"]), float(d["v1"]), float(d["u2"]), float(d["v2"])
        nu1, nu2 = min(nu1, nu2), max(nu1, nu2)
        nv1, nv2 = min(nv1, nv2), max(nv1, nv2)
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        return gr.skip()
    sj = splats_json if splats_json else "[]"
    new_draft = json.dumps({"u1": nu1, "v1": nv1, "u2": nu2, "v2": nv2}, ensure_ascii=False)
    html, mat = refresh_preview(
        nu1,
        nv1,
        nu2,
        nv2,
        angle_deg,
        sj,
        n_rows,
        n_cols,
        opacity_regulator,
        scale_multiplier,
        gate_tau,
        chaos_scale,
        chaos_power,
    )
    return new_draft, html, mat, "", "Bounding box updated from canvas drag."


def on_compile_preview(stencil_grid_json: str):
    _, _, _, summary, _ = _resolve_protos_rules(stencil_grid_json)
    return summary


def _stencil_bundle(stencil_grid_json: str):
    j = stencil_grid_json or _DEFAULT_STENCIL_JSON
    return build_stencil_grid_html(j), j, on_compile_preview(j)


def handle_stencil_signal(signal: str, stencil_state: str):
    j = _stencil_json_from_signal(signal, stencil_state)
    html, j, summary = _stencil_bundle(j)
    choices = tile_choices_for_pattern(j)
    return (
        html,
        j,
        summary,
        gr.CheckboxGroup(choices=choices, value=default_cell_b_states(j)),
        summarize_unit_cell_mapping(
            j,
            _CELL_LABEL_CHOICES[0],
            _CELL_LABEL_CHOICES[1],
            default_cell_b_states(j),
            _DEFAULT_CELL_B_JSON,
            5.0,
            6.0,
            0.08,
            0.14,
        ),
    )


def load_stencil_preset(preset_name: str):
    if preset_name == "Bridging-Vert":
        st = preset_bridging_vert()
    elif preset_name == "Bridging-Horiz":
        st = preset_bridging_horiz()
    elif preset_name == "Deflection-Snake":
        st = preset_deflection_snake()
    else:
        st = stencil_from_grid(empty_grid())
    j = json.dumps({"grid": st.normalized_grid()}, ensure_ascii=False)
    return _stencil_bundle(j)




def _cell_label_to_id(label: str) -> str:
    if label and "Custom" in str(label):
        return "custom"
    return preset_id_from_label(str(label))


def preview_unit_cells_ui(
    cell_a_label: str,
    cell_b_label: str,
    cell_b_json: str,
    radius_soft_frac: float,
    radius_hard_frac: float,
):
    return unit_cell_preview_image(
        _cell_label_to_id(cell_a_label),
        _cell_label_to_id(cell_b_label),
        cell_b_json or "",
        radius_frac_soft=float(radius_soft_frac),
        radius_frac_hard=float(radius_hard_frac),
    )


def toggle_cell_b_custom_panel(cell_b_label: str):
    return gr.update(visible="Custom" in str(cell_b_label))


def summarize_unit_cell_mapping(
    stencil_grid_json: str,
    cell_a_label: str,
    cell_b_label: str,
    cell_b_states: list,
    cell_b_json: str,
    cell_size_mm: float,
    extrude_thickness_mm: float,
    radius_soft_frac: float,
    radius_hard_frac: float,
):
    a_id = _cell_label_to_id(cell_a_label)
    b_id = _cell_label_to_id(cell_b_label)
    b_set = set(cell_b_states or [])
    all_tiles = tile_choices_for_pattern(stencil_grid_json)
    n_coarse = sum(sum(row) for row in parse_stencil_grid_json(stencil_grid_json))
    lines = [
        f"Stencil: {n_coarse} coarse cells (rest L/Hd fine)",
        f"Cell A: {cell_a_label} → {', '.join(t for t in all_tiles if t not in b_set) or '(none)'}",
        f"Cell B: {cell_b_label} → {', '.join(sorted(b_set)) or '(none)'}",
        f"Size: cell={cell_size_mm:.2f} mm, extrude={extrude_thickness_mm:.2f} mm",
        f"Rod radius (×cell): soft={radius_soft_frac:.3f}, hard={radius_hard_frac:.3f}",
    ]
    if b_id == "custom":
        try:
            n = len(json.loads(cell_b_json or "[]"))
            lines.append(f"Custom B segments: {n}")
        except json.JSONDecodeError:
            lines.append("Custom B JSON invalid — preview may be empty.")
    return "\n".join(lines)


def run_pipeline(
    splats_json: str,
    n_rows: int,
    n_cols: int,
    stencil_grid_json: str,
    opacity_regulator: float,
    scale_multiplier: float,
    gate_tau: float,
    target_hard: float,
    chaos_scale: float,
    chaos_power: float,
    max_attempts: int,
    seed: int,
    cell_a_label: str,
    cell_b_label: str,
    cell_b_states: list,
    cell_b_custom_json: str,
    cell_size_mm: float,
    extrude_thickness_mm: float,
    radius_soft_frac: float,
    radius_hard_frac: float,
):
    splats = _splats_from_state(splats_json)
    if not splats:
        return None, None, None, "Error: Add at least one splat (L or Hd) first."
    ri, ci = int(n_rows), int(n_cols)
    if ri < 4 or ci < 4:
        return None, None, None, "Error: Grid must be at least 4×4."

    splats_eff = apply_splat_hyperparams(splats, scale_multiplier, opacity_regulator)
    protos, rules, head_tiles, compile_msg, motif = _resolve_protos_rules(stencil_grid_json)
    L_map, Hd_map, Chaos_map, weak_map, overlap_map = generate_field_maps(
        ri, ci, splats_eff, chaos_scale=chaos_scale, chaos_power=chaos_power
    )
    gate_map = compute_pattern_gate_map(
        Chaos_map, weak_map=weak_map, overlap_map=overlap_map, tau=float(gate_tau)
    )

    grid = run_wfc_guided(
        (ri, ci),
        protos,
        rules,
        L_map,
        Hd_map,
        Chaos_map,
        target_hard_ratio=target_hard,
        max_attempts=max_attempts,
        seed=seed if seed >= 0 else None,
        gate_map=gate_map,
        head_tiles=head_tiles,
        motif=motif,
        gate_tau=float(gate_tau),
    )

    fields_img = fields_and_result_to_image(
        L_map,
        Hd_map,
        Chaos_map,
        grid,
        protos,
        title=f"Fields + WFC | alpha={opacity_regulator:.2f} S_mul={scale_multiplier:.2f}",
    )
    grid_img = collapsed_grid_lattice_image(
        grid,
        protos,
        cell_a_id=_cell_label_to_id(cell_a_label),
        cell_b_id=_cell_label_to_id(cell_b_label),
        cell_b_states=set(cell_b_states or []),
        cell_b_custom_json=cell_b_custom_json or "",
        title=f"Lattice {ri}×{ci} — 2D strut layout (A/B topology per cell)",
    )
    s, h = calculate_ratio(grid, protos)
    stl_path = None
    stl_note = ""
    try:
        stl_path = export_lattice_stl(
            grid,
            protos,
            cell_a_id=_cell_label_to_id(cell_a_label),
            cell_b_id=_cell_label_to_id(cell_b_label),
            cell_b_states=set(cell_b_states or []),
            cell_b_custom_json=cell_b_custom_json or "",
            cell_size_mm=float(cell_size_mm),
            extrude_thickness_mm=float(extrude_thickness_mm),
            radius_soft_frac=float(radius_soft_frac),
            radius_hard_frac=float(radius_hard_frac),
        )
        stl_note = f"\nSTL: {stl_path}"
    except Exception as e:
        stl_note = f"\nSTL export error: {e}"

    extra = f"\n{compile_msg[:240]}" if compile_msg else ""
    return (
        fields_img,
        grid_img,
        stl_path,
        f"Done. Soft={s:.1%} Hard={h:.1%} (target={target_hard:.0%}) gate_tau={gate_tau:.2f}{stl_note}\n{extra}",
    )


def build_demo():
    css = """
    /* Typography */
    .gradio-container { font-family: 'Inter', system-ui, -apple-system, sans-serif; }
    h1, h2, h3 { text-wrap: balance; letter-spacing: -0.02em; }
    h1 { font-size: 1.5rem; font-weight: 600; }
    h3, .step-heading { font-size: 0.95rem; font-weight: 600; color: var(--neutral-800); }
    /* Layout */
    .main-wrap { max-width: 1100px; margin: 0 auto; }
    .center-plot { display: flex !important; justify-content: center !important; flex-direction: column; align-items: center; }
    #wfc-splat-host canvas, #wfc_matplotlib_preview img {
        display: block; margin: 0 auto; max-width: 100%; height: auto;
    }
    /* Cards / sections */
    .step-card {
        background: var(--neutral-50, #f9fafb);
        border: 1px solid var(--neutral-200, #e5e7eb);
        border-radius: 10px;
        padding: 1rem 1.2rem;
        margin-bottom: 0.6rem;
    }
    /* Focus */
    button:focus-visible, input:focus-visible, select:focus-visible {
        outline: 2px solid var(--color-accent, #2563eb); outline-offset: 2px;
    }
    /* Hover */
    button:hover { filter: brightness(0.95); transition: filter 0.15s; }
    /* Scroll safety */
    .gradio-container { overscroll-behavior: contain; }
    /* Status message colors */
    .status-error { color: #dc2626; font-weight: 500; }
    .status-success { color: #16a34a; }
    .status-info { color: var(--neutral-600, #6b7280); }
    /* Dark mode basics */
    @media (prefers-color-scheme: dark) {
        .step-card { background: rgba(255,255,255,0.04); border-color: rgba(255,255,255,0.08); }
    }
    /* Number inputs */
    input[type=number] { max-width: 130px; }
    """
    with gr.Blocks(title="WFC Gaussian Lattice Designer", css=css) as demo:
        demo.load(None, None, None, js=CANVAS_BIND_JS)
        demo.load(None, None, None, js=STENCIL_BIND_JS)

        gr.Markdown(
            "# Non-Periodic Lattice Design\n"
            "Paint Gaussian ellipses → WFC collapse → STL export. "
            "Drag on canvas to define splats, toggle the 5×5 stencil, then run WFC."
        )

        splats_state = gr.State("[]")
        draft_rect_state = gr.State(DEFAULT_DRAFT_RECT_JSON)
        init_html = build_splat_canvas_html(15, 18, "[]", *_parse_draft_rect(DEFAULT_DRAFT_RECT_JSON), 0.0)

        with gr.Column(elem_classes=["main-wrap"]):
            with gr.Row():
                n_rows = gr.Number(value=15, precision=0, label="Rows", minimum=4, maximum=40)
                n_cols = gr.Number(value=18, precision=0, label="Cols", minimum=4, maximum=40)
                btn_apply_grid = gr.Button("Apply Grid", variant="secondary")

            gr.Markdown("### Splats")

            drag_signal = gr.Textbox(
                elem_id="wfc_drag_signal",
                value="",
                label="",
                show_label=False,
                visible=False,
                max_lines=1,
            )
            with gr.Column(elem_classes=["center-plot"]):
                design_canvas = gr.HTML(value=init_html)

            channel = gr.Radio(choices=["L (soft)", "Hd (hard)"], value="L (soft)", label="Channel")
            angle_deg = gr.Slider(-90, 90, value=0, step=1, label="Angle (deg)")
            with gr.Row():
                btn_add = gr.Button("Add Splat", variant="primary")
                btn_demo = gr.Button("Load Demo", variant="secondary")
                btn_clear = gr.Button("Clear", variant="secondary")
            canvas_preview = gr.Image(
                label="Field Preview",
                type="numpy",
                elem_id="wfc_matplotlib_preview",
            )

            gr.Markdown("### Field Params & Stencil")
            with gr.Row():
                opacity_regulator = gr.Slider(0.2, 2.5, value=1.0, step=0.05, label="Opacity factor α")
                scale_multiplier = gr.Slider(0.5, 2.5, value=1.0, step=0.05, label="Scale factor S_mul")
            gate_tau = gr.Slider(
                0.15, 0.85, value=0.4, step=0.05,
                label="Gate threshold τ (lower = wider pattern zone)",
            )
            stencil_state = gr.State(_DEFAULT_STENCIL_JSON)
            stencil_signal = gr.Textbox(
                elem_id="wfc_stencil_signal", value="", label="",
                show_label=False, visible=False, max_lines=1,
            )
            stencil_canvas = gr.HTML(value=build_stencil_grid_html(_DEFAULT_STENCIL_JSON))
            with gr.Row():
                btn_stencil_clear = gr.Button("Clear Stencil", variant="secondary")
                btn_preset_br_vert = gr.Button("Bridging Vert", variant="secondary")
                btn_preset_br_horiz = gr.Button("Bridging Horiz", variant="secondary")
                btn_preset_defl = gr.Button("Deflection", variant="secondary")
            compile_preview = gr.Textbox(
                label="Compilation Summary", lines=4, interactive=False,
                value=on_compile_preview(_DEFAULT_STENCIL_JSON),
            )

            gr.Markdown(
                "### Cell Topology\n"
                "- **Cell A** (fine): `L` state — high stiffness, thin rods\n"
                "- **Cell B** (coarse): `Hd`, `PAT*`, other non-soft states — thick rods"
            )
            pipeline_diagram = gr.Image(
                label="Pipeline Diagram",
                value=lattice_pipeline_diagram_image(),
                type="numpy", interactive=False,
            )
            with gr.Row():
                cell_a_preset = gr.Dropdown(
                    choices=_CELL_LABEL_CHOICES,
                    value=_CELL_LABEL_CHOICES[0],
                    label="Cell A (fine)",
                )
                cell_b_preset = gr.Dropdown(
                    choices=_CELL_B_LABEL_CHOICES,
                    value=_CELL_LABEL_CHOICES[1],
                    label="Cell B (coarse)",
                )
            with gr.Row(visible=False) as cell_b_custom_row:
                cell_b_custom_json = gr.Textbox(
                    label="Cell B Custom JSON",
                    value=_DEFAULT_CELL_B_JSON, lines=4,
                    info="Format: [[[u1,v1],[u2,v2]], ...], coords in [0,1]²",
                )
            _init_tile_choices = tile_choices_for_pattern(_DEFAULT_STENCIL_JSON)
            cell_b_tile_states = gr.CheckboxGroup(
                choices=_init_tile_choices,
                value=default_cell_b_states(_DEFAULT_STENCIL_JSON),
                label="Use Cell B for these states",
            )
            with gr.Row():
                cell_size_mm = gr.Slider(1.0, 20.0, value=5.0, step=0.5, label="Cell size (mm)")
                extrude_thickness_mm = gr.Slider(0.5, 30.0, value=6.0, step=0.5, label="Extrude thickness (mm)")
                radius_soft_frac = gr.Slider(0.02, 0.25, value=0.08, step=0.005, label="Soft rod radius (×cell)")
                radius_hard_frac = gr.Slider(0.05, 0.35, value=0.14, step=0.005, label="Hard rod radius (×cell)")
            unit_cell_preview = gr.Image(label="Cell A / B Preview", type="numpy")
            unit_cell_summary = gr.Textbox(label="Cell Mapping", lines=4, interactive=False)

            with gr.Accordion("Advanced", open=False):
                opacity_splat = gr.Slider(5, 150, value=80, step=1, label="Splat write opacity")
                su_floor = gr.Slider(0.01, 0.08, value=0.02, step=0.005, label="Min semi-axis")
                target_hard = gr.Slider(0.2, 0.75, value=0.45, step=0.01, label="Target hard ratio")
                chaos_scale = gr.Slider(1.0, 12.0, value=6.0, step=0.1, label="Chaos scale")
                chaos_power = gr.Slider(0.5, 2.5, value=1.3, step=0.05, label="Chaos power")
                max_attempts = gr.Slider(50, 800, value=350, step=10, label="Max WFC attempts")
                seed = gr.Number(value=0, precision=0, label="Seed (<0 = random)")

            btn_run = gr.Button("Run WFC", variant="primary", size="lg")
            with gr.Row():
                out_fields = gr.Image(label="Field Maps", type="numpy")
                out_grid = gr.Image(label="Lattice Diagram", type="numpy")
            out_stl = gr.File(label="Download STL", interactive=False)
            out_msg = gr.Textbox(label="Log", lines=4)

        def _apply_grid(nr, nc, angle_deg, sj, draft_json, opacity_regulator, scale_multiplier, gate_tau, chaos_scale, chaos_power):
            r, c = int(nr), int(nc)
            r, c = max(4, min(40, r)), max(4, min(40, c))
            html, mat = refresh_from_draft(
                draft_json,
                angle_deg,
                sj,
                r,
                c,
                opacity_regulator,
                scale_multiplier,
                gate_tau,
                chaos_scale,
                chaos_power,
            )
            return html, mat, f"Grid set to {r}×{c}.", r, c

        btn_apply_grid.click(
            _apply_grid,
            inputs=[
                n_rows,
                n_cols,
                angle_deg,
                splats_state,
                draft_rect_state,
                opacity_regulator,
                scale_multiplier,
                gate_tau,
                chaos_scale,
                chaos_power,
            ],
            outputs=[design_canvas, canvas_preview, out_msg, n_rows, n_cols],
        )

        corner_inputs = [
            draft_rect_state,
            angle_deg,
            splats_state,
            n_rows,
            n_cols,
            opacity_regulator,
            scale_multiplier,
            gate_tau,
            chaos_scale,
            chaos_power,
        ]
        angle_deg.change(refresh_from_draft, inputs=corner_inputs, outputs=[design_canvas, canvas_preview])

        n_rows.change(refresh_from_draft, inputs=corner_inputs, outputs=[design_canvas, canvas_preview])
        n_cols.change(refresh_from_draft, inputs=corner_inputs, outputs=[design_canvas, canvas_preview])

        splats_state.change(refresh_from_draft, inputs=corner_inputs, outputs=[design_canvas, canvas_preview])

        opacity_regulator.change(refresh_from_draft, inputs=corner_inputs, outputs=[design_canvas, canvas_preview])
        scale_multiplier.change(refresh_from_draft, inputs=corner_inputs, outputs=[design_canvas, canvas_preview])

        drag_signal.input(
            handle_drag_rect,
            inputs=[
                drag_signal,
                splats_state,
                n_rows,
                n_cols,
                angle_deg,
                opacity_regulator,
                scale_multiplier,
                gate_tau,
                chaos_scale,
                chaos_power,
            ],
            outputs=[draft_rect_state, design_canvas, canvas_preview, drag_signal, out_msg],
        )

        btn_add.click(
            add_splat,
            inputs=[
                splats_state,
                draft_rect_state,
                channel,
                angle_deg,
                opacity_splat,
                su_floor,
                n_rows,
                n_cols,
                opacity_regulator,
                scale_multiplier,
                gate_tau,
                chaos_scale,
                chaos_power,
            ],
            outputs=[design_canvas, splats_state, canvas_preview, out_msg],
        )
        btn_clear.click(
            clear_splats,
            inputs=[
                draft_rect_state,
                angle_deg,
                n_rows,
                n_cols,
                opacity_regulator,
                scale_multiplier,
                gate_tau,
                chaos_scale,
                chaos_power,
            ],
            outputs=[design_canvas, splats_state, canvas_preview, draft_rect_state, out_msg],
        )
        btn_demo.click(
            default_splats_demo,
            inputs=[
                draft_rect_state,
                angle_deg,
                n_rows,
                n_cols,
                opacity_regulator,
                scale_multiplier,
                gate_tau,
                chaos_scale,
                chaos_power,
            ],
            outputs=[design_canvas, splats_state, canvas_preview, out_msg],
        )

        stencil_outputs = [
            stencil_canvas,
            stencil_state,
            compile_preview,
            cell_b_tile_states,
            unit_cell_summary,
        ]

        stencil_signal.input(
            handle_stencil_signal,
            inputs=[stencil_signal, stencil_state],
            outputs=stencil_outputs,
        )

        def _apply_preset(name: str):
            html, j, summary = load_stencil_preset(name)
            choices = tile_choices_for_pattern(j)
            return (
                html,
                j,
                summary,
                gr.CheckboxGroup(choices=choices, value=default_cell_b_states(j)),
                summarize_unit_cell_mapping(
                    j,
                    _CELL_LABEL_CHOICES[0],
                    _CELL_LABEL_CHOICES[1],
                    default_cell_b_states(j),
                    _DEFAULT_CELL_B_JSON,
                    5.0,
                    6.0,
                    0.08,
                    0.14,
                ),
            )

        btn_stencil_clear.click(lambda: _apply_preset("clear"), outputs=stencil_outputs)
        btn_preset_br_vert.click(lambda: _apply_preset("Bridging-Vert"), outputs=stencil_outputs)
        btn_preset_br_horiz.click(lambda: _apply_preset("Bridging-Horiz"), outputs=stencil_outputs)
        btn_preset_defl.click(lambda: _apply_preset("Deflection-Snake"), outputs=stencil_outputs)

        unit_cell_preview_inputs = [
            cell_a_preset,
            cell_b_preset,
            cell_b_custom_json,
            radius_soft_frac,
            radius_hard_frac,
        ]
        unit_cell_summary_inputs = [
            stencil_state,
            cell_a_preset,
            cell_b_preset,
            cell_b_tile_states,
            cell_b_custom_json,
            cell_size_mm,
            extrude_thickness_mm,
            radius_soft_frac,
            radius_hard_frac,
        ]

        stencil_state.change(
            lambda j: on_compile_preview(j),
            inputs=[stencil_state],
            outputs=[compile_preview],
        )
        stencil_state.change(
            summarize_unit_cell_mapping,
            inputs=unit_cell_summary_inputs,
            outputs=[unit_cell_summary],
        )
        cell_b_preset.change(toggle_cell_b_custom_panel, inputs=[cell_b_preset], outputs=[cell_b_custom_row])

        for evt in (
            cell_a_preset.change,
            cell_b_preset.change,
            cell_b_custom_json.change,
            radius_soft_frac.change,
            radius_hard_frac.change,
            cell_b_tile_states.change,
            cell_size_mm.change,
            extrude_thickness_mm.change,
        ):
            evt(preview_unit_cells_ui, inputs=unit_cell_preview_inputs, outputs=[unit_cell_preview])
            evt(summarize_unit_cell_mapping, inputs=unit_cell_summary_inputs, outputs=[unit_cell_summary])

        demo.load(refresh_from_draft, inputs=corner_inputs, outputs=[design_canvas, canvas_preview])
        demo.load(preview_unit_cells_ui, inputs=unit_cell_preview_inputs, outputs=[unit_cell_preview])
        demo.load(summarize_unit_cell_mapping, inputs=unit_cell_summary_inputs, outputs=[unit_cell_summary])

        btn_run.click(
            run_pipeline,
            inputs=[
                splats_state,
                n_rows,
                n_cols,
                stencil_state,
                opacity_regulator,
                scale_multiplier,
                gate_tau,
                target_hard,
                chaos_scale,
                chaos_power,
                max_attempts,
                seed,
                cell_a_preset,
                cell_b_preset,
                cell_b_tile_states,
                cell_b_custom_json,
                cell_size_mm,
                extrude_thickness_mm,
                radius_soft_frac,
                radius_hard_frac,
            ],
            outputs=[out_fields, out_grid, out_stl, out_msg],
        )

    return demo


def _pick_server_port(preferred: int = 7860, *, max_tries: int = 20) -> int:
    """Use preferred port if free, else try preferred+1 … (7860 often left by a prior Gradio)."""
    import socket

    for p in range(preferred, preferred + max_tries):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.bind(("127.0.0.1", p))
                return p
            except OSError:
                continue
    raise OSError(
        f"Ports {preferred}-{preferred + max_tries - 1} are all in use. "
        f"Close the other python app.py or set GRADIO_SERVER_PORT."
    )


if __name__ == "__main__":
    preferred = int(os.environ.get("GRADIO_SERVER_PORT", "7860"))
    port = _pick_server_port(preferred)
    url = f"http://127.0.0.1:{port}"
    print(f"\n{'='*50}")
    print(f"  WFC Gaussian Lattice Designer")
    print(f"  Local: {url}")
    print(f"  Press Ctrl+C to exit")
    print(f"{'='*50}\n")

    import webbrowser
    webbrowser.open(url)

    build_demo().launch(
        inbrowser=True,
        server_name="127.0.0.1",
        server_port=port,
        show_api=False,
        _frontend=False,
        prevent_thread_lock=False,
    )
    input("\nPress Enter to exit...")
