# Copyright (c) 2026 Tang Hubocheng. All rights reserved.
# Original project code: no use or redistribution without written permission.
# See the repository LICENSE for scope, exceptions and third-party rights.
"""
Gradio：长方形网格 → HTML 画布鼠标拖出外接矩形 + 旋转角（类矢量椭圆工具）→
全局 α / S_mul → 5×5 图案 stencil + WFC 坍缩 → 大图展示构型（无 CT 试件几何）。
"""

from __future__ import annotations

import json
import os
from pathlib import Path

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
_CELL_B_LABEL_CHOICES = _CELL_LABEL_CHOICES + ["自定义 JSON 杆段（步骤①）"]
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


_BIND_JS_PATH = Path(__file__).resolve().parent / "wfc_toolbox" / "splat_canvas_bind.js"
_STENCIL_BIND_JS_PATH = Path(__file__).resolve().parent / "wfc_toolbox" / "stencil_grid_bind.js"
CANVAS_BIND_JS = _BIND_JS_PATH.read_text(encoding="utf-8")
STENCIL_BIND_JS = _STENCIL_BIND_JS_PATH.read_text(encoding="utf-8")


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
    return html, sj, mat, f"已添加椭球 #{len(splats)}（{channel}）。"


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
    return html, "[]", mat, DEFAULT_DRAFT_RECT_JSON, "已清空椭球列表；外接框已恢复默认。"


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
    return html, sj, mat, "已加载示例椭球。"


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
    return new_draft, html, mat, "", "已根据鼠标拖拽更新外接框（与画布一致；可再调旋转角后点「添加椭球」）。"


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
    if label and "自定义" in str(label):
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
    return gr.update(visible="自定义" in str(cell_b_label))


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
        f"5×5 模板: {n_coarse} 个粗单元（其余为细单元 L/Hd）",
        f"单元格 A（默认）: {cell_a_label} → 用于 {', '.join(t for t in all_tiles if t not in b_set) or '（无）'}",
        f"单元格 B（第二拓扑）: {cell_b_label} → 用于 {', '.join(sorted(b_set)) or '（未选）'}",
        f"物理尺度: cell_size={cell_size_mm:.2f} mm，挤出厚度={extrude_thickness_mm:.2f} mm",
        f"杆径比例（×cell_size）: 软 L={radius_soft_frac:.3f}，硬/功能={radius_hard_frac:.3f}（同 CT generate_mesh）",
    ]
    if b_id == "custom":
        try:
            n = len(json.loads(cell_b_json or "[]"))
            lines.append(f"自定义 B 杆段数: {n}")
        except json.JSONDecodeError:
            lines.append("自定义 B JSON 格式无效，预览可能为空。")
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
        return None, None, None, "请先添加至少一个椭球（L 或 Hd）。"
    ri, ci = int(n_rows), int(n_cols)
    if ri < 4 or ci < 4:
        return None, None, None, "网格至少 4×4。"

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
        stl_note = f"\nSTL 导出失败: {e}"

    extra = f"\n{compile_msg[:240]}" if compile_msg else ""
    return (
        fields_img,
        grid_img,
        stl_path,
        f"完成。软={s:.1%} 硬={h:.1%}（目标硬≈{target_hard:.0%}）gate_tau={gate_tau:.2f}{stl_note}\n{extra}",
    )


def build_demo():
    css = """
    .main-wrap { max-width: 1100px; margin-left: auto; margin-right: auto; }
    .center-plot { display: flex !important; justify-content: center !important; flex-direction: column; align-items: center; }
    #wfc-splat-host canvas, #wfc_matplotlib_preview img {
        display: block;
        margin-left: auto;
        margin-right: auto;
        max-width: 100%;
        height: auto;
    }
    """
    with gr.Blocks(title="高斯场引导 WFC", css=css) as demo:
        demo.load(None, None, None, js=CANVAS_BIND_JS)
        demo.load(None, None, None, js=STENCIL_BIND_JS)

        gr.Markdown(
            "# 非周期点阵设计（长方形网格）\n"
            "1. 设定 **行×列**，点「应用网格」后中央 **HTML 画布** 纵横比与网格一致。\n"
            "2. **鼠标拖拽**：在画布上拖出轴对齐外接框（**唯一**几何输入，与下方无滑块冲突）；再选 **L/Hd**、**旋转角**，点「添加椭球」。\n"
            "3. **全局 α、S_mul**：与 ``generate_field_maps`` 前 ``apply_splat_hyperparams`` 一致，**下方 Matplotlib 预览**会实时显示软/硬高斯叠加强度与范围；运行 WFC 时同样作用于场图。「高级」里另有**写入椭球时的强度**（splats 内 `opacity`）与**最小半轴**，属几何/写入细节，默认即可。\n"
            "4. **5×5 画板**涂粗单元（点击切换）；整组保持相对位置，在大网格 **Gate τ** 黄区放置。\n"
            "5. 在 **④ 单元格结构** 中配置杆件拓扑与 WFC 态映射。\n"
            "6. 「运行 WFC」：场图 + 晶格杆系图 + **可下载 STL**（2.5D 挤出，单位 mm）。"
        )

        splats_state = gr.State("[]")
        draft_rect_state = gr.State(DEFAULT_DRAFT_RECT_JSON)
        init_html = build_splat_canvas_html(15, 18, "[]", *_parse_draft_rect(DEFAULT_DRAFT_RECT_JSON), 0.0)

        with gr.Column(elem_classes=["main-wrap"]):
            with gr.Row():
                n_rows = gr.Number(value=15, precision=0, label="行数 M", minimum=4, maximum=40)
                n_cols = gr.Number(value=18, precision=0, label="列数 N", minimum=4, maximum=40)
                btn_apply_grid = gr.Button("① 应用网格", variant="secondary")

            gr.Markdown("### ② 在画布上布置高斯椭球（归一化坐标 [0,1]²，v 向下）")
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

            gr.Markdown(
                "在画布上拖拽定义外接矩形；**不再**使用四角坐标滑块，避免与鼠标不同步。"
            )
            channel = gr.Radio(choices=["L 软场", "Hd 硬场"], value="L 软场", label="椭球通道")
            angle_deg = gr.Slider(-90, 90, value=0, step=1, label="旋转角（度）")
            with gr.Row():
                btn_add = gr.Button("添加椭球", variant="primary")
                btn_demo = gr.Button("示例椭球")
                btn_clear = gr.Button("清空")
            canvas_preview = gr.Image(
                label="椭球场 Matplotlib 预览（高斯叠加热图，α / S_mul 与场图一致）",
                type="numpy",
                elem_id="wfc_matplotlib_preview",
            )

            gr.Markdown("### ③ 全局场超参（α、S_mul）与 5×5 粗/细模板")
            with gr.Row():
                opacity_regulator = gr.Slider(0.2, 2.5, value=1.0, step=0.05, label="不透明度调节因子 α（全局，作用于场图）")
                scale_multiplier = gr.Slider(0.5, 2.5, value=1.0, step=0.05, label="尺度缩放因子 S_mul（全局，作用于场图）")
            gate_tau = gr.Slider(
                0.15,
                0.85,
                value=0.4,
                step=0.05,
                label="图案允许区门槛 τ（看上方预览黄层；越小黄区越宽）",
            )
            stencil_state = gr.State(_DEFAULT_STENCIL_JSON)
            stencil_signal = gr.Textbox(
                elem_id="wfc_stencil_signal",
                value="",
                label="",
                show_label=False,
                visible=False,
                max_lines=1,
            )
            stencil_canvas = gr.HTML(value=build_stencil_grid_html(_DEFAULT_STENCIL_JSON))
            with gr.Row():
                btn_stencil_clear = gr.Button("清空画板")
                btn_preset_br_vert = gr.Button("预设 竖桥")
                btn_preset_br_horiz = gr.Button("预设 横桥")
                btn_preset_defl = gr.Button("预设 蛇形")
            compile_preview = gr.Textbox(
                label="模板编译摘要（相对位置 + link）",
                lines=4,
                interactive=False,
                value=on_compile_preview(_DEFAULT_STENCIL_JSON),
            )

            gr.Markdown(
                "### ④ 单元格结构：高斯场 → 点阵（Python 杆段拓扑 + 杆径）\n"
                "WFC 每个格子得到一个 **离散态**（如 `L`、`Hd`、`H_1`…），其 `stiffness_level` 决定 **杆径**；"
                "同一态在所有格子上复用 **归一化 [0,1]² 内的杆段列表**（见 `CT_GaussianField_Vertical_v2.get_octet_unit` 与 `jason_wfc_stl`）。\n"
                "- **步骤①**：为 **单元格 B** 选择预设或粘贴自定义 JSON 杆段。\n"
                "- **步骤②**：**L → 单元格 A（细）**；**Hd / PAT* / 其它非 L → 单元格 B（粗）**（按 stiffness，与勾选一致）。"
            )
            pipeline_diagram = gr.Image(
                label="流程示意",
                value=lattice_pipeline_diagram_image(),
                type="numpy",
                interactive=False,
            )
            with gr.Row():
                cell_a_preset = gr.Dropdown(
                    choices=_CELL_LABEL_CHOICES,
                    value=_CELL_LABEL_CHOICES[0],
                    label="单元格 A（默认拓扑，CT Octet）",
                )
                cell_b_preset = gr.Dropdown(
                    choices=_CELL_B_LABEL_CHOICES,
                    value=_CELL_LABEL_CHOICES[1],
                    label="单元格 B（第二拓扑，步骤①）",
                )
            with gr.Row(visible=False) as cell_b_custom_row:
                cell_b_custom_json = gr.Textbox(
                    label="单元格 B 自定义杆段 JSON",
                    value=_DEFAULT_CELL_B_JSON,
                    lines=4,
                    info="格式: [[[u1,v1],[u2,v2]], ...]，坐标 ∈ [0,1]²",
                )
            _init_tile_choices = tile_choices_for_pattern(_DEFAULT_STENCIL_JSON)
            cell_b_tile_states = gr.CheckboxGroup(
                choices=_init_tile_choices,
                value=default_cell_b_states(_DEFAULT_STENCIL_JSON),
                label="步骤②：粗单元拓扑 B（默认除 L 外全选；晶格图按 stiff≠3 映射）",
            )
            with gr.Row():
                cell_size_mm = gr.Slider(1.0, 20.0, value=5.0, step=0.5, label="单格边长 cell_size (mm)")
                extrude_thickness_mm = gr.Slider(0.5, 30.0, value=6.0, step=0.5, label="挤出厚度 (mm，2.5D/3D)")
                radius_soft_frac = gr.Slider(0.02, 0.25, value=0.08, step=0.005, label="软区杆径比例 (stiff=3)")
                radius_hard_frac = gr.Slider(0.05, 0.35, value=0.14, step=0.005, label="硬/功能杆径比例 (stiff=1)")
            unit_cell_preview = gr.Image(label="单元格 A / B 拓扑预览", type="numpy")
            unit_cell_summary = gr.Textbox(
                label="当前单元格映射说明",
                lines=5,
                interactive=False,
            )

            with gr.Accordion("高级（椭球写入细节、Chaos 场与 WFC 采样）", open=False):
                gr.Markdown(
                    "以下 **椭球写入强度 / 最小半轴** 只影响「点添加椭球」写入 JSON 的数值；**全局 α、S_mul** 与主预览高斯热图及运行 WFC 时的场图均通过 `apply_splat_hyperparams` 一致应用。"
                )
                opacity_splat = gr.Slider(5, 150, value=80, step=1, label="椭球写入强度（写入 splat.opacity，运行 WFC 前再乘 α）")
                su_floor = gr.Slider(0.01, 0.08, value=0.02, step=0.005, label="最小半轴下限")
                target_hard = gr.Slider(0.2, 0.75, value=0.45, step=0.01, label="目标硬区占比")
                chaos_scale = gr.Slider(1.0, 12.0, value=6.0, step=0.1, label="Chaos 系数")
                chaos_power = gr.Slider(0.5, 2.5, value=1.3, step=0.05, label="Chaos 幂次")
                max_attempts = gr.Slider(50, 800, value=350, step=10, label="最大随机尝试次数")
                seed = gr.Number(value=0, precision=0, label="随机种子（<0 为每次随机）")

            btn_run = gr.Button("⑤ 运行 WFC", variant="primary")
            with gr.Row():
                out_fields = gr.Image(label="场与缩略结果", type="numpy")
                out_grid = gr.Image(label="晶格构型（2D 杆系，按单元格 A/B 拓扑）", type="numpy")
            out_stl = gr.File(label="下载晶格 STL（运行 WFC 后生成）", interactive=False)
            out_msg = gr.Textbox(label="日志", lines=3)

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
            return html, mat, f"画布已设为 {r}×{c}（中央区域纵横比 = 行:列）。", r, c

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
    if port != preferred:
        print(f"[WFC] Port {preferred} is busy → http://127.0.0.1:{port}")
    build_demo().launch(
        inbrowser=False,
        server_name="127.0.0.1",
        server_port=port,
        show_api=False,
        _frontend=False,
    )
