# Copyright (c) 2026 Tang Hubocheng. All rights reserved.
# Original project code: no use or redistribution without written permission.
# See the repository LICENSE for scope, exceptions and third-party rights.
"""
Unit-cell topology for lattice export (2D segment patterns in [0,1]² per grid cell).

Pipeline (see also CT_GaussianField_Vertical_v2 / jason_wfc_stl):
  Gaussian field → WFC discrete tile state → per-cell line template + beam radius from stiffness_level → 2.5D/3D mesh.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Callable

import numpy as np

# CT generate_mesh: stiffness_level → radius as fraction of cell_size
DEFAULT_STIFFNESS_RADIUS_FRAC: dict[int, float] = {3: 0.08, 1: 0.14, 2: 0.11}


def get_octet_unit() -> list[tuple[tuple[float, float], tuple[float, float]]]:
    """CT_GaussianField_Vertical_v2 — corner octet + square edges."""
    nodes = {"bl": (0, 0), "br": (1, 0), "tr": (1, 1), "tl": (0, 1), "c": (0.5, 0.5)}
    internal = [
        (nodes["bl"], nodes["c"]),
        (nodes["br"], nodes["c"]),
        (nodes["tr"], nodes["c"]),
        (nodes["tl"], nodes["c"]),
    ]
    edges = [
        (nodes["bl"], nodes["br"]),
        (nodes["br"], nodes["tr"]),
        (nodes["tr"], nodes["tl"]),
        (nodes["tl"], nodes["bl"]),
    ]
    return internal + edges


def get_isotropic_octet_truss() -> list[tuple[tuple[float, float], tuple[float, float]]]:
    """jason_wfc_stl — isotropic octet truss (more mid-edge spokes)."""
    nodes = {
        "bl": (0, 0),
        "br": (1, 0),
        "tr": (1, 1),
        "tl": (0, 1),
        "b": (0.5, 0),
        "r": (1, 0.5),
        "t": (0.5, 1),
        "l": (0, 0.5),
        "c": (0.5, 0.5),
    }
    return [
        (nodes["bl"], nodes["br"]),
        (nodes["br"], nodes["tr"]),
        (nodes["tr"], nodes["tl"]),
        (nodes["tl"], nodes["bl"]),
        (nodes["bl"], nodes["c"]),
        (nodes["br"], nodes["c"]),
        (nodes["tr"], nodes["c"]),
        (nodes["tl"], nodes["c"]),
        (nodes["b"], nodes["c"]),
        (nodes["r"], nodes["c"]),
        (nodes["t"], nodes["c"]),
        (nodes["l"], nodes["c"]),
    ]


def get_chiral_unit() -> list[tuple[tuple[float, float], tuple[float, float]]]:
    """jason_wfc_stl — chiral unit (rotated square + corner links)."""
    r = 0.35
    nodes = {
        "tl": (0.5 - r, 0.5 + r),
        "tr": (0.5 + r, 0.5 + r),
        "br": (0.5 + r, 0.5 - r),
        "bl": (0.5 - r, 0.5 - r),
    }
    lines = [
        ((0.5, 1), nodes["tl"]),
        ((1, 0.5), nodes["tr"]),
        ((0.5, 0), nodes["br"]),
        ((0, 0.5), nodes["bl"]),
    ]
    lines.extend(
        [
            (nodes["tl"], nodes["tr"]),
            (nodes["tr"], nodes["br"]),
            (nodes["br"], nodes["bl"]),
            (nodes["bl"], nodes["tl"]),
        ]
    )
    return lines


@dataclass(frozen=True)
class UnitCellPreset:
    id: str
    label: str
    source: str
    description: str
    get_lines: Callable[[], list[tuple[tuple[float, float], tuple[float, float]]]]


UNIT_CELL_PRESETS: dict[str, UnitCellPreset] = {
    "octet_ct": UnitCellPreset(
        id="octet_ct",
        label="Octet（CT 试件，8 杆）",
        source="CT_GaussianField_Vertical_v2.get_octet_unit",
        description="四角连中心 + 方框边；WFC 坍缩后按 stiffness_level 统一拓扑、不同杆径（generate_mesh）。",
        get_lines=get_octet_unit,
    ),
    "isotropic_octet": UnitCellPreset(
        id="isotropic_octet",
        label="Isotropic-Octet-Truss（jason_wfc_stl）",
        source="jason_wfc_stl.get_lines_isotropic_octet_truss",
        description="边 + 角点/边中点连中心，共 12 段；适合 jason 库 WFC 原型名 Isotropic-Octet-Truss_*。",
        get_lines=get_isotropic_octet_truss,
    ),
    "chiral": UnitCellPreset(
        id="chiral",
        label="Chiral（jason_wfc_stl）",
        source="jason_wfc_stl.get_lines_chiral",
        description="手性四边形 + 四角拉向边中点；适合 Chiral_* 原型。",
        get_lines=get_chiral_unit,
    ),
}

PRESET_IDS = list(UNIT_CELL_PRESETS.keys())
PRESET_LABELS = {p.id: p.label for p in UNIT_CELL_PRESETS.values()}


def preset_id_from_label(label: str) -> str:
    for pid, p in UNIT_CELL_PRESETS.items():
        if p.label == label:
            return pid
    return "octet_ct"


def get_unit_lines(preset_id: str, custom_segments_json: str = "") -> list[tuple[tuple[float, float], tuple[float, float]]]:
    """Resolve line list: built-in preset or custom JSON segments."""
    if preset_id == "custom":
        return parse_custom_segments(custom_segments_json)
    preset = UNIT_CELL_PRESETS.get(preset_id)
    if preset is None:
        return get_octet_unit()
    return preset.get_lines()


def parse_custom_segments(text: str) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    """
    JSON: list of segments [[[u1,v1],[u2,v2]], ...] in [0,1]².
    """
    if not text or not str(text).strip():
        return []
    data = json.loads(text)
    out: list[tuple[tuple[float, float], tuple[float, float]]] = []
    for seg in data:
        p1, p2 = seg[0], seg[1]
        out.append((tuple(float(x) for x in p1), tuple(float(x) for x in p2)))
    return out


def radius_for_stiffness(
    stiffness_level: int,
    cell_size: float,
    *,
    radius_frac_map: dict[int, float] | None = None,
) -> float:
    frac_map = radius_frac_map or DEFAULT_STIFFNESS_RADIUS_FRAC
    frac = frac_map.get(int(stiffness_level), 0.11)
    return float(frac) * float(cell_size)


def use_cell_b_topology(
    tile_state: str,
    *,
    cell_a_id: str,
    cell_b_id: str,
    cell_b_states: set[str],
    protos: dict | None = None,
) -> bool:
    """
    Fine cell A: only ``L`` (stiffness_level 3).
    Coarse cell B: ``Hd``, ``PAT*``, and other non-soft tiles (stiffness != 3).
    """
    if cell_b_id == cell_a_id:
        return False
    if tile_state == "L":
        return False
    if protos and tile_state in protos:
        return int(protos[tile_state].stiffness_level) != 3
    return tile_state in cell_b_states


def topology_for_tile_state(
    tile_state: str,
    *,
    cell_a_id: str,
    cell_b_id: str,
    cell_b_states: set[str],
    cell_b_custom_json: str = "",
    protos: dict | None = None,
) -> tuple[str, list[tuple[tuple[float, float], tuple[float, float]]]]:
    """Pick unit cell A (fine) or B (coarse) for a collapsed WFC tile name."""
    use_b = use_cell_b_topology(
        tile_state,
        cell_a_id=cell_a_id,
        cell_b_id=cell_b_id,
        cell_b_states=cell_b_states,
        protos=protos,
    )
    pid = cell_b_id if use_b else cell_a_id
    custom = cell_b_custom_json if use_b and cell_b_id == "custom" else ""
    return pid, get_unit_lines(pid, custom)


def tile_choices_for_pattern(stencil_grid_json: str | None = None) -> list[str]:
    """WFC tile names for CheckboxGroup (assign cell B)."""
    from wfc_toolbox.pattern_stencil import compile_stencil, stencil_from_grid, parse_stencil_grid_json

    protos, _, _, _, _ = compile_stencil(stencil_from_grid(parse_stencil_grid_json(stencil_grid_json)))
    return sorted(protos.keys())


def default_cell_b_states(stencil_grid_json: str | None = None) -> list[str]:
    """Default: all non-L tiles use cell B (Hd, PAT*, H_Rand, …)."""
    names = tile_choices_for_pattern(stencil_grid_json)
    return [n for n in names if n != "L"]
