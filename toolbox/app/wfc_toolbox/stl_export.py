"""2.5D lattice STL export: WFC grid + per-cell topology -> extruded rods (mm)."""

from __future__ import annotations

import tempfile
from pathlib import Path

from wfc_toolbox.ct_core import Prototype
from wfc_toolbox.unit_cells import radius_for_stiffness, topology_for_tile_state


def _ensure_triangulation_engine() -> None:
    """Trimesh ``extrude_polygon`` needs mapbox-earcut (or triangle / manifold3d)."""
    try:
        import mapbox_earcut  # noqa: F401
        return
    except ImportError:
        pass
    try:
        import trimesh

        if trimesh.triangles.available_engines():
            return
    except Exception:
        pass
    raise ImportError(
        "STL 导出需要三角剖分引擎。请运行: pip install mapbox-earcut"
    ) from None


def _grid_to_buffered_union(
    grid: list[list[list[str]]],
    protos: dict[str, Prototype],
    *,
    cell_a_id: str,
    cell_b_id: str,
    cell_b_states: set[str],
    cell_b_custom_json: str,
    cell_size_mm: float,
    radius_soft_frac: float,
    radius_hard_frac: float,
):
    from shapely.geometry import MultiLineString
    from shapely.ops import unary_union

    rows, cols = len(grid), len(grid[0])
    cs = float(cell_size_mm)
    frac_map = {3: float(radius_soft_frac), 1: float(radius_hard_frac), 2: 0.11}
    b_set = set(cell_b_states or [])
    polys = []

    for r in range(rows):
        for c in range(cols):
            cell = grid[r][c]
            state = cell[0] if isinstance(cell, list) else cell
            _, lines = topology_for_tile_state(
                state,
                cell_a_id=cell_a_id,
                cell_b_id=cell_b_id,
                cell_b_states=b_set,
                cell_b_custom_json=cell_b_custom_json or "",
                protos=protos,
            )
            rad = radius_for_stiffness(
                protos[state].stiffness_level,
                cs,
                radius_frac_map=frac_map,
            )
            for p1, p2 in lines:
                x1, y1 = (c + float(p1[0])) * cs, (r + float(p1[1])) * cs
                x2, y2 = (c + float(p2[0])) * cs, (r + float(p2[1])) * cs
                line = MultiLineString([[(x1, y1), (x2, y2)]])
                polys.append(line.buffer(rad, cap_style=1, join_style=1))

    if not polys:
        raise ValueError("No strut segments to export.")

    return unary_union(polys)


def export_lattice_stl(
    grid: list[list[list[str]]],
    protos: dict[str, Prototype],
    *,
    cell_a_id: str,
    cell_b_id: str,
    cell_b_states: set[str] | list[str],
    cell_b_custom_json: str = "",
    cell_size_mm: float = 5.0,
    extrude_thickness_mm: float = 6.0,
    radius_soft_frac: float = 0.08,
    radius_hard_frac: float = 0.14,
    output_path: str | Path | None = None,
) -> str:
    """
    Build 2.5D STL (XY lattice, extruded along Z) and return file path.
    Coordinates match 2D lattice preview (row down, column right), units mm.
    """
    import trimesh

    _ensure_triangulation_engine()

    union_2d = _grid_to_buffered_union(
        grid,
        protos,
        cell_a_id=cell_a_id,
        cell_b_id=cell_b_id,
        cell_b_states=set(cell_b_states or []),
        cell_b_custom_json=cell_b_custom_json,
        cell_size_mm=cell_size_mm,
        radius_soft_frac=radius_soft_frac,
        radius_hard_frac=radius_hard_frac,
    )

    if union_2d.is_empty:
        raise ValueError("Buffered 2D geometry is empty.")

    thickness = float(extrude_thickness_mm)
    meshes = []
    if union_2d.geom_type == "Polygon":
        meshes.append(trimesh.creation.extrude_polygon(union_2d, height=thickness))
    elif union_2d.geom_type == "MultiPolygon":
        for poly in union_2d.geoms:
            if not poly.is_empty:
                meshes.append(trimesh.creation.extrude_polygon(poly, height=thickness))
    else:
        raise ValueError(f"Unsupported geometry type: {union_2d.geom_type}")

    if not meshes:
        raise ValueError("No meshes to export.")

    mesh = meshes[0] if len(meshes) == 1 else trimesh.util.concatenate(meshes)

    if output_path is None:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".stl", prefix="wfc_lattice_")
        output_path = tmp.name
        tmp.close()

    out = Path(output_path)
    mesh.export(str(out))
    return str(out.resolve())
