"""5×5 coarse-unit motif → WFC prototypes/rules (fixed relative layout)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Literal

from wfc_toolbox.ct_core import Prototype, create_wfc_prototypes

STENCIL_SIZE = 5


@dataclass
class MotifSpec:
    """Coarse cells in stencil coordinates; anchor is min (r,c); placements are relative offsets."""

    anchor: tuple[int, int]
    placements: list[tuple[int, int, str]]  # dr, dc, tile_name

    @property
    def head_tile(self) -> str:
        for dr, dc, name in self.placements:
            if dr == 0 and dc == 0:
                return name
        return self.placements[0][2]

    @property
    def pat_tiles(self) -> set[str]:
        return {name for _, _, name in self.placements}


@dataclass
class PatternStencil:
    """5×5 grid: 1 = coarse unit, 0 = fine (implicit)."""

    grid: list[list[int]] = field(default_factory=lambda: [[0] * STENCIL_SIZE for _ in range(STENCIL_SIZE)])

    def coarse_positions(self) -> list[tuple[int, int]]:
        out: list[tuple[int, int]] = []
        g = self.normalized_grid()
        for r in range(STENCIL_SIZE):
            for c in range(STENCIL_SIZE):
                if g[r][c]:
                    out.append((r, c))
        return out

    def normalized_grid(self) -> list[list[int]]:
        g = self.grid[:STENCIL_SIZE]
        out: list[list[int]] = []
        for i in range(STENCIL_SIZE):
            row = list(g[i]) if i < len(g) else []
            while len(row) < STENCIL_SIZE:
                row.append(0)
            out.append([1 if int(row[j]) else 0 for j in range(STENCIL_SIZE)])
        while len(out) < STENCIL_SIZE:
            out.append([0] * STENCIL_SIZE)
        return out[:STENCIL_SIZE]

    def to_cells_legacy(self) -> list[list[str]]:
        """Internal H/. representation for link building."""
        g = self.normalized_grid()
        return [["H" if g[r][c] else "." for c in range(STENCIL_SIZE)] for r in range(STENCIL_SIZE)]


def empty_grid() -> list[list[int]]:
    return [[0] * STENCIL_SIZE for _ in range(STENCIL_SIZE)]


def parse_stencil_grid_json(raw: str | None) -> list[list[int]]:
    if not raw or not str(raw).strip():
        return empty_grid()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return empty_grid()
    if isinstance(data, dict) and "grid" in data:
        data = data["grid"]
    if not isinstance(data, list):
        return empty_grid()
    st = PatternStencil(grid=data)
    return st.normalized_grid()


def stencil_from_grid(grid: list[list[int]] | None) -> PatternStencil:
    if grid is None:
        return PatternStencil()
    return PatternStencil(grid=grid)


def stencil_from_dataframe(df) -> PatternStencil:
    """Legacy: map L/H/. dataframe → binary grid (H or L → coarse)."""
    if df is None:
        return PatternStencil()
    try:
        import pandas as pd

        if isinstance(df, pd.DataFrame):
            arr = df.astype(str).to_numpy()
        else:
            arr = [[str(x) for x in row] for row in df]
    except Exception:
        arr = df
    grid = []
    for i in range(STENCIL_SIZE):
        row = []
        for j in range(STENCIL_SIZE):
            try:
                t = str(arr[i][j]).strip().upper()
            except (IndexError, TypeError):
                t = "."
            row.append(1 if t in ("H", "HARD", "FUNC", "F", "L", "SOFT", "S", "1") else 0)
        grid.append(row)
    return PatternStencil(grid=grid)


def preset_bridging_vert() -> PatternStencil:
    g = empty_grid()
    for r in range(STENCIL_SIZE):
        g[r][2] = 1
    return PatternStencil(grid=g)


def preset_bridging_horiz() -> PatternStencil:
    g = empty_grid()
    for c in range(STENCIL_SIZE):
        g[2][c] = 1
    return PatternStencil(grid=g)


def preset_deflection_snake() -> PatternStencil:
    g = empty_grid()
    path = [(0, 0), (0, 1), (1, 1), (1, 2), (2, 2), (2, 3), (3, 3), (3, 4), (4, 4)]
    for r, c in path:
        if r < STENCIL_SIZE and c < STENCIL_SIZE:
            g[r][c] = 1
    return PatternStencil(grid=g)


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    out = []
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nr, nc = r + dr, c + dc
        if 0 <= nr < STENCIL_SIZE and 0 <= nc < STENCIL_SIZE:
            out.append((nr, nc))
    return out


def _dir_between(a: tuple[int, int], b: tuple[int, int]) -> str:
    dr, dc = b[0] - a[0], b[1] - a[1]
    if dr == 1 and dc == 0:
        return "DOWN"
    if dr == -1 and dc == 0:
        return "UP"
    if dc == 1 and dr == 0:
        return "RIGHT"
    if dc == -1 and dr == 0:
        return "LEFT"
    raise ValueError(f"non-adjacent cells {a} {b}")


def compile_stencil(
    stencil: PatternStencil,
) -> tuple[dict[str, Prototype], dict[str, dict[str, set]], set[str], str, MotifSpec | None]:
    """
    One motif for all coarse cells: fixed relative offsets from a single anchor.
    4-adjacent coarse pairs get directional links; non-adjacent coarse cells stay
    in the same motif and are placed together at seed time.
    """
    hs = stencil.coarse_positions()
    if not hs:
        protos, rules = create_wfc_prototypes("Baseline")
        return protos, rules, set(), "无粗单元（仅细单元 L / Hd）。", None

    r0, c0 = min(hs)  # lexicographic min among coarse cells (not min-row × min-col)
    pos_to_name: dict[tuple[int, int], str] = {}
    for r, c in hs:
        pos_to_name[(r, c)] = f"PAT_r{r}c{c}"

    protos: dict[str, Prototype] = {
        "L": Prototype("L", 3, "."),
        "Hd": Prototype("Hd", 1, "#"),
    }
    for name in pos_to_name.values():
        protos[name] = Prototype(name, 1, "+")

    all_names = sorted(protos.keys())
    rules: dict[str, dict[str, set]] = {}
    for n in all_names:
        rules[n] = {d: {"L", "Hd"} for d in ("UP", "DOWN", "LEFT", "RIGHT")}
    rules["L"] = {d: set(all_names) for d in ("UP", "DOWN", "LEFT", "RIGHT")}
    rules["Hd"] = {d: set(all_names) for d in ("UP", "DOWN", "LEFT", "RIGHT")}

    links_log: list[str] = []
    hset = set(hs)
    for r, c in hs:
        t1 = pos_to_name[(r, c)]
        for nr, nc in _neighbors4(r, c):
            if (nr, nc) not in hset or (nr, nc) < (r, c):
                continue
            d = _dir_between((r, c), (nr, nc))
            opp = {"UP": "DOWN", "DOWN": "UP", "LEFT": "RIGHT", "RIGHT": "LEFT"}[d]
            t2 = pos_to_name[(nr, nc)]
            rules[t1][d].add(t2)
            rules[t2][opp].add(t1)
            rules[t1][d].discard("L")
            rules[t1][d].discard("Hd")
            rules[t2][opp].discard("L")
            rules[t2][opp].discard("Hd")
            links_log.append(f"link({t1}, {d}, {t2})")

    placements = [
        (r - r0, c - c0, pos_to_name[(r, c)])
        for r, c in sorted(hs)
    ]
    head_name = pos_to_name[(r0, c0)]
    head_tiles = {head_name}
    motif = MotifSpec(anchor=(r0, c0), placements=placements)

    offset_txt = ", ".join(f"({dr},{dc})→{nm}" for dr, dc, nm in placements[:8])
    if len(placements) > 8:
        offset_txt += ", ..."
    summary = (
        f"粗单元模板 {len(hs)} 格，锚点 stencil({r0},{c0})→{head_name}；"
        f"整组相对偏移一次放置。\n"
        f"偏移: {offset_txt}\n"
        + "\n".join(links_log[:20])
        + ("\n..." if len(links_log) > 20 else "")
    )
    return protos, rules, head_tiles, summary, motif
