"""
WFC with: (1) min-entropy cell, (2) tie-break by max candidate preference;
(3) field-only weights; (4) pattern head tiles gated by chaos map.
"""

from __future__ import annotations

import random
from typing import Any

import numpy as np

from wfc_toolbox.ct_core import Prototype
from wfc_toolbox.pattern_stencil import MotifSpec


def _ratio_adj(current_hard_ratio: float, target_hard_ratio: float) -> float:
    if current_hard_ratio < target_hard_ratio - 0.08:
        return 1.8
    if current_hard_ratio > target_hard_ratio + 0.08:
        return 0.3
    return 1.0


def _collapse_weight(
    r: int,
    c: int,
    s: str,
    possibilities: list[str],
    grid: list[list[Any]],
    collapsed: set[tuple[int, int]],
    rows: int,
    cols: int,
    prototypes: dict[str, Prototype],
    L_map: np.ndarray,
    Hd_map: np.ndarray,
    Chaos_map: np.ndarray,
    ratio_adj: float,
    gate_map: np.ndarray,
    head_tiles: set[str],
    motif_pat_tiles: set[str],
) -> float:
    p_L = L_map[r, c]
    p_Hd = Hd_map[r, c]
    p_Chaos = Chaos_map[r, c]
    func_states = [x for x in possibilities if x not in ("L", "Hd")]
    num_func = max(1, len(func_states))

    if s == "L":
        w = p_L * (2.0 - ratio_adj)
    elif s == "Hd":
        w = p_Hd * np.exp(-p_Chaos / 20.0) * ratio_adj
        w = max(float(w), 0.5)
    else:
        w = ((p_Chaos / num_func) + 0.5) * ratio_adj
        if s in head_tiles:
            w *= float(gate_map[r, c])
        elif s in motif_pat_tiles:
            w *= 0.15
            has_chain = False
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) in collapsed:
                    nb = grid[nr][nc][0]
                    if nb in motif_pat_tiles:
                        has_chain = True
                        break
            if has_chain:
                w *= 3.0
        else:
            has_chain = False
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) in collapsed:
                    nb = grid[nr][nc][0]
                    if nb.startswith("PAT") or nb not in ("L", "Hd"):
                        has_chain = True
                        break
            if has_chain:
                w *= 2.5
    return max(float(w), 0.001)


def _max_pref_in_cell(
    r: int,
    c: int,
    possibilities: list[str],
    grid: list[list[Any]],
    collapsed: set[tuple[int, int]],
    rows: int,
    cols: int,
    prototypes: dict[str, Prototype],
    L_map: np.ndarray,
    Hd_map: np.ndarray,
    Chaos_map: np.ndarray,
    ratio_adj: float,
    gate_map: np.ndarray,
    head_tiles: set[str],
    motif_pat_tiles: set[str],
) -> float:
    best = 0.0
    for s in possibilities:
        w = _collapse_weight(
            r,
            c,
            s,
            possibilities,
            grid,
            collapsed,
            rows,
            cols,
            prototypes,
            L_map,
            Hd_map,
            Chaos_map,
            ratio_adj,
            gate_map,
            head_tiles,
            motif_pat_tiles,
        )
        best = max(best, w)
    return best


def _pick_cell_min_entropy_max_pref(
    grid: list[list[Any]],
    collapsed: set[tuple[int, int]],
    rows: int,
    cols: int,
    prototypes: dict[str, Prototype],
    L_map: np.ndarray,
    Hd_map: np.ndarray,
    Chaos_map: np.ndarray,
    target_hard_ratio: float,
    gate_map: np.ndarray,
    head_tiles: set[str],
    motif_pat_tiles: set[str],
) -> tuple[int, int] | None:
    current_hard = sum(
        1
        for r in range(rows)
        for c in range(cols)
        if (r, c) in collapsed and prototypes[grid[r][c][0]].stiffness_level != 3
    )
    current_total = len(collapsed)
    current_hard_ratio = current_hard / max(1, current_total)
    radj = _ratio_adj(current_hard_ratio, target_hard_ratio)

    min_e = None
    for r in range(rows):
        for c in range(cols):
            if (r, c) in collapsed:
                continue
            cell = grid[r][c]
            if not isinstance(cell, list):
                continue
            e = len(cell)
            if e < 1:
                continue
            if min_e is None or e < min_e:
                min_e = e
    if min_e is None:
        return None

    pool: list[tuple[int, int]] = []
    for r in range(rows):
        for c in range(cols):
            if (r, c) in collapsed:
                continue
            cell = grid[r][c]
            if not isinstance(cell, list) or len(cell) != min_e:
                continue
            pool.append((r, c))
    if not pool:
        return None

    best_mp = -1.0
    best_cells: list[tuple[int, int]] = []
    for r, c in pool:
        mp = _max_pref_in_cell(
            r,
            c,
            grid[r][c],
            grid,
            collapsed,
            rows,
            cols,
            prototypes,
            L_map,
            Hd_map,
            Chaos_map,
            radj,
            gate_map,
            head_tiles,
            motif_pat_tiles,
        )
        if mp > best_mp:
            best_mp = mp
            best_cells = [(r, c)]
        elif abs(mp - best_mp) < 1e-12:
            best_cells.append((r, c))
    return random.choice(best_cells)


def _propagate(
    grid: list[list[Any]],
    cr: int,
    cc: int,
    collapsed: set[tuple[int, int]],
    rows: int,
    cols: int,
    rules: dict[str, dict[str, set]],
):
    stack = [(cr, cc)]
    dirs = {"UP": (-1, 0), "DOWN": (1, 0), "LEFT": (0, -1), "RIGHT": (0, 1)}
    while stack:
        r, c = stack.pop(0)
        cur = grid[r][c][0]
        for d_name, (dr, dc) in dirs.items():
            nr, nc = r + dr, c + dc
            if not (0 <= nr < rows and 0 <= nc < cols):
                continue
            if (nr, nc) in collapsed:
                continue
            valid = rules[cur][d_name]
            old_len = len(grid[nr][nc])
            grid[nr][nc] = [s for s in grid[nr][nc] if s in valid]
            if len(grid[nr][nc]) == 0:
                raise ValueError("WFC Contradiction")
            if len(grid[nr][nc]) < old_len and (nr, nc) not in stack:
                stack.append((nr, nc))


def _try_place_motif(
    grid: list[list[Any]],
    collapsed: set[tuple[int, int]],
    rows: int,
    cols: int,
    gate_map: np.ndarray,
    rules: dict[str, dict[str, set]],
    motif: MotifSpec,
    gate_threshold: float = 0.35,
    max_placements: int = 4,
):
    """Place the full coarse motif at anchors in high-gate regions (fixed relative offsets)."""
    if not motif.placements:
        return

    candidates: list[tuple[float, int, int]] = []
    for ar in range(1, rows - 1):
        for ac in range(1, cols - 1):
            if (ar, ac) in collapsed:
                continue
            cells: list[tuple[int, int, str]] = []
            ok = True
            gate_vals: list[float] = []
            for dr, dc, name in motif.placements:
                rr, cc = ar + dr, ac + dc
                if not (1 <= rr < rows - 1 and 1 <= cc < cols - 1):
                    ok = False
                    break
                if (rr, cc) in collapsed:
                    ok = False
                    break
                if name not in grid[rr][cc]:
                    ok = False
                    break
                cells.append((rr, cc, name))
                gate_vals.append(float(gate_map[rr, cc]))
            if not ok or not cells:
                continue
            score = float(np.mean(gate_vals))
            candidates.append((score, ar, ac))

    candidates.sort(key=lambda x: x[0], reverse=True)
    placed = 0
    for score, ar, ac in candidates:
        if placed >= max_placements:
            break
        if score < gate_threshold:
            continue
        order = sorted(
            ((ar + dr, ac + dc, name) for dr, dc, name in motif.placements),
            key=lambda x: (x[0], x[1]),
        )
        for rr, cc, name in order:
            if name not in grid[rr][cc]:
                break
        else:
            for rr, cc, name in order:
                grid[rr][cc] = [name]
                collapsed.add((rr, cc))
                _propagate(grid, rr, cc, collapsed, rows, cols, rules)
            placed += 1


def run_wfc_guided(
    grid_size: tuple[int, int],
    prototypes: dict[str, Prototype],
    rules: dict[str, dict[str, set]],
    L_map: np.ndarray,
    Hd_map: np.ndarray,
    Chaos_map: np.ndarray,
    target_hard_ratio: float = 0.45,
    max_attempts: int = 400,
    seed: int | None = None,
    gate_map: np.ndarray | None = None,
    head_tiles: set[str] | None = None,
    motif: MotifSpec | None = None,
    gate_tau: float = 0.4,
) -> list[list[list[str]]]:
    rows, cols = grid_size
    from wfc_toolbox.ct_core import compute_pattern_gate_map

    if gate_map is None:
        gate_map = compute_pattern_gate_map(Chaos_map, tau=gate_tau)  # legacy: chaos-only
    else:
        gate_map = np.asarray(gate_map, dtype=float).reshape(rows, cols)
    heads = set(head_tiles or [])
    motif_pats = motif.pat_tiles if motif else set()
    gate_threshold = max(0.2, float(gate_tau) * 0.85)
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    best_grid = None
    best_score = -float("inf")

    for attempt in range(max_attempts):
        try:
            grid = [[list(prototypes.keys()) for _ in range(cols)] for _ in range(rows)]
            collapsed: set[tuple[int, int]] = set()

            if motif and motif.placements and attempt % 3 == 0:
                _try_place_motif(
                    grid,
                    collapsed,
                    rows,
                    cols,
                    gate_map,
                    rules,
                    motif,
                    gate_threshold=gate_threshold,
                    max_placements=max(2, (rows * cols) // 80),
                )

            while len(collapsed) < rows * cols:
                cell = _pick_cell_min_entropy_max_pref(
                    grid,
                    collapsed,
                    rows,
                    cols,
                    prototypes,
                    L_map,
                    Hd_map,
                    Chaos_map,
                    target_hard_ratio,
                    gate_map,
                    heads,
                    motif_pats,
                )
                if cell is None:
                    break
                r, c = cell
                possibilities = grid[r][c]
                if len(possibilities) == 0:
                    raise ValueError("empty candidates")
                if len(possibilities) == 1:
                    chosen = possibilities[0]
                else:
                    current_hard = sum(
                        1
                        for rr in range(rows)
                        for cc in range(cols)
                        if (rr, cc) in collapsed
                        and prototypes[grid[rr][cc][0]].stiffness_level != 3
                    )
                    current_total = len(collapsed)
                    current_hard_ratio = current_hard / max(1, current_total)
                    radj = _ratio_adj(current_hard_ratio, target_hard_ratio)
                    weights = [
                        _collapse_weight(
                            r,
                            c,
                            s,
                            possibilities,
                            grid,
                            collapsed,
                            rows,
                            cols,
                            prototypes,
                            L_map,
                            Hd_map,
                            Chaos_map,
                            radj,
                            gate_map,
                            heads,
                            motif_pats,
                        )
                        for s in possibilities
                    ]
                    chosen = random.choices(possibilities, weights=weights, k=1)[0]

                grid[r][c] = [chosen]
                collapsed.add((r, c))
                _propagate(grid, r, c, collapsed, rows, cols, rules)

            soft_ratio, hard_ratio = _calculate_ratio_grid(grid, prototypes)
            ratio_diff = abs(hard_ratio - target_hard_ratio)
            chaos_norm = Chaos_map / (np.max(Chaos_map) + 1e-9)
            func_mask = np.zeros((rows, cols))
            for r in range(rows):
                for c in range(cols):
                    st = grid[r][c][0]
                    if st not in ("L", "Hd"):
                        func_mask[r, c] = 1.0
            interface_score = float(np.sum(func_mask * chaos_norm * gate_map) / (rows * cols + 1e-9))
            score = -ratio_diff * 10.0 + interface_score * 5.0

            if score > best_score:
                best_score = score
                best_grid = [row[:] for row in grid]

            if ratio_diff < 0.08:
                return grid

        except ValueError:
            continue

    if best_grid is not None:
        return best_grid
    return [[["L"] for _ in range(cols)] for _ in range(rows)]


def _calculate_ratio_grid(grid, protos: dict[str, Prototype]):
    total = len(grid) * len(grid[0])
    soft_count = 0
    hard_count = 0
    for row in grid:
        for cell in row:
            state = cell[0]
            if protos[state].stiffness_level == 3:
                soft_count += 1
            else:
                hard_count += 1
    return soft_count / total, hard_count / total
