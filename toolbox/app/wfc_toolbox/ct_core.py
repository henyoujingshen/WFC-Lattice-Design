# Copyright (c) 2026 Tang Hubocheng. All rights reserved.
# Original project code: no use or redistribution without written permission.
# See the repository LICENSE for scope, exceptions and third-party rights.
"""Gaussian splats, field maps, and CT-style WFC prototype/rules."""

from __future__ import annotations

import numpy as np


class GaussianSplat2D:
    def __init__(self, config: dict):
        self.color = config["color"]
        self.mu = np.array(config["mu"], dtype=float)
        self.scale = np.array(config["scale"], dtype=float)
        self.opacity = float(config["opacity"])
        a = np.deg2rad(config.get("angle", 0.0))
        c, s = np.cos(a), np.sin(a)
        R = np.array([[c, -s], [s, c]])
        S = np.diag(self.scale**2)
        self.cov = R @ S @ R.T
        self.inv_cov = np.linalg.inv(self.cov)

    def evaluate(self, p: np.ndarray) -> float:
        d = p - self.mu
        e = float(d.dot(self.inv_cov).dot(d))
        return self.opacity * np.exp(-0.5 * e)


class Prototype:
    def __init__(self, name: str, stiffness_level: int, char_symbol: str):
        self.name = name
        self.stiffness_level = stiffness_level
        self.char_symbol = char_symbol


def compute_chaos_components(
    val_L: float,
    val_Hd: float,
    *,
    chaos_power: float = 1.3,
    neither_power: float = 1.0,
) -> tuple[float, float]:
    """Raw weak-control (neither) and overlap terms in [0, ~1] before chaos_scale."""
    mx = max(val_L, val_Hd)
    mn = min(val_L, val_Hd)
    neither = max(0.0, 1.0 - mx) ** neither_power
    overlap = max(0.0, mn) ** chaos_power
    return float(neither), float(overlap)


def compute_chaos_raw(
    val_L: float,
    val_Hd: float,
    *,
    chaos_scale: float = 6.0,
    chaos_power: float = 1.3,
    neither_power: float = 1.0,
) -> float:
    """
    Weak-control chaos: both fields weak (neither dominates) OR both present (overlap).
    Uses raw splat sums before the +0.5 display floor.
    """
    neither, overlap = compute_chaos_components(
        val_L, val_Hd, chaos_power=chaos_power, neither_power=neither_power
    )
    return float(chaos_scale) * 0.5 * (neither + overlap)


def _logistic_stable(z: np.ndarray) -> np.ndarray:
    """Sigmoid without exp(large) overflow warnings."""
    z = np.asarray(z, dtype=float)
    out = np.empty_like(z, dtype=float)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    zneg = z[~pos]
    ez = np.exp(zneg)
    out[~pos] = ez / (1.0 + ez)
    return out


def _norm_gate_channel(x: np.ndarray, *, mode: str = "p75") -> np.ndarray:
    """Normalize a gate channel to ~[0,1] without one hot peak crushing the rest."""
    x = np.maximum(np.asarray(x, dtype=float), 0.0)
    if mode == "max":
        ref = float(np.max(x)) + 1e-9
    else:
        ref = float(np.percentile(x, 75)) + 1e-9
    return np.clip(x / ref, 0.0, 1.0)


def compute_pattern_gate_map(
    chaos_map: np.ndarray,
    *,
    weak_map: np.ndarray | None = None,
    overlap_map: np.ndarray | None = None,
    tau: float = 0.4,
    k: float = 8.0,
) -> np.ndarray:
    """
    Sigmoid gate in [0,1] for pattern seeding / stencil weight.

    When ``weak_map`` and ``overlap_map`` are given (from ``generate_field_maps``):
    score = max(norm_weak, norm_overlap), where weak uses p75 normalization so
    fringe cells with both fields weak are not drowned by a single overlap peak.
    ``tau`` is applied to this score (not to chaos/max(chaos)); lower tau -> wider yellow zone.
    """
    if weak_map is not None and overlap_map is not None:
        weak_n = _norm_gate_channel(weak_map, mode="p75")
        ovl_n = _norm_gate_channel(overlap_map, mode="max")
        score = np.maximum(weak_n, ovl_n)
        return _logistic_stable(k * (score - float(tau)))

    chaos = np.maximum(np.asarray(chaos_map, dtype=float), 0.0)
    cmax = float(np.max(chaos)) + 1e-9
    norm = chaos / cmax
    return _logistic_stable(k * (norm - float(tau)))


def generate_field_maps(
    rows: int,
    cols: int,
    splat_configs: list[dict],
    chaos_scale: float = 6.0,
    chaos_power: float = 1.3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """L / Hd / Chaos / Weak / Overlap maps (weak & overlap are raw components for gate)."""
    L_map = np.zeros((rows, cols))
    Hd_map = np.zeros((rows, cols))
    Chaos_map = np.zeros((rows, cols))
    Weak_map = np.zeros((rows, cols))
    Overlap_map = np.zeros((rows, cols))
    splats = [GaussianSplat2D(cfg) for cfg in splat_configs]

    for r in range(rows):
        for c in range(cols):
            u = c / max(1, cols - 1)
            v = r / max(1, rows - 1)
            p = np.array([u, v])
            val_L = 0.0
            val_Hd = 0.0
            for s in splats:
                if s.color == "L":
                    val_L += s.evaluate(p)
                elif s.color == "Hd":
                    val_Hd += s.evaluate(p)
            L_map[r, c] = val_L + 0.5
            Hd_map[r, c] = val_Hd + 0.5
            neither, overlap = compute_chaos_components(val_L, val_Hd, chaos_power=chaos_power)
            Weak_map[r, c] = neither
            Overlap_map[r, c] = overlap
            Chaos_map[r, c] = float(chaos_scale) * 0.5 * (neither + overlap)
    return L_map, Hd_map, Chaos_map, Weak_map, Overlap_map


def create_wfc_prototypes(pattern_type: str):
    protos: dict[str, Prototype] = {}
    rules: dict[str, dict[str, set]] = {}

    def init_rules(tile_names: list[str]):
        for n in tile_names:
            rules[n] = {
                "UP": {"L", "Hd"},
                "DOWN": {"L", "Hd"},
                "LEFT": {"L", "Hd"},
                "RIGHT": {"L", "Hd"},
            }
        rules["L"] = {d: set(tile_names) for d in ["UP", "DOWN", "LEFT", "RIGHT"]}
        rules["Hd"] = {d: set(tile_names) for d in ["UP", "DOWN", "LEFT", "RIGHT"]}

    def link(t1: str, dir_name: str, t2: str):
        opp = {"UP": "DOWN", "DOWN": "UP", "LEFT": "RIGHT", "RIGHT": "LEFT"}[dir_name]
        rules[t1][dir_name].add(t2)
        rules[t2][opp].add(t1)
        rules[t1][dir_name].discard("L")
        rules[t1][dir_name].discard("Hd")
        rules[t2][opp].discard("L")
        rules[t2][opp].discard("Hd")

    if pattern_type == "Baseline":
        names = ["L", "Hd", "H_Rand"]
        protos = {
            "L": Prototype("L", 3, "."),
            "Hd": Prototype("Hd", 1, "█"),
            "H_Rand": Prototype("H_Rand", 1, "*"),
        }
        init_rules(names)
        for d in ["UP", "DOWN", "LEFT", "RIGHT"]:
            rules["H_Rand"][d] = set(names)

    elif pattern_type == "Bridging":
        names = ["L", "Hd", "H_T", "H_1", "H_2", "H_3", "H_4", "H_M", "H_B"]
        symbols = {
            "L": ".",
            "Hd": "█",
            "H_T": "T",
            "H_1": "|",
            "H_2": "|",
            "H_3": "|",
            "H_4": "|",
            "H_M": "|",
            "H_B": "B",
        }
        protos = {n: Prototype(n, 1 if n != "L" else 3, symbols[n]) for n in names}
        init_rules(names)
        link("H_T", "DOWN", "H_1")
        link("H_1", "DOWN", "H_2")
        link("H_2", "DOWN", "H_3")
        link("H_3", "DOWN", "H_4")
        rules["H_4"]["DOWN"].update({"H_M", "H_B"})
        rules["H_M"]["UP"].add("H_4")
        rules["H_B"]["UP"].update({"H_4", "H_M"})
        rules["H_M"]["DOWN"].update({"H_M", "H_B"})

    elif pattern_type == "Deflection":
        dr_names = [f"DR_{i}" for i in range(8)]
        ur_names = [f"UR_{i}" for i in range(8)]
        names = ["L", "Hd"] + dr_names + ur_names
        symbols: dict[str, str] = {"L": ".", "Hd": "█"}
        for i in range(8):
            if i % 2 == 0:
                symbols[f"DR_{i}"] = "→"
                symbols[f"UR_{i}"] = "→"
            else:
                symbols[f"DR_{i}"] = "↓"
                symbols[f"UR_{i}"] = "↑"
        protos = {n: Prototype(n, 1 if n not in ["L"] else 3, symbols[n]) for n in names}
        init_rules(names)
        link("DR_0", "RIGHT", "DR_1")
        link("DR_1", "DOWN", "DR_2")
        link("DR_2", "RIGHT", "DR_3")
        link("DR_3", "DOWN", "DR_4")
        link("DR_4", "RIGHT", "DR_5")
        link("DR_5", "DOWN", "DR_6")
        link("DR_6", "RIGHT", "DR_7")
        link("UR_0", "RIGHT", "UR_1")
        link("UR_1", "UP", "UR_2")
        link("UR_2", "RIGHT", "UR_3")
        link("UR_3", "UP", "UR_4")
        link("UR_4", "RIGHT", "UR_5")
        link("UR_5", "UP", "UR_6")
        link("UR_6", "RIGHT", "UR_7")
        rules["DR_7"]["DOWN"].update({"DR_0", "UR_0"})
        rules["DR_7"]["RIGHT"].add("L")
        rules["UR_7"]["UP"].update({"UR_0", "DR_0"})
        rules["UR_7"]["RIGHT"].add("L")
        for i in [1, 3, 5]:
            rules[f"DR_{i}"]["LEFT"].add("L")
            rules[f"UR_{i}"]["LEFT"].add("L")
        for d in ["UP", "DOWN", "LEFT"]:
            rules["DR_0"][d] = {"L", "DR_7", "UR_7"}
            rules["UR_0"][d] = {"L", "DR_7", "UR_7"}
    else:
        raise ValueError(f"Unknown pattern_type: {pattern_type}")

    return protos, rules




def calculate_ratio(grid, protos: dict[str, Prototype]):
    total = len(grid) * len(grid[0])
    soft_count = 0
    hard_count = 0
    for row in grid:
        for cell in row:
            state = cell[0] if isinstance(cell, list) else cell
            if protos[state].stiffness_level == 3:
                soft_count += 1
            else:
                hard_count += 1
    return soft_count / total, hard_count / total
