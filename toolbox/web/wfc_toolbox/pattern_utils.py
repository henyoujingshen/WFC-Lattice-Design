# Copyright (c) 2026 Tang Hubocheng. All rights reserved.
# Original project code: no use or redistribution without written permission.
# See the repository LICENSE for scope, exceptions and third-party rights.
"""Splat hyperparameters (pattern brush removed; use pattern_stencil instead)."""

from __future__ import annotations


def apply_splat_hyperparams(splats: list[dict], scale_multiplier: float, opacity_regulator: float) -> list[dict]:
    """Apply paper-style S_mul and alpha to a copy of splat configs."""
    sm = float(scale_multiplier)
    ar = float(opacity_regulator)
    out = []
    for cfg in splats:
        d = dict(cfg)
        sc = d.get("scale", [0.1, 0.1])
        d["scale"] = [float(sc[0]) * sm, float(sc[1]) * sm]
        d["opacity"] = float(d.get("opacity", 80.0)) * ar
        out.append(d)
    return out
