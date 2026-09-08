# Copyright (c) 2026 Tang Hubocheng. All rights reserved.
# Original project code: no use or redistribution without written permission.
# See the repository LICENSE for scope, exceptions and third-party rights.
"""Toolbox: Gaussian-guided WFC (CT rules + field-driven weights)."""

from wfc_toolbox.ct_core import (
    GaussianSplat2D,
    Prototype,
    create_wfc_prototypes,
    generate_field_maps,
    calculate_ratio,
)
from wfc_toolbox.wfc_guided import run_wfc_guided

__all__ = [
    "GaussianSplat2D",
    "Prototype",
    "create_wfc_prototypes",
    "generate_field_maps",
    "calculate_ratio",
    "run_wfc_guided",
]
