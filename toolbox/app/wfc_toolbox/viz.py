"""Matplotlib previews for Gradio (numpy RGB uint8)."""

from __future__ import annotations

import io

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

from wfc_toolbox.ct_core import GaussianSplat2D, Prototype
from wfc_toolbox.pattern_utils import apply_splat_hyperparams

# Matplotlib splat preview width (px); keep in sync with HTML canvas default in splat_canvas_html.
SPLAT_PREVIEW_PIXEL_W = 640

# Soft preview: maps raw splat sum (after α on opacity) to overlay alpha via tanh(field / tau).
_SPLAT_PREVIEW_TAU = 32.0
_L_RGB = np.array([0.10, 0.42, 0.95], dtype=np.float32)
_HD_RGB = np.array([0.92, 0.14, 0.12], dtype=np.float32)


def _raster_splat_fields(
    splat_configs: list[dict],
    *,
    nx: int,
    ny: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Accumulate L / Hd Gaussian splat values on a normalized [0,1]² grid (v increases with row index)."""
    uu = np.linspace(0.0, 1.0, nx)
    vv = np.linspace(0.0, 1.0, ny)
    UU, VV = np.meshgrid(uu, vv)
    L_acc = np.zeros_like(UU, dtype=np.float64)
    Hd_acc = np.zeros_like(UU, dtype=np.float64)
    for cfg in splat_configs:
        gs = GaussianSplat2D(cfg)
        inv = gs.inv_cov
        mu = gs.mu
        du = UU - mu[0]
        dv = VV - mu[1]
        e = inv[0, 0] * du * du + 2.0 * inv[0, 1] * du * dv + inv[1, 1] * dv * dv
        val = float(gs.opacity) * np.exp(-0.5 * e)
        if gs.color == "L":
            L_acc += val
        elif gs.color == "Hd":
            Hd_acc += val
    return L_acc, Hd_acc


def _field_rgb_to_bg(L_acc: np.ndarray, Hd_acc: np.ndarray, *, tau: float) -> np.ndarray:
    """RGBA-style blend onto light gray; returns float RGB (ny, nx, 3) in [0, 1]."""
    bg = np.ones((*L_acc.shape, 3), dtype=np.float32) * 0.96
    a_l = np.clip(np.tanh(L_acc / max(tau, 1e-6)), 0.0, 0.90).astype(np.float32)[..., None]
    a_h = np.clip(np.tanh(Hd_acc / max(tau, 1e-6)), 0.0, 0.90).astype(np.float32)[..., None]
    blue = _L_RGB.reshape(1, 1, 3)
    red = _HD_RGB.reshape(1, 1, 3)
    img = bg * (1.0 - a_l) + blue * a_l
    img = img * (1.0 - a_h) + red * a_h
    return np.clip(img, 0.0, 1.0)


def fields_and_result_to_image(
    L_map: np.ndarray,
    Hd_map: np.ndarray,
    Chaos_map: np.ndarray,
    grid: list[list[list[str]]],
    protos: dict[str, Prototype],
    title: str = "",
) -> np.ndarray:
    rows, cols = len(grid), len(grid[0])
    result_map = np.zeros((rows, cols))
    for r in range(rows):
        for c in range(cols):
            state = grid[r][c][0]
            if protos[state].stiffness_level == 3:
                result_map[r, c] = 0
            elif state == "Hd":
                result_map[r, c] = 1
            else:
                result_map[r, c] = 2

    fig, axs = plt.subplots(1, 4, figsize=(20, 5))
    if title:
        fig.suptitle(title, fontsize=12)

    im0 = axs[0].imshow(L_map, cmap="Blues", aspect="auto", vmin=L_map.min(), vmax=L_map.max())
    axs[0].set_title("L (soft)")
    fig.colorbar(im0, ax=axs[0], fraction=0.046, pad=0.04)

    im1 = axs[1].imshow(Hd_map, cmap="Reds", aspect="auto", vmin=Hd_map.min(), vmax=Hd_map.max())
    axs[1].set_title("Hd (hard)")
    fig.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.04)

    im2 = axs[2].imshow(Chaos_map, cmap="magma", aspect="auto")
    axs[2].set_title("Chaos (overlap)")
    fig.colorbar(im2, ax=axs[2], fraction=0.046, pad=0.04)

    cmap_result = ListedColormap(["#E0F7FA", "#607D8B", "#E53935"])
    axs[3].imshow(result_map, cmap=cmap_result, vmin=0, vmax=2, aspect="auto")
    axs[3].set_title("WFC")
    for r in range(rows):
        for c in range(cols):
            sym = protos[grid[r][c][0]].char_symbol
            color = "black" if result_map[r, c] == 0 else "white"
            fs = max(4, min(10, 120 // max(rows, cols)))
            axs[3].text(c, r, sym, ha="center", va="center", fontsize=fs, color=color, fontweight="bold")

    for ax in axs:
        ax.set_xlabel("col")
        ax.set_ylabel("row")

    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    from PIL import Image

    img = Image.open(buf).convert("RGB")
    return np.array(img, dtype=np.uint8)


def splats_canvas_to_image(
    splat_configs: list[dict],
    title: str = "Splats [0,1]^2",
    *,
    aspect_cols: int = 18,
    aspect_rows: int = 15,
    pixel_width: int = SPLAT_PREVIEW_PIXEL_W,
    dpi: int = 100,
    show_cell_grid: bool = True,
    opacity_regulator: float = 1.0,
    scale_multiplier: float = 1.0,
    gate_map: np.ndarray | None = None,
) -> np.ndarray:
    """Preview splats in [0,1]^2; PNG size matches HTML canvas (pixel_width × pixel_width*rows/cols).

    Plot uses **cell coordinates** X=u·N, Y=v·M with ``set_aspect("equal")`` so each grid cell is a
    square on screen (same as the HTML canvas, which scales [0,1]^2 to a cols:rows rectangle).

    Filled Gaussian blobs use the same ``GaussianSplat2D`` kernel as ``generate_field_maps``, with
    ``apply_splat_hyperparams`` (S_mul on scale, α on opacity) so α / S_mul 滑块与场图一致。
    """
    rows = max(1, int(aspect_rows))
    cols = max(1, int(aspect_cols))
    pixel_h = max(1, int(round(int(pixel_width) * rows / cols)))
    fw = int(pixel_width) / dpi
    fh = pixel_h / dpi

    splats_eff = apply_splat_hyperparams(
        list(splat_configs),
        float(scale_multiplier),
        float(opacity_regulator),
    )

    fig, ax = plt.subplots(1, 1, figsize=(fw, fh), dpi=dpi)
    ax.set_xlim(0, cols)
    ax.set_ylim(rows, 0)
    ax.set_aspect("equal")
    ax.set_xlabel("u (column)")
    ax.set_ylabel("v (row, down)")
    ax.set_title(title, fontsize=10)
    xt = np.linspace(0, cols, 5)
    ax.set_xticks(xt, [f"{x / cols:.2f}" for x in xt])
    yt = np.linspace(0, rows, 5)
    ax.set_yticks(yt, [f"{y / rows:.2f}" for y in yt])

    if splats_eff:
        nx = int(np.clip(14 * cols, 96, 420))
        ny = int(np.clip(14 * rows, 96, 420))
        L_acc, Hd_acc = _raster_splat_fields(splats_eff, nx=nx, ny=ny)
        rgb = _field_rgb_to_bg(L_acc, Hd_acc, tau=_SPLAT_PREVIEW_TAU)
        ax.imshow(
            rgb,
            extent=[0, cols, rows, 0],
            origin="upper",
            interpolation="bilinear",
            zorder=1,
        )

    if gate_map is not None and gate_map.size > 0:
        gm = np.asarray(gate_map, dtype=float)
        if gm.shape == (rows, cols):
            rgba = np.zeros((rows, cols, 4), dtype=float)
            rgba[..., 0] = 1.0
            rgba[..., 1] = 0.85
            rgba[..., 2] = 0.0
            rgba[..., 3] = np.clip(gm * 0.45, 0.0, 0.45)
            ax.imshow(rgba, extent=[0, cols, rows, 0], origin="upper", interpolation="nearest", zorder=2)

    if show_cell_grid:
        for i in range(cols + 1):
            ax.axvline(i, color="#5c5c5c", linewidth=0.85, alpha=0.55, zorder=3)
        for j in range(rows + 1):
            ax.axhline(j, color="#5c5c5c", linewidth=0.85, alpha=0.55, zorder=3)

    fig.subplots_adjust(left=0.1, right=0.97, top=0.9, bottom=0.12)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi)
    plt.close(fig)
    buf.seek(0)
    from PIL import Image

    return np.array(Image.open(buf).convert("RGB"), dtype=np.uint8)


def stencil_preview_image(cells: list[list[str]]) -> np.ndarray:
    """5x5 stencil: L=green, H=red, .=gray."""
    from wfc_toolbox.pattern_stencil import STENCIL_SIZE, PatternStencil

    norm = PatternStencil(cells=cells).normalized()
    cmap = {".": "#eeeeee", "L": "#81C784", "H": "#E53935"}
    arr = np.zeros((STENCIL_SIZE, STENCIL_SIZE, 3), dtype=float)
    for r in range(STENCIL_SIZE):
        for c in range(STENCIL_SIZE):
            hex_c = cmap.get(norm[r][c], "#eeeeee")
            arr[r, c] = [int(hex_c[i : i + 2], 16) / 255.0 for i in (1, 3, 5)]
    fig, ax = plt.subplots(1, 1, figsize=(2.4, 2.4))
    ax.imshow(arr, aspect="equal", interpolation="nearest")
    ax.set_xticks(range(STENCIL_SIZE))
    ax.set_yticks(range(STENCIL_SIZE))
    ax.set_xticklabels(range(1, STENCIL_SIZE + 1))
    ax.set_yticklabels(range(1, STENCIL_SIZE + 1))
    ax.set_title("5x5 stencil (L / H / .)")
    for i in range(STENCIL_SIZE + 1):
        ax.axhline(i - 0.5, color="#888", lw=0.5)
        ax.axvline(i - 0.5, color="#888", lw=0.5)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    from PIL import Image

    return np.array(Image.open(buf).convert("RGB"), dtype=np.uint8)


def collapsed_grid_large_image(
    grid: list[list[list[str]]],
    protos: dict[str, Prototype],
    title: str = "WFC collapsed grid",
    max_fig_width: float = 14.0,
) -> np.ndarray:
    """单张大图：每格显示类别色 + 原型符号，纵横比与网格一致。"""
    rows, cols = len(grid), len(grid[0])
    result_map = np.zeros((rows, cols))
    for r in range(rows):
        for c in range(cols):
            state = grid[r][c][0]
            if protos[state].stiffness_level == 3:
                result_map[r, c] = 0
            elif state == "Hd":
                result_map[r, c] = 1
            else:
                result_map[r, c] = 2

    ar = rows / max(1, cols)
    fw = min(max_fig_width, max(6.0, cols * 0.35))
    fh = fw * ar
    fig, ax = plt.subplots(1, 1, figsize=(fw, fh))
    cmap_result = ListedColormap(["#E0F7FA", "#607D8B", "#E53935"])
    ax.imshow(result_map, cmap=cmap_result, vmin=0, vmax=2, aspect="equal", interpolation="nearest")
    ax.set_xticks(np.arange(-0.5, cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, rows, 1), minor=True)
    ax.grid(which="minor", color="#444444", linewidth=0.4, alpha=0.5)
    ax.tick_params(which="both", bottom=False, left=False, labelbottom=False, labelleft=False)
    fs = max(5, min(14, int(180 / max(rows, cols))))
    for r in range(rows):
        for c in range(cols):
            sym = protos[grid[r][c][0]].char_symbol
            color = "black" if result_map[r, c] == 0 else "white"
            ax.text(c, r, sym, ha="center", va="center", fontsize=fs, color=color, fontweight="bold")
    ax.set_title(title)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    from PIL import Image

    return np.array(Image.open(buf).convert("RGB"), dtype=np.uint8)


def _draw_unit_cell_on_ax(
    ax,
    lines: list[tuple[tuple[float, float], tuple[float, float]]],
    *,
    title: str,
) -> None:
    ax.set_xlim(-0.08, 1.08)
    ax.set_ylim(1.08, -0.08)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("u")
    ax.set_ylabel("v (down)")
    for i in range(2):
        ax.axvline(i, color="#ccc", lw=0.5)
        ax.axhline(i, color="#ccc", lw=0.5)
    for p1, p2 in lines:
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color="#1565C0", lw=2.2, solid_capstyle="round", zorder=2)


def unit_cell_preview_image(
    cell_a_id: str,
    cell_b_id: str,
    cell_b_custom_json: str,
    *,
    radius_frac_soft: float = 0.08,
    radius_frac_hard: float = 0.14,
) -> np.ndarray:
    """Side-by-side preview of unit cell A and B (2D strut topology in [0,1]^2)."""
    del radius_frac_soft, radius_frac_hard  # reserved for future STL / line-width mapping
    from wfc_toolbox.unit_cells import get_unit_lines

    lines_a = get_unit_lines(cell_a_id, "")
    lines_b = get_unit_lines(cell_b_id, cell_b_custom_json if cell_b_id == "custom" else "")
    _plot_names = {
        "octet_ct": "Octet-CT (8 struts)",
        "isotropic_octet": "Isotropic-Octet-Truss",
        "chiral": "Chiral",
        "custom": "Custom JSON segments",
    }
    la = _plot_names.get(cell_a_id, cell_a_id)
    lb = _plot_names.get(cell_b_id, cell_b_id)

    fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.6))
    _draw_unit_cell_on_ax(axs[0], lines_a, title=f"Cell A: {la}")
    _draw_unit_cell_on_ax(axs[1], lines_b, title=f"Cell B: {lb}")
    fig.suptitle("Unit strut topology in [0,1]^2 (per grid cell)", fontsize=10, y=1.02)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    from PIL import Image

    return np.array(Image.open(buf).convert("RGB"), dtype=np.uint8)


_CELL_LABELS = {
    "octet_ct": "Octet-CT",
    "isotropic_octet": "Isotropic-Octet-Truss",
    "chiral": "Chiral",
    "custom": "Custom JSON",
}


def collapsed_grid_lattice_image(
    grid: list[list[list[str]]],
    protos: dict[str, Prototype],
    *,
    cell_a_id: str,
    cell_b_id: str,
    cell_b_states: list[str] | set[str],
    cell_b_custom_json: str = "",
    title: str = "Lattice configuration (2D struts)",
    max_fig_width: float = 16.0,
) -> np.ndarray:
    """
    M×N lattice strip: each WFC cell draws unit-cell struts in global coordinates (v down).
    Cell B assignment follows ``topology_for_tile_state`` (same as STL path).
    """
    from wfc_toolbox.unit_cells import topology_for_tile_state, use_cell_b_topology

    rows, cols = len(grid), len(grid[0])
    b_set = set(cell_b_states or [])

    ar = rows / max(1, cols)
    fw = min(max_fig_width, max(7.0, cols * 0.42))
    fh = fw * ar
    fig, ax = plt.subplots(1, 1, figsize=(fw, fh))
    ax.set_facecolor("#f5f5f5")
    ax.set_aspect("equal")
    ax.set_xlim(-0.02, cols + 0.02)
    ax.set_ylim(rows + 0.02, -0.02)
    ax.set_xticks([])
    ax.set_yticks([])

    for i in range(cols + 1):
        ax.axvline(i, color="#bdbdbd", linewidth=0.35, alpha=0.85, zorder=0)
    for j in range(rows + 1):
        ax.axhline(j, color="#bdbdbd", linewidth=0.35, alpha=0.85, zorder=0)

    lw_a, lw_b = 1.6, 2.4
    col_a, col_b = "#455A64", "#1565C0"

    for r in range(rows):
        for c in range(cols):
            state = grid[r][c][0] if isinstance(grid[r][c], list) else grid[r][c]
            _, lines = topology_for_tile_state(
                state,
                cell_a_id=cell_a_id,
                cell_b_id=cell_b_id,
                cell_b_states=b_set,
                cell_b_custom_json=cell_b_custom_json or "",
                protos=protos,
            )
            use_b = use_cell_b_topology(
                state,
                cell_a_id=cell_a_id,
                cell_b_id=cell_b_id,
                cell_b_states=b_set,
                protos=protos,
            )
            lw = lw_b if use_b else lw_a
            color = col_b if use_b else col_a
            for p1, p2 in lines:
                x1, y1 = c + float(p1[0]), r + float(p1[1])
                x2, y2 = c + float(p2[0]), r + float(p2[1])
                ax.plot(
                    [x1, x2],
                    [y1, y2],
                    color=color,
                    lw=lw,
                    solid_capstyle="round",
                    zorder=2,
                )

    la = _CELL_LABELS.get(cell_a_id, cell_a_id)
    lb = _CELL_LABELS.get(cell_b_id, cell_b_id)
    ax.plot([], [], color=col_a, lw=lw_a, label=f"A: {la}")
    ax.plot([], [], color=col_b, lw=lw_b, label=f"B: {lb}")
    ax.legend(loc="upper right", fontsize=8, framealpha=0.92)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("column")
    ax.set_ylabel("row (down)")
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    from PIL import Image

    return np.array(Image.open(buf).convert("RGB"), dtype=np.uint8)


def lattice_pipeline_diagram_image() -> np.ndarray:
    """Static schematic: Gaussian field → WFC → unit cell + thickness → 3D."""
    fig, ax = plt.subplots(1, 1, figsize=(9.0, 2.2))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 2)
    ax.axis("off")
    boxes = [
        (0.2, 0.55, 1.8, 0.9, "Gaussian field\nL / Hd / Chaos"),
        (2.4, 0.55, 1.8, 0.9, "WFC collapse\ndiscrete tiles"),
        (4.6, 0.55, 2.0, 0.9, "Unit topology\n[0,1]^2 struts"),
        (7.0, 0.55, 2.2, 0.9, "rod radius x cell_size\n-> 2.5D/3D STL"),
    ]
    for x, y, w, h, txt in boxes:
        ax.add_patch(plt.Rectangle((x, y), w, h, fill=True, facecolor="#E3F2FD", edgecolor="#1565C0", lw=1.5))
        ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=9)
    for x0, x1 in [(2.0, 2.4), (4.2, 4.6), (6.6, 7.0)]:
        ax.annotate("", xy=(x1, 1.0), xytext=(x0, 1.0), arrowprops=dict(arrowstyle="->", lw=1.5))
    ax.set_title("Gaussian field -> lattice (field + WFC + unit cells + STL)", fontsize=10)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    from PIL import Image

    return np.array(Image.open(buf).convert("RGB"), dtype=np.uint8)
