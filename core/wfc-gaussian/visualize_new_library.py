# Copyright (c) 2026 Tang Hubocheng. All rights reserved.
# Original project code: no use or redistribution without written permission.
# See the repository LICENSE for scope, exceptions and third-party rights.
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.transforms import Affine2D


# ==============================================================================
# --- 辅助函数 (已更新) ---
# ==============================================================================

def plot_lines(ax, lines, color, lw, transform=None):  # CHANGED: Added transform parameter
    """在一个Axes上绘制多条线段，并应用一个可选的变换"""
    for line in lines:
        p1, p2 = line
        # CHANGED: Passed transform to the plot function
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color=color, linewidth=lw, solid_capstyle='round', transform=transform)


def setup_ax(ax, title):
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.1, 1.1)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=9, wrap=True)
    ax.plot([0, 1, 1, 0, 0], [0, 0, 1, 1, 0], 'k--', lw=0.8, color='gray')


# ==============================================================================
# --- 所有绘图函数都已更新以接受和传递 `transform` ---
# ==============================================================================

def draw_isotropic_octet_truss(ax, color='#3477b5', lw=2.5, transform=None):  # CHANGED
    nodes = {'bl': (0, 0), 'br': (1, 0), 'tr': (1, 1), 'tl': (0, 1), 'b': (0.5, 0), 'r': (1, 0.5), 't': (0.5, 1),
             'l': (0, 0.5), 'c': (0.5, 0.5)}
    lines = [
        (nodes['bl'], nodes['br']), (nodes['br'], nodes['tr']), (nodes['tr'], nodes['tl']), (nodes['tl'], nodes['bl']),
        (nodes['bl'], nodes['c']), (nodes['br'], nodes['c']), (nodes['tr'], nodes['c']), (nodes['tl'], nodes['c']),
        (nodes['b'], nodes['c']), (nodes['r'], nodes['c']), (nodes['t'], nodes['c']), (nodes['l'], nodes['c'])
    ]
    plot_lines(ax, lines, color, lw, transform=transform)  # CHANGED


def draw_honeycomb(ax, color='#3477b5', lw=2.5, transform=None):  # CHANGED
    w = 0.25;
    p_left = (w, 0.5);
    p_right = (1 - w, 0.5)
    lines = [((0, 1), p_left), ((0, 0), p_left), ((1, 1), p_right), ((1, 0), p_right), (p_left, p_right)]
    plot_lines(ax, lines, color, lw, transform=transform)  # CHANGED


def draw_anisotropic_dense_vertical(ax, color='#3477b5', lw=2.5, transform=None):  # CHANGED
    lines = [((1 / 3, 0), (1 / 3, 1)), ((2 / 3, 0), (2 / 3, 1))]
    plot_lines(ax, lines, color, lw, transform=transform)  # CHANGED


def draw_anisotropic_dense_horizontal(ax, color='#3477b5', lw=2.5, transform=None):  # CHANGED
    lines = [((0, 1 / 3), (1, 1 / 3)), ((0, 2 / 3), (1, 2 / 3))]
    plot_lines(ax, lines, color, lw, transform=transform)  # CHANGED


def draw_anisotropic_dense_cross(ax, color='#3477b5', lw=2.5, transform=None):  # CHANGED
    lines = [((1 / 3, 0), (1 / 3, 1)), ((2 / 3, 0), (2 / 3, 1)), ((0, 1 / 3), (1, 1 / 3)), ((0, 2 / 3), (1, 2 / 3))]
    plot_lines(ax, lines, color, lw, transform=transform)  # CHANGED


def draw_anisotropic_diagonal_cross(ax, color='#3477b5', lw=2.5, transform=None):  # CHANGED
    lines = [((0, 0), (1, 1)), ((0, 1), (1, 0))]
    plot_lines(ax, lines, color, lw, transform=transform)  # CHANGED


def draw_chiral(ax, color='#3477b5', lw=2.5, transform=None):  # CHANGED
    r = 0.35
    nodes = {'tl': (0.5 - r, 0.5 + r), 'tr': (0.5 + r, 0.5 + r), 'br': (0.5 + r, 0.5 - r), 'bl': (0.5 - r, 0.5 - r)}
    lines = [((0.5, 1), nodes['tl']), ((1, 0.5), nodes['tr']), ((0.5, 0), nodes['br']), ((0, 0.5), nodes['bl'])]
    lines.extend([(nodes['tl'], nodes['tr']), (nodes['tr'], nodes['br']), (nodes['br'], nodes['bl']),
                  (nodes['bl'], nodes['tl'])])
    plot_lines(ax, lines, color, lw, transform=transform)  # CHANGED


def draw_connector_midpoint_to_corners(ax, color='#d64045', lw=2.5, transform=None):  # CHANGED
    center_node = (0.5, 0.5)
    lines = [((0, 0.5), center_node), (center_node, (1, 1)), (center_node, (1, 0))]
    plot_lines(ax, lines, color, lw, transform=transform)  # CHANGED


def draw_connector_midpoint_to_thirds(ax, color='#d64045', lw=2.5, transform=None):  # CHANGED
    center_node = (0.5, 0.5)
    lines = [((0, 0.5), center_node), (center_node, (1, 1 / 3)), (center_node, (1, 2 / 3))]
    plot_lines(ax, lines, color, lw, transform=transform)  # CHANGED


def draw_connector_corners_to_thirds(ax, color='#d64045', lw=2.5, transform=None):  # CHANGED
    lines = [((0, 1), (1, 2 / 3)), ((0, 0), (1, 1 / 3))]
    plot_lines(ax, lines, color, lw, transform=transform)  # CHANGED


def draw_connector_thirds_t_junction(ax, color='#d64045', lw=2.5, transform=None):  # CHANGED
    center_node = (0.5, 0.5)
    lines = [((1 / 3, 1), center_node), ((2 / 3, 1), center_node), (center_node, (0, 1 / 3)),
             (center_node, (0, 2 / 3)), (center_node, (1, 1 / 3)), (center_node, (1, 2 / 3))]
    plot_lines(ax, lines, color, lw, transform=transform)  # CHANGED


def draw_connector_y_split(ax, color='#d64045', lw=2.5, transform=None):  # CHANGED
    junction_point = (0.5, 0.5)
    lines = [((0.5, 0), junction_point), (junction_point, (1, 1))]
    plot_lines(ax, lines, color, lw, transform=transform)  # CHANGED


def draw_cap_midpoint_input(ax, color='#d64045', lw=2.5, transform=None):  # CHANGED
    lines = [((0, 0.5), (1, 1)), ((0, 0.5), (1, 0))]
    plot_lines(ax, lines, color, lw, transform=transform)  # CHANGED


def draw_cap_thirds_input(ax, color='#d64045', lw=2.5, transform=None):  # CHANGED
    lines = [((1 / 3, 0), (0, 1)), ((2 / 3, 0), (1, 1))]
    plot_lines(ax, lines, color, lw, transform=transform)  # CHANGED


def draw_cap_corners_input(ax, color='#d64045', lw=2.5, transform=None):  # CHANGED
    lines = [((0, 0), (0.5, 0.5)), ((0, 1), (0.5, 0.5))]
    plot_lines(ax, lines, color, lw, transform=transform)  # CHANGED


# ==============================================================================
# --- Function to Get the Right Drawing Function (无变化) ---
# ==============================================================================
def get_draw_function(name):
    draw_map = {
        "Isotropic-Octet-Truss": draw_isotropic_octet_truss, "Honeycomb": draw_honeycomb,
        "Chiral": draw_chiral, "Anisotropic-Dense-Vertical": draw_anisotropic_dense_vertical,
        "Anisotropic-Dense-Horizontal": draw_anisotropic_dense_horizontal,
        "Anisotropic-Dense-Cross": draw_anisotropic_dense_cross,
        "Anisotropic-Diagonal-Cross": draw_anisotropic_diagonal_cross,
        "Connector-Midpoint-to-Corners": draw_connector_midpoint_to_corners,
        "Connector-Midpoint-to-Thirds": draw_connector_midpoint_to_thirds,
        "Connector-Corners-to-Thirds": draw_connector_corners_to_thirds,
        "Connector-Thirds-T-Junction": draw_connector_thirds_t_junction, "Connector-Y-Split": draw_connector_y_split,
        "Cap-Midpoint-Input": draw_cap_midpoint_input, "Cap-Thirds-Input": draw_cap_thirds_input,
        "Cap-Corners-Input": draw_cap_corners_input,
    }
    base_name = name.split('_')[0]
    return draw_map.get(base_name)


# ==============================================================================
# --- Main Visualization Functions (draw_grid_from_wfc_result 已更新) ---
# ==============================================================================

def draw_entire_library():
    # 此函数无需修改，因为调用绘图函数时未使用 transform 参数，
    # 默认值 transform=None 会被正确使用。
    all_funcs = [
        (draw_isotropic_octet_truss, "1. Isotropic Octet-Truss"), (draw_honeycomb, "2. Honeycomb"),
        (draw_chiral, "5. Chiral"), (draw_anisotropic_dense_vertical, "3a. Dense Vertical"),
        (draw_anisotropic_dense_horizontal, "3b. Dense Horizontal"), (draw_anisotropic_dense_cross, "3c. Dense Cross"),
        (draw_anisotropic_diagonal_cross, "3d. Diagonal Cross"),
        (draw_connector_midpoint_to_corners, "6a. Conn: Mid-Corners"),
        (draw_connector_midpoint_to_thirds, "6b. Conn: Mid-Thirds"),
        (draw_connector_corners_to_thirds, "6c. Conn: Corners-Thirds"),
        (draw_connector_thirds_t_junction, "6d. Conn: T-Junction"),
        (draw_connector_y_split, "6e. Conn: Y-Split"),
        (draw_cap_midpoint_input, "7a. Cap: Midpoint"),
        (draw_cap_thirds_input, "7b. Cap: Thirds"),
        (draw_cap_corners_input, "7c. Cap: Corners"),
    ]
    n_items = len(all_funcs)
    n_cols = 4
    n_rows = int(np.ceil(n_items / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 3.2))
    axes = axes.flatten()

    for i, (func, title) in enumerate(all_funcs):
        setup_ax(axes[i], title)
        func(axes[i])

    for i in range(n_items, len(axes)):
        axes[i].axis('off')

    fig.suptitle("Complete Library of Unit Cells", fontsize=16, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


def draw_grid_from_wfc_result(grid, title="WFC Result"):
    """Visualizes the final grid from the WFC algorithm."""
    rows, cols = len(grid), len(grid[0])
    # 增加 figsize 使每个单元格更大更清晰
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.2, rows * 1.2))

    # 确保在1x1或1xN的情况下axes仍然是可索引的数组
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = np.array([axes])
    elif cols == 1:
        axes = np.array([[ax] for ax in axes])

    for r in range(rows):
        for c in range(cols):
            ax = axes[r, c]

            # 先绘制背景和边框
            setup_ax(ax, '')

            if grid[r][c] and isinstance(grid[r][c], list):
                state_name = grid[r][c][0]
                parts = state_name.split('_')
                rotation = int(parts[1]) if len(parts) > 1 else 0

                draw_func = get_draw_function(state_name)
                if draw_func:
                    # ==========================================================
                    #  核心修正：将旋转角度取反以匹配顺时针逻辑
                    # ==========================================================
                    # Matplotlib的rotate_deg是逆时针的，而我们的逻辑是顺时针的
                    # 因此，我们需要传递一个负角度。
                    t = Affine2D().rotate_deg_around(0.5, 0.5, -rotation)  # <-- 修正点

                    # 将变换应用到绘图函数上
                    draw_func(ax, transform=t + ax.transData)

    plt.subplots_adjust(wspace=0.05, hspace=0.05)
    fig.suptitle(title, fontsize=16)
    plt.show()


if __name__ == '__main__':
    # ================ 测试旋转绘图功能 ================
    # 创建一个测试网格来验证旋转是否正常工作
    test_grid = [
        [['Honeycomb_0'], ['Honeycomb_90'], ['Anisotropic-Dense-Vertical_90']],
        [['Connector-Midpoint-to-Corners_0'], ['Connector-Midpoint-to-Corners_90'],
         ['Connector-Midpoint-to-Corners_180']],
        [['Chiral_0'], ['Connector-Y-Split_90'], ['Connector-Y-Split_270']]
    ]
    draw_grid_from_wfc_result(test_grid, title="Rotation Test")

    # 显示完整的单元库
    # draw_entire_library()