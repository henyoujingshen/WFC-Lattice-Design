# Copyright (c) 2026 Tang Hubocheng. All rights reserved.
# Original project code: no use or redistribution without written permission.
# See the repository LICENSE for scope, exceptions and third-party rights.
# new_gaussian_field_merged.py
import numpy as np
import random
import time
from collections import deque
from copy import deepcopy
import matplotlib.pyplot as plt
import json
import os
import trimesh
import shapely.geometry
from shapely.ops import unary_union


# Part 0: Geometry Definitions for STL Generation
def get_lines_isotropic_octet_truss():
    nodes = {'bl': (0, 0), 'br': (1, 0), 'tr': (1, 1), 'tl': (0, 1), 'b': (0.5, 0), 'r': (1, 0.5), 't': (0.5, 1),
             'l': (0, 0.5), 'c': (0.5, 0.5)}
    return [(nodes['bl'], nodes['br']), (nodes['br'], nodes['tr']), (nodes['tr'], nodes['tl']),
            (nodes['tl'], nodes['bl']), (nodes['bl'], nodes['c']), (nodes['br'], nodes['c']), (nodes['tr'], nodes['c']),
            (nodes['tl'], nodes['c']), (nodes['b'], nodes['c']), (nodes['r'], nodes['c']), (nodes['t'], nodes['c']),
            (nodes['l'], nodes['c'])]


def get_lines_chiral():
    r = 0.35
    nodes = {'tl': (0.5 - r, 0.5 + r), 'tr': (0.5 + r, 0.5 + r), 'br': (0.5 + r, 0.5 - r), 'bl': (0.5 - r, 0.5 - r)}
    lines = [((0.5, 1), nodes['tl']), ((1, 0.5), nodes['tr']), ((0.5, 0), nodes['br']), ((0, 0.5), nodes['bl'])]
    lines.extend([(nodes['tl'], nodes['tr']), (nodes['tr'], nodes['br']), (nodes['br'], nodes['bl']),
                  (nodes['bl'], nodes['tl'])])
    return lines


GET_LINES_MAP = {
    "Isotropic-Octet-Truss": get_lines_isotropic_octet_truss,
    "Chiral": get_lines_chiral,
}


def get_lines_function(name):
    base_name = name.split('_')[0]
    return GET_LINES_MAP.get(base_name)


# Part 1, 2, 3: WFC Core
from visualize_new_library import draw_grid_from_wfc_result

SOCKET_EMPTY = ()
SOCKET_MIDPOINT = ('0.5',)
SOCKET_CORNERS = ('0.0', '1.0')
SOCKET_THIRDS = ('0.33', '0.67')
SOCKET_ALL = ('0.0', '0.5', '1.0')


class Prototype:
    def __init__(self, name, mesh, rotation, sockets,
                 neighbour_list): self.name = name;self.mesh = mesh;self.rotation = rotation;self.sockets = sockets;self.neighbour_list = neighbour_list

    def __repr__(self): return self.name


def get_opposite_direction(d): return {"posX": "negX", "negX": "posX", "posY": "negY", "negY": "posY"}.get(d)


def rotate_sockets(s, a):
    if a == 0: return s
    if a == 90: return {"posX": s["posY"], "negX": s["negY"], "posY": s["negX"], "negY": s["posX"]}
    if a == 180: return {"posX": s["negX"], "negX": s["posX"], "posY": s["negY"], "negY": s["posY"]}
    if a == 270: return {"posX": s["negY"], "negX": s["posY"], "posY": s["posX"], "negY": s["negX"]}


def are_sockets_compatible(s1, s2):
    if (s1 == SOCKET_ALL and s2 != SOCKET_THIRDS) or (s2 == SOCKET_ALL and s1 != SOCKET_THIRDS): return True
    return s1 == s2


def create_prototypes():
    p, t, a = [], {"Isotropic-Octet-Truss": {
        "sockets": {"posX": SOCKET_ALL, "negX": SOCKET_ALL, "posY": SOCKET_ALL, "negY": SOCKET_ALL}, "rotations": [0]},
        "Chiral": {"sockets": {"posX": SOCKET_MIDPOINT, "negX": SOCKET_MIDPOINT, "posY": SOCKET_MIDPOINT,
                               "negY": SOCKET_MIDPOINT}, "rotations": [0]}}, []
    for n, d in t.items():
        for r in d["rotations"]: a.append(
            {"name": f"{n}_{r}", "mesh": f"{n}.obj", "rotation": r, "sockets": rotate_sockets(d["sockets"], r)})
    for s in a:
        nl = {};
        for dn, st in s["sockets"].items():
            cs, od = [], get_opposite_direction(dn)
            for os in a:
                if are_sockets_compatible(st, os["sockets"][od]): cs.append(os["name"])
            nl[dn] = cs
        p.append(
            Prototype(name=s["name"], mesh=s["mesh"], rotation=s["rotation"], sockets=s["sockets"], neighbour_list=nl))
    return p


class GaussianSplat2D:
    def __init__(self, color="", mu=np.array([0., 0.]), scale=np.array([1., 1.]), angle_deg=0.,
                 opacity=1.): self.color, self.mu, self.scale, self.angle_rad, self.opacity = color, np.array(mu,
                                                                                                              dtype=float), np.array(
        scale, dtype=float), np.deg2rad(angle_deg), float(
        opacity);self.rotation_matrix = self._angle_to_rotation_matrix(
        self.angle_rad);self.cov = self._compute_covariance_matrix(self.rotation_matrix,
                                                                   self.scale);self.evaluate = self._create_evaluate_closure()

    def _angle_to_rotation_matrix(self, a): c, s = np.cos(a), np.sin(a);return np.array([[c, -s], [s, c]])

    def _compute_covariance_matrix(self, R, s): S = np.diag(s ** 2);return R @ S @ R.T

    def _create_evaluate_closure(self):
        mu, op, cov = self.mu, self.opacity, self.cov
        if np.linalg.det(cov) < 1e-10: return lambda p: np.zeros(np.asarray(p).shape[:-1])
        inv_cov = np.linalg.inv(cov)

        def evaluate(p): p = np.asarray(p);s = p.shape[:-1];pf = p.reshape(-1, 2);d = pf - mu;e = np.sum(
            np.einsum("ij,jk->ik", d, inv_cov) * d, axis=1);g = np.exp(-.5 * e);return g.reshape(s) * op

        return evaluate


class GaussianField:
    def __init__(self): self.splats_by_color = {}

    def add_splat(self, s): c = s.color;self.splats_by_color.setdefault(c, []).append(s)

    def at(self, p): p = np.asarray(p);r = {};[r.update({c: np.sum([s.evaluate(p) for s in sp], axis=0)}) for c, sp in
                                               self.splats_by_color.items() if sp];return r


def entropy(c): return len(c)


def calculate_state_preference_gaussian(r, c, sn, gf, pp):
    if not gf: return 1.
    pm = pp.get(sn, None);
    return (pm[r, c] if pm is not None else 0.) + .01


def get_lowest_entropy_cell(g, co, gf=None, pp=None):
    cc = {(i, j) for i, j, _ in co};
    me, c = float('inf'), []
    for r in range(len(g)):
        for c_ in range(len(g[0])):
            if (r, c_) in cc or not isinstance(g[r][c_], list): continue
            ce = entropy(g[r][c_]);
            if ce <= 1: continue
            if ce < me:
                me, c = ce, [(r, c_)]
            elif ce == me:
                c.append((r, c_))
    if not c: return None
    if not gf: return random.choice(c)
    bc, mops = [], -1.
    for rc, cc_ in c:
        cmsp = 0.;
        if not g[rc][cc_]: continue
        for sn in g[rc][cc_]: p = calculate_state_preference_gaussian(rc, cc_, sn, gf, pp);cmsp = max(cmsp, p)
        if cmsp > mops:
            mops, bc = cmsp, [(rc, cc_)]
        elif abs(cmsp - mops) < 1e-9:
            bc.append((rc, cc_))
    return random.choice(bc) if bc else random.choice(c)


def collapse_cell(g, c, o, gf=None, pp=None):
    i, j = c;
    ps = g[i][j]
    if not ps: raise ValueError(f"Cell ({i},{j}) has no possible states.")
    cs = ps[0] if len(ps) == 1 else random.choice(ps) if not gf else (
        random.choices(ps, weights=[calculate_state_preference_gaussian(i, j, s, gf, pp) for s in ps], k=1)[0] if sum(
            [calculate_state_preference_gaussian(i, j, s, gf, pp) for s in ps]) > 0 else random.choice(ps))
    g[i][j] = [cs];
    o.append((i, j, cs));
    return cs


def propagate(g, sc, fl):
    s = [sc]
    while s:
        ci, cj = s.pop()
        for dn, (di, dj) in {"posX": (0, 1), "negX": (0, -1), "posY": (-1, 0), "negY": (1, 0)}.items():
            ni, nj = ci + di, cj + dj
            if not (0 <= ni < len(g) and 0 <= nj < len(g[0])): continue
            nis = g[ni][nj]
            if not isinstance(nis, list) or len(nis) <= 1: continue
            pnfd = set()
            for sn in g[ci][cj]:
                p = fl.get(sn);
                if p: pnfd.update(p.neighbour_list[dn])
            oc = len(g[ni][nj]);
            g[ni][nj] = [s_ for s_ in g[ni][nj] if s_ in pnfd];
            nc = len(g[ni][nj])
            if nc < oc:
                if nc == 0: raise ValueError(f"Contradiction from ({ci},{cj}) to ({ni},{nj}).")
                if (ni, nj) not in s: s.append((ni, nj))


def wfc_algorithm(g, co, fl, gf=None):
    r, c = len(g), len(g[0]);
    ms = r * c;
    pp = None
    if gf:
        print("Pre-computing Gaussian preferences...");
        rc, cc = np.mgrid[0:r, 0:c];
        pos = np.stack((rc, cc), axis=-1);
        pp = gf.at(pos);
        print("Pre-computation finished.")
    if co:
        try:
            propagate(g, (co[0][0], co[0][1]), fl)
        except ValueError:
            return g
    S = len(co)
    while S < ms:
        cell = get_lowest_entropy_cell(g, co, gf, pp)
        if cell is None: break
        try:
            collapse_cell(g, cell, co, gf, pp);
            propagate(g, cell, fl)
        except ValueError:
            return g
        S += 1
        if S % 20 == 0 or S == ms: print(f"Step: {S}, Collapsed cells: {len(co)}/{ms}")
    return g


def visualize_grid_simple(grid):
    for row in grid: print(" ".join([f"{c[0].split('_')[0][:4]: >4}" if isinstance(c, list) and len(
        c) == 1 else f"{len(c): >4}" if isinstance(c, list) else "ERR " for c in row]))


# Part 4: 2.5D STL Generation Functionality
def rotate_point(point, angle_deg, center=(0.5, 0.5)):
    """围绕中心点旋转一个2D点"""
    angle_rad = np.radians(angle_deg)
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    x, y = point[0] - center[0], point[1] - center[1]
    x_new = x * c - y * s
    y_new = x * s + y * c
    return x_new + center[0], y_new + center[1]


def generate_2_5d_stl_from_wfc_result(grid, filename="wfc_2.5d_structure.stl", rod_radius=0.05, thickness=0.2):
    """
    根据WFC结果生成一个2.5D的板状STL文件。

    Args:
        grid (list of lists): WFC算法的输出网格。
        filename (str): 输出的STL文件名。
        rod_radius (float): 2D图案中“杆”的半径，即描粗的宽度。
        thickness (float): 沿Z轴拉伸的高度/厚度。
    """
    all_global_lines = []
    rows, cols = len(grid), len(grid[0])

    for r in range(rows):
        for c in range(cols):
            if not (grid[r][c] and isinstance(grid[r][c], list) and len(grid[r][c]) == 1):
                continue

            state_name = grid[r][c][0]
            parts = state_name.split('_')
            rotation = int(parts[1]) if len(parts) > 1 else 0

            lines_func = get_lines_function(state_name)
            if not lines_func:
                continue

            local_lines = lines_func()
            for p1_local, p2_local in local_lines:
                p1_rot = rotate_point(p1_local, -rotation)
                p2_rot = rotate_point(p2_local, -rotation)
                p1_global = (p1_rot[0] + c, p1_rot[1] + r)
                p2_global = (p2_rot[0] + c, p2_rot[1] + r)
                all_global_lines.append((p1_global, p2_global))

    if not all_global_lines:
        print("警告：没有找到任何线段来生成几何体。")
        return

    try:
        path = shapely.geometry.MultiLineString(all_global_lines)
        polygon_2d = path.buffer(rod_radius, cap_style='round', join_style='round')

        if polygon_2d.is_empty:
            print("警告：缓冲后的2D多边形为空。")
            return

    except Exception as e:
        print(f"Shapely处理时出错: {e}")
        return

    try:
        mesh_3d = trimesh.creation.extrude_polygon(polygon_2d, height=thickness)
    except Exception as e:
        print(f"Trimesh拉伸时出错: {e}")
        return

    mesh_3d.export(file_obj=filename)


# Part 5: Main Execution Logic
def load_gaussian_field_from_json(json_path, grid_size, color_map):
    print(f"Loading Gaussian field from {json_path}...")
    field = GaussianField();
    grid_rows, grid_cols = grid_size
    with open(json_path, 'r')as f:
        data = json.load(f)
    gaussians_data = data.get("field", {}).get("gaussians", [])
    if not gaussians_data: return field
    for i, g_data in enumerate(gaussians_data):
        json_color = g_data.get("color")
        if json_color not in color_map: continue
        prototype_name = color_map[json_color]
        mu_norm = g_data.get("mu", [0.5, 0.5]);
        scale_norm = g_data.get("scale", [0.1, 0.1]);
        angle_rad = g_data.get("angle", 0.0);
        opacity = g_data.get("opacity", 1.0)
        mu_grid = [(1 - mu_norm[1]) * (grid_rows - 1), mu_norm[0] * (grid_cols - 1)]
        scale_grid = [scale_norm[1] * grid_rows, scale_norm[0] * grid_cols]
        angle_deg = -np.rad2deg(angle_rad)
        splat = GaussianSplat2D(color=prototype_name, mu=mu_grid, scale=scale_grid, angle_deg=angle_deg,
                                opacity=opacity * 100.0)
        field.add_splat(splat)
        print(f"  Loaded Gaussian {i}: color='{json_color}' -> prototype='{prototype_name}'")
    return field


def main():
    start_time = time.time()

    prototypes = create_prototypes()
    fiber_block_library = {proto.name: proto for proto in prototypes}
    print(f"Generated {len(prototypes)} unique prototype states.")

    grid_size = (40, 40)
    json_file_path = "saveTheRabbit.json"
    color_map = {"red": "Isotropic-Octet-Truss_0", "green": "Chiral_0"}

    if not os.path.exists(json_file_path):
        print(f"Error: Gaussian definition file not found at '{json_file_path}'.")
        return

    gaussian_field = load_gaussian_field_from_json(json_file_path, grid_size, color_map)

    grid = [[list(fiber_block_library.keys()) for _ in range(grid_size[1])] for _ in range(grid_size[0])]
    collapse_order = []
    final_grid = wfc_algorithm(grid, collapse_order, fiber_block_library, gaussian_field)

    end_time = time.time()

    print(f"\nTotal execution time: {end_time - start_time:.2f} seconds.")
    print("\nFinal Grid State (Simple Text View):")
    visualize_grid_simple(final_grid)

    is_successful = all(isinstance(cell, list) and len(cell) == 1 for row in final_grid for cell in row)

    if is_successful:
        print(f"\nWFC completed successfully.")

        try:
            print("\nGenerating 2D visualization...")
            draw_grid_from_wfc_result(final_grid, title="WFC Final Result from JSON Field")
        except Exception as e:
            print(f"   Error during 2D visualization: {e}")

        try:
            print("\nGenerating 2.5D structure STL file...")
            output_filename = "wfc_gaussian_2.5d_result.stl"
            generate_2_5d_stl_from_wfc_result(
                grid=final_grid,
                filename=output_filename,
                rod_radius=0.05,
                thickness=0.4
            )
            print(f"Successfully generated {output_filename}")
        except Exception as e:
            print(f"   Error during STL generation: {e}")

    else:
        print("\nWFC failed to find a valid configuration or was interrupted.")


if __name__ == "__main__":
    main()