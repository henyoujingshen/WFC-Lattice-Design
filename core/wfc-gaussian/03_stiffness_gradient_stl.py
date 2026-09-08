import numpy as np
import random
import time
from collections import deque
import json
import os
import trimesh
import shapely.geometry
from shapely.ops import unary_union
from typing import Dict, List, Tuple


# ... [Part 0 to Part 3 remain unchanged] ...
# -----------------------------------10_three stiffness_stl.py---------------------------------------
# Part 0: Geometry Definitions (FROM STIFFNESS SCRIPT)
# --------------------------------------------------------------------------
def get_lines_stiff_high():
    """高刚度 (Level 1): Isotropic-Octet-Truss + 边缘框架"""
    nodes = {'bl': (0, 0), 'br': (1, 0), 'tr': (1, 1), 'tl': (0, 1), 'c': (0.5, 0.5)}
    internal_lines = [(nodes['bl'], nodes['c']), (nodes['br'], nodes['c']), (nodes['tr'], nodes['c']),
                      (nodes['tl'], nodes['c'])]
    edge_bars = [(nodes['bl'], nodes['br']), (nodes['br'], nodes['tr']), (nodes['tr'], nodes['tl']),
                 (nodes['tl'], nodes['bl'])]
    return internal_lines + edge_bars


def get_lines_stiff_medium():
    """中刚度 (Level 2): 带有较小的中心星形 + 边缘框架"""
    p, d = 0.25, 0.18
    c = (0.5, 0.5)
    p_tl, p_tr, p_br, p_bl = (c[0] - p, c[1] + p), (c[0] + p, c[1] + p), (c[0] + p, c[1] - p), (c[0] - p, c[1] - p)
    v_t, v_r, v_b, v_l = (c[0], c[1] + d), (c[0] + d, c[1]), (c[0], c[1] - d), (c[0] - d, c[1])
    c_tl, c_tr, c_br, c_bl = (0, 1), (1, 1), (1, 0), (0, 0)
    internal_lines = [
        (c_tl, p_tl), (c_tr, p_tr), (c_br, p_br), (c_bl, p_bl),
        (p_tl, v_t), (p_tl, v_l), (p_tr, v_t), (p_tr, v_r),
        (p_br, v_b), (p_br, v_r), (p_bl, v_b), (p_bl, v_l),
    ]
    edge_bars = [(c_bl, c_br), (c_br, c_tr), (c_tr, c_tl), (c_tl, c_bl)]
    return internal_lines + edge_bars


def get_lines_stiff_low():
    """低刚度 (Level 3): 带有较大的中心星形 + 边缘框架"""
    p, d = 0.35, 0.25
    c = (0.5, 0.5)
    p_tl, p_tr, p_br, p_bl = (c[0] - p, c[1] + p), (c[0] + p, c[1] + p), (c[0] + p, c[1] - p), (c[0] - p, c[1] - p)
    v_t, v_r, v_b, v_l = (c[0], c[1] + d), (c[0] + d, c[1]), (c[0], c[1] - d), (c[0] - d, c[1])
    c_tl, c_tr, c_br, c_bl = (0, 1), (1, 1), (1, 0), (0, 0)
    internal_lines = [
        (c_tl, p_tl), (c_tr, p_tr), (c_br, p_br), (c_bl, p_bl),
        (p_tl, v_t), (p_tl, v_l), (p_tr, v_t), (p_tr, v_r),
        (p_br, v_b), (p_br, v_r), (p_bl, v_b), (p_bl, v_l),
    ]
    edge_bars = [(c_bl, c_br), (c_br, c_tr), (c_tr, c_tl), (c_tl, c_bl)]
    return internal_lines + edge_bars


GET_LINES_MAP = {
    "Stiff-High": get_lines_stiff_high,
    "Stiff-Medium": get_lines_stiff_medium,
    "Stiff-Low": get_lines_stiff_low,
}


def get_lines_function(name):
    base_name = name.split('_')[0]
    return GET_LINES_MAP.get(base_name)


# --------------------------------------------------------------------------
# Part 1: WFC Core (PROTOTYPES FROM STIFFNESS SCRIPT)
# --------------------------------------------------------------------------
SOCKET_GRADIENT = ('gradient_interface',)


class Prototype:
    def __init__(self, name, rotation, sockets, neighbour_list, stiffness_level):
        self.name = name
        self.rotation = rotation
        self.sockets = sockets
        self.neighbour_list = neighbour_list
        self.stiffness_level = stiffness_level

    def __repr__(self):
        return f"{self.name} (S:{self.stiffness_level})"


def get_opposite_direction(d):
    return {"posX": "negX", "negX": "posX", "posY": "negY", "negY": "posY"}.get(d)


def rotate_sockets(s, a):
    if a == 0: return s
    if a == 90: return {"posX": s["posY"], "negX": s["negY"], "posY": s["negX"], "negY": s["posX"]}
    return s


def are_sockets_compatible(s1, s2):
    return s1 == s2


def create_prototypes():
    prototypes = []
    type_definitions = {
        "Stiff-High": {"sockets": {"posX": SOCKET_GRADIENT, "negX": SOCKET_GRADIENT, "posY": SOCKET_GRADIENT,
                                   "negY": SOCKET_GRADIENT}, "rotations": [0], "stiffness_level": 1},
        "Stiff-Medium": {"sockets": {"posX": SOCKET_GRADIENT, "negX": SOCKET_GRADIENT, "posY": SOCKET_GRADIENT,
                                     "negY": SOCKET_GRADIENT}, "rotations": [0], "stiffness_level": 2},
        "Stiff-Low": {"sockets": {"posX": SOCKET_GRADIENT, "negX": SOCKET_GRADIENT, "posY": SOCKET_GRADIENT,
                                  "negY": SOCKET_GRADIENT}, "rotations": [0], "stiffness_level": 3},
    }
    all_states = []
    for name, data in type_definitions.items():
        for rot in data["rotations"]:
            all_states.append(
                {"name": f"{name}_{rot}", "rotation": rot, "sockets": rotate_sockets(data["sockets"], rot),
                 "stiffness_level": data["stiffness_level"]})
    for state_data in all_states:
        neighbour_list = {}
        current_stiffness = state_data["stiffness_level"]
        for direction, socket_type in state_data["sockets"].items():
            compatible_sockets = []
            opposite_dir = get_opposite_direction(direction)
            for other_state_data in all_states:
                if not are_sockets_compatible(socket_type, other_state_data["sockets"][opposite_dir]): continue
                other_stiffness = other_state_data["stiffness_level"]
                if abs(current_stiffness - other_stiffness) <= 1:
                    compatible_sockets.append(other_state_data["name"])
            neighbour_list[direction] = compatible_sockets
        prototypes.append(
            Prototype(name=state_data["name"], rotation=state_data["rotation"], sockets=state_data["sockets"],
                      neighbour_list=neighbour_list, stiffness_level=state_data["stiffness_level"]))
    return prototypes


# --------------------------------------------------------------------------
# Part 2: Gaussian Field Logic (FROM GAUSSIAN SCRIPT)
# --------------------------------------------------------------------------
class GaussianSplat2D:
    def __init__(self, color="", mu=np.array([0., 0.]), scale=np.array([1., 1.]), angle_deg=0., opacity=1.):
        self.color = color
        self.mu = np.array(mu, dtype=float)
        self.scale = np.array(scale, dtype=float)
        self.angle_rad = np.deg2rad(angle_deg)
        self.opacity = float(opacity)
        self.rotation_matrix = self._angle_to_rotation_matrix(self.angle_rad)
        self.cov = self._compute_covariance_matrix(self.rotation_matrix, self.scale)
        self.evaluate = self._create_evaluate_closure()

    def _angle_to_rotation_matrix(self, a):
        c, s = np.cos(a), np.sin(a)
        return np.array([[c, -s], [s, c]])

    def _compute_covariance_matrix(self, R, s):
        S = np.diag(s ** 2)
        return R @ S @ R.T

    def _create_evaluate_closure(self):
        mu, op, cov = self.mu, self.opacity, self.cov
        if np.linalg.det(cov) < 1e-10: return lambda p: np.zeros(np.asarray(p).shape[:-1])
        inv_cov = np.linalg.inv(cov)

        def evaluate(p):
            p = np.asarray(p)
            s = p.shape[:-1]
            pf = p.reshape(-1, 2)
            d = pf - mu
            e = np.sum(np.einsum("ij,jk->ik", d, inv_cov) * d, axis=1)
            g = np.exp(-.5 * e)
            return g.reshape(s) * op

        return evaluate


class GaussianField:
    def __init__(self):
        self.splats_by_color = {}

    def add_splat(self, s):
        self.splats_by_color.setdefault(s.color, []).append(s)

    def at(self, p):
        p = np.asarray(p)
        r = {}
        for c, sp in self.splats_by_color.items():
            if sp:
                r[c] = np.sum([s.evaluate(p) for s in sp], axis=0)
        return r


# --------------------------------------------------------------------------
# Part 3: WFC Algorithm (MODIFIED FOR GAUSSIAN GUIDANCE)
# --------------------------------------------------------------------------
def entropy(c): return len(c)


def calculate_state_preference_gaussian(r, c, state_name, gaussian_field, preference_maps):
    if not gaussian_field: return 1.0
    base_name = state_name.split('_')[0]
    preference_map = preference_maps.get(base_name, None)
    return (preference_map[r, c] if preference_map is not None else 0.0) + 0.01


def get_lowest_entropy_cell(grid, collapsed_coords, gaussian_field=None, preference_maps=None):
    collapsed_set = {(r, c) for r, c, _ in collapsed_coords}
    min_entropy, candidates = float('inf'), []
    for r in range(len(grid)):
        for c_ in range(len(grid[0])):
            if (r, c_) in collapsed_set or not isinstance(grid[r][c_], list): continue
            cell_entropy = entropy(grid[r][c_])
            if cell_entropy <= 1: continue
            if cell_entropy < min_entropy:
                min_entropy, candidates = cell_entropy, [(r, c_)]
            elif cell_entropy == min_entropy:
                candidates.append((r, c_))
    if not candidates: return None
    if not gaussian_field: return random.choice(candidates)
    best_cells, max_overall_preference = [], -1.0
    for r_cand, c_cand in candidates:
        cell_max_pref = 0.0
        if not grid[r_cand][c_cand]: continue
        for state_name in grid[r_cand][c_cand]:
            pref = calculate_state_preference_gaussian(r_cand, c_cand, state_name, gaussian_field, preference_maps)
            cell_max_pref = max(cell_max_pref, pref)
        if cell_max_pref > max_overall_preference:
            max_overall_preference, best_cells = cell_max_pref, [(r_cand, c_cand)]
        elif abs(cell_max_pref - max_overall_preference) < 1e-9:
            best_cells.append((r_cand, c_cand))
    return random.choice(best_cells) if best_cells else random.choice(candidates)


def collapse_cell(grid, cell, collapsed_order, gaussian_field=None, preference_maps=None):
    r, c = cell
    possibilities = grid[r][c]
    if not possibilities: raise ValueError(f"Cell ({r},{c}) has no possible states.")
    if len(possibilities) == 1:
        chosen_state = possibilities[0]
    elif not gaussian_field:
        chosen_state = random.choice(possibilities)
    else:
        weights = [calculate_state_preference_gaussian(r, c, s, gaussian_field, preference_maps) for s in possibilities]
        chosen_state = random.choices(possibilities, weights=weights, k=1)[0] if sum(weights) > 0 else random.choice(
            possibilities)
    grid[r][c] = [chosen_state]
    collapsed_order.append((r, c, chosen_state))
    return chosen_state


def propagate(grid, start_cell, prototype_library):
    stack = [start_cell]
    while stack:
        (r, c) = stack.pop(0)
        current_states = grid[r][c]
        for direction, (dr, dc) in {"posX": (0, 1), "negX": (0, -1), "posY": (-1, 0), "negY": (1, 0)}.items():
            nr, nc = r + dr, c + dc
            if not (0 <= nr < len(grid) and 0 <= nc < len(grid[0])): continue
            neighbor_possibilities = grid[nr][nc]
            if not isinstance(neighbor_possibilities, list) or len(neighbor_possibilities) <= 1: continue
            valid_neighbor_states = set()
            for state_name in current_states:
                prototype = prototype_library.get(state_name)
                if prototype: valid_neighbor_states.update(prototype.neighbour_list[direction])
            if len(set(neighbor_possibilities) - valid_neighbor_states) > 0:
                grid[nr][nc] = [s for s in neighbor_possibilities if s in valid_neighbor_states]
                if not grid[nr][nc]: raise ValueError(f"Contradiction propagated to cell ({nr},{nc}) from ({r},{c})")
                if (nr, nc) not in stack: stack.append((nr, nc))


def wfc_algorithm(grid, collapsed_order, prototype_library, gaussian_field=None):
    rows, cols = len(grid), len(grid[0])
    total_cells = rows * cols
    preference_maps = None
    if gaussian_field:
        print("Pre-computing Gaussian preferences...")
        r_coords, c_coords = np.mgrid[0:rows, 0:cols]
        positions = np.stack((c_coords, r_coords), axis=-1)
        preference_maps = gaussian_field.at(positions)
        print("Pre-computation finished.")
    while len(collapsed_order) < total_cells:
        cell_to_collapse = get_lowest_entropy_cell(grid, collapsed_order, gaussian_field, preference_maps)
        if cell_to_collapse is None: break
        try:
            collapse_cell(grid, cell_to_collapse, collapsed_order, gaussian_field, preference_maps)
            propagate(grid, cell_to_collapse, prototype_library)
        except ValueError as e:
            print(f"WFC failed: {e}")
            return grid
        S = len(collapsed_order)
        if S % 20 == 0 or S == total_cells: print(f"Step: {S}, Collapsed cells: {S}/{total_cells}")
    print("\n WFC completed.")
    return grid


def visualize_grid_simple(grid):
    name_map = {"Stiff-High": "High", "Stiff-Medium": "Med ", "Stiff-Low": "Low "}
    for row in grid:
        print(" ".join(
            [f"{name_map.get(c[0].split('_')[0], c[0][:4]): >4}" if len(c) == 1 else f"{len(c): >4}" for c in row]))


# --------------------------------------------------------------------------
# Part 4: 2.5D STL Generation (MODIFIED FOR ASYMMETRIC CLAMPS)
# --------------------------------------------------------------------------
def rotate_point(point, angle_deg, center=(0.5, 0.5)):
    angle_rad = np.radians(angle_deg)
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    x, y = point[0] - center[0], point[1] - center[1]
    return (x * c - y * s + center[0], x * s + y * c + center[1])


def generate_2_5d_stl_from_wfc_result(
        grid: List[List[str]],
        prototype_library: Dict[str, Prototype],
        filename: str,
        stiffness_to_radius_map: Dict[int, float],
        thickness: float = 2.0,
        add_dog_bone_clamps: bool = True,
        clamp_length: float = 8.0,
        transition_length: float = 8.0,
        clamp_width_factor: float = 1.2,
        fillet_radius: float = 3.0
):
    if not trimesh or not shapely:
        print("Trimesh或Shapely库不可用，无法生成STL文件。")
        return

    print("\n开始生成2.5D STL文件...")
    rows, cols = len(grid), len(grid[0])
    all_cell_polygons = []

    print("  Step 1/5: 为每个单元格生成独立的2D多边形...")
    for r in range(rows):
        for c in range(cols):
            if not (isinstance(grid[r][c], list) and len(grid[r][c]) == 1):
                all_cell_polygons.append(None)
                continue

            state_name = grid[r][c][0]
            prototype = prototype_library[state_name]
            stiffness_level = prototype.stiffness_level
            lines_func = get_lines_function(state_name)
            if not lines_func:
                all_cell_polygons.append(None)
                continue

            lines = []
            for p1_local, p2_local in lines_func():
                p1_rot = rotate_point(p1_local, -prototype.rotation)
                p2_rot = rotate_point(p2_local, -prototype.rotation)
                p1_global, p2_global = (p1_rot[0] + c, p1_rot[1] + r), (p2_rot[0] + c, p2_rot[1] + r)
                lines.append((p1_global, p2_global))

            radius = stiffness_to_radius_map[stiffness_level]
            multiline = shapely.geometry.MultiLineString(lines)
            buffered_polygon = multiline.buffer(radius, cap_style='round', join_style='round')
            all_cell_polygons.append(buffered_polygon)

    print("  Step 2/5: 合并所有单元格以形成核心拓扑结构...")
    valid_polygons = [p for p in all_cell_polygons if p is not None and not p.is_empty]
    if not valid_polygons:
        print("警告：没有找到任何几何体来生成。")
        return
    final_polygon_2d = unary_union(valid_polygons)

    all_clamp_polygons = []
    if add_dog_bone_clamps:
        print(f"  Step 3/5: 为左右两侧独立生成对齐的夹持臂...")

        # -- 计算左侧夹持头 --
        left_column_polygons = [all_cell_polygons[r * cols] for r in range(rows) if all_cell_polygons[r * cols]]
        if left_column_polygons:
            left_boundary_union = unary_union(left_column_polygons)
            _, min_y_left, _, max_y_left = left_boundary_union.bounds

            actual_height_left = max_y_left - min_y_left
            clamp_width_left = actual_height_left * clamp_width_factor
            y_center_left = (max_y_left + min_y_left) / 2.0
            y_top_clamp_left = y_center_left + clamp_width_left / 2.0
            y_bottom_clamp_left = y_center_left - clamp_width_left / 2.0

            left_grip = shapely.geometry.box(-clamp_length - transition_length, y_bottom_clamp_left, -transition_length,
                                             y_top_clamp_left)
            left_transition = shapely.geometry.Polygon([
                (0, max_y_left), (0, min_y_left),
                (-transition_length, y_bottom_clamp_left), (-transition_length, y_top_clamp_left)
            ])
            left_clamp_sharp = unary_union([left_grip, left_transition])
            all_clamp_polygons.append(left_clamp_sharp)

        # -- 计算右侧夹持头 --
        right_column_polygons = [all_cell_polygons[r * cols + (cols - 1)] for r in range(rows) if
                                 all_cell_polygons[r * cols + (cols - 1)]]
        if right_column_polygons:
            right_boundary_union = unary_union(right_column_polygons)
            _, min_y_right, _, max_y_right = right_boundary_union.bounds

            actual_height_right = max_y_right - min_y_right
            clamp_width_right = actual_height_right * clamp_width_factor
            y_center_right = (max_y_right + min_y_right) / 2.0
            y_top_clamp_right = y_center_right + clamp_width_right / 2.0
            y_bottom_clamp_right = y_center_right - clamp_width_right / 2.0

            right_transition = shapely.geometry.Polygon([
                (cols, min_y_right), (cols, max_y_right),
                (cols + transition_length, y_top_clamp_right), (cols + transition_length, y_bottom_clamp_right)
            ])
            right_grip_correct = shapely.geometry.box(
                cols + transition_length, y_bottom_clamp_right,
                cols + transition_length + clamp_length, y_top_clamp_right
            )
            right_clamp_sharp = unary_union([right_grip_correct, right_transition])
            all_clamp_polygons.append(right_clamp_sharp)

        if all_clamp_polygons:
            clamps_union = unary_union(all_clamp_polygons)
            if fillet_radius > 0:
                clamps_filleted = clamps_union.buffer(fillet_radius, join_style='round').buffer(-fillet_radius,
                                                                                                join_style='round')
            else:
                clamps_filleted = clamps_union
            final_polygon_2d = unary_union([final_polygon_2d, clamps_filleted])

    print(f"  Step 4/5: 将合并后的2D多边形拉伸为厚度为 {thickness} 的3D网格...")
    mesh_3d = trimesh.creation.extrude_polygon(final_polygon_2d, height=thickness)
    print(f"  Step 5/5: 将最终网格导出到 {filename}...")
    mesh_3d.export(file_obj=filename)
    print(f"2.5D STL文件 '{filename}' 生成完毕！")


# --------------------------------------------------------------------------
# Part 5: Main Execution Logic
# --------------------------------------------------------------------------
def load_gaussian_field_from_json(json_path, grid_size, color_map):
    print(f"Loading Gaussian field from {json_path}...")
    field = GaussianField()
    grid_rows, grid_cols = grid_size
    with open(json_path, 'r') as f:
        data = json.load(f)
    gaussians_data = data.get("field", {}).get("gaussians", [])
    if not gaussians_data: return field
    for i, g_data in enumerate(gaussians_data):
        json_color = g_data.get("color")
        if json_color not in color_map: continue
        prototype_base_name = color_map[json_color]
        mu_norm = g_data.get("mu", [0.5, 0.5])
        scale_norm = g_data.get("scale", [0.1, 0.1])
        angle_rad = g_data.get("angle", 0.0)
        opacity = g_data.get("opacity", 1.0)
        mu_grid = [mu_norm[0] * (grid_cols - 1), (1 - mu_norm[1]) * (grid_rows - 1)]
        scale_grid = [scale_norm[0] * grid_cols, scale_norm[1] * grid_rows]
        angle_deg = -np.rad2deg(angle_rad)
        splat = GaussianSplat2D(color=prototype_base_name, mu=mu_grid, scale=scale_grid, angle_deg=angle_deg,
                                opacity=opacity * 100.0)
        field.add_splat(splat)
        print(f"  Loaded Gaussian {i}: color='{json_color}' -> prototype='{prototype_base_name}'")
    return field


def main():
    start_time = time.time()

    print("1. Creating stiffness-based prototypes...")
    prototypes = create_prototypes()
    prototype_library = {proto.name: proto for proto in prototypes}
    print(f"   Generated {len(prototypes)} unique prototype states: {list(prototype_library.keys())}")

    json_file_path = "stiffness_experiments/2_1_interlocking_gradient.json"

    if not os.path.exists(json_file_path):
        print(f"错误: 高斯场定义文件未在 '{json_file_path}' 找到。")
        print("请先运行 '11_10_generate_experimental_fields.py' 脚本。")
        return

    with open(json_file_path, 'r') as f:
        data = json.load(f)
    source_shape = data.get("source_image_shape")

    GRID_BASE_RESOLUTION = 8

    if source_shape:
        w, h = source_shape
        aspect_ratio = w / h
        grid_cols = int(GRID_BASE_RESOLUTION * aspect_ratio) if aspect_ratio >= 1 else GRID_BASE_RESOLUTION
        grid_rows = GRID_BASE_RESOLUTION if aspect_ratio >= 1 else int(GRID_BASE_RESOLUTION / aspect_ratio)
        print(f"从JSON加载的图像尺寸: {w}x{h} (宽高比: {aspect_ratio:.2f})")
    else:
        print("警告: JSON文件中未找到 'source_image_shape'。将回退到40x40的方形网格。")
        grid_rows, grid_cols = 40, 40

    grid_size = (grid_rows, grid_cols)

    color_map = {"red": "Stiff-High", "green": "Stiff-Medium", "blue": "Stiff-Low"}

    print("\n2. Loading Gaussian field...")
    gaussian_field = load_gaussian_field_from_json(json_file_path, grid_size, color_map)

    print(f"\n3. Initializing {grid_size[0]}x{grid_size[1]} grid and starting WFC...")
    grid = [[list(prototype_library.keys()) for _ in range(grid_size[1])] for _ in range(grid_size[0])]
    collapse_order = []
    final_grid = wfc_algorithm(grid, collapse_order, prototype_library, gaussian_field)

    end_time = time.time()
    print(f"\nTotal execution time: {end_time - start_time:.2f} seconds.")

    print("\nFinal Grid State (Simple Text View):")
    visualize_grid_simple(final_grid)

    is_successful = all(isinstance(cell, list) and len(cell) == 1 for row in final_grid for cell in row)
    if is_successful:
        print(f"\nWFC completed successfully.")
        stiffness_radius_map = {1: 0.05, 2: 0.05, 3: 0.05}
        try:
            print("\nGenerating 2.5D structure STL file with filleted dog-bone clamps...")
            base_name = os.path.splitext(os.path.basename(json_file_path))[0]
            output_filename = f"wfc_result_{base_name}_with_asymmetric_clamps.stl"

            specimen_thickness = grid_rows * 0.02

            generate_2_5d_stl_from_wfc_result(
                grid=final_grid,
                prototype_library=prototype_library,
                filename=output_filename,
                stiffness_to_radius_map=stiffness_radius_map,
                thickness=specimen_thickness,
                add_dog_bone_clamps=True,
                clamp_length=8.0,
                transition_length=8.0,
                clamp_width_factor=1.5,
                fillet_radius=3.0
            )
            print(f"Successfully generated {output_filename}")
        except Exception as e:
            print(f"   Error during STL generation: {e}")
    else:
        print("\nWFC failed to find a valid configuration.")


if __name__ == "__main__":
    main()