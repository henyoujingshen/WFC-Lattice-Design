# Copyright (c) 2026 Tang Hubocheng. All rights reserved.
# Original project code: no use or redistribution without written permission.
# See the repository LICENSE for scope, exceptions and third-party rights.
import random
import time
from collections import deque
import numpy as np
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib import patches
import Homoge  # 假设这是自定义模块
import Position_constrain
import torch
from copy import deepcopy
def set_random_seed(seed=42):
    """
    设置随机种子以保证结果的可重复性。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # 如果使用GPU
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
import visl


class Prototype:
    def __init__(self, mesh, rotation, sockets, neighbour_list,spaced_neighbour_list):
        self.mesh = mesh
        self.rotation = rotation
        self.sockets = sockets
        self.neighbour_list = neighbour_list
        self.spaced_neighbour_list=spaced_neighbour_list

    def __repr__(self):
        return f"{self.mesh}_{self.rotation}"

def get_opposite_direction(direction):
    opposite = {"posX": "negX", "negX": "posX", "posY": "negY", "negY": "posY"}
    return opposite.get(direction)

def create_prototypes():
    prototypes = []

    # 定义每种类型和旋转角度的接口类型
    socket_types = {
        "Type1": {
            0: {"posX": 0, "negX": 1, "posY": 0, "negY": 1},
            90: {"posX": 0, "negX": 1, "posY": 1, "negY": 0},
            180: {"posX": 1, "negX": 0, "posY": 1, "negY": 0},
            270: {"posX": 1, "negX": 0, "posY": 0, "negY": 1}
        },
        "Type3": {
            0: {"posX": 1, "negX": 1, "posY": 1, "negY": 1},
        },
        "Type4": {
            0: {"posX": 1, "negX": 1, "posY": 0, "negY": 0},
            90: {"posX": 0, "negX": 0, "posY": 1, "negY": 1},
        },
        "Type5": {
            0: {"posX": 0, "negX": 0, "posY": 0, "negY": 0}
        }
    }

    for block_type, rotations in socket_types.items():
        for rotation, sockets in rotations.items():
            neighbour_list = {}
            spaced_neighbour_list = {}
            for direction, socket_type in sockets.items():
                compatible_sockets = []
                spaced_compatible_sockets = []
                for other_block_type, other_rotations in socket_types.items():
                    for other_rotation, other_sockets in other_rotations.items():
                        opposite_direction = get_opposite_direction(direction)
                        other_state = f"{other_block_type}_{other_rotation}"

                        # 检查直接邻居的兼容性
                        if socket_type == 0 and other_sockets[opposite_direction] == 0:
                            compatible_sockets.append(other_state)
                        elif socket_type == 1 and other_sockets[opposite_direction] == 1:
                            compatible_sockets.append(other_state)
                        elif socket_type == 1 and other_sockets[opposite_direction] == 1:
                            compatible_sockets.append(other_state)

                        # 检查间隔一个单元格的邻居的兼容性
                        # 如果当前边界为1，那么间隔的邻居的相对边界不能为1
                        if socket_type == 1:
                            if other_sockets[opposite_direction] != 1:
                                spaced_compatible_sockets.append(other_state)
                        # 如果当前边界为1，那么间隔的邻居的相对边界不能为1
                        elif socket_type == 1:
                            if other_sockets[opposite_direction] != 1:
                                spaced_compatible_sockets.append(other_state)
                        # 如果当前边界为0，没有限制
                        else:  # socket_type == 0
                            spaced_compatible_sockets.append(other_state)

                neighbour_list[direction] = compatible_sockets
                spaced_neighbour_list[direction] = spaced_compatible_sockets

            prototype = Prototype(
                mesh=f"{block_type}.obj",
                rotation=rotation,
                sockets=sockets,
                neighbour_list=neighbour_list,
                spaced_neighbour_list=spaced_neighbour_list  # 添加间隔邻居列表
            )
            prototypes.append(prototype)

    return prototypes


def entropy(cell):
    return len(cell)

# def get_lowest_entropy_cell(grid, collapse_order, fiber_block_library):
#     collapsed_cells = {(i, j) for i, j, _ in collapse_order}
#
#     min_entropy = float('inf')
#     candidates = []
#     for i in range(len(grid)):
#         for j in range(len(grid[0])):
#             if (i, j) in collapsed_cells:
#                 continue
#             cell_entropy = entropy(grid[i][j])
#             # 排除已经坍缩的单元格和全状态未排除的单元格
#             if cell_entropy == len(fiber_block_library) or cell_entropy == 0:
#                 continue
#             if cell_entropy < min_entropy:
#                 min_entropy = cell_entropy
#                 candidates = [(i, j)]
#             elif cell_entropy == min_entropy:
#                 candidates.append((i, j))
#     if not candidates:
#         return None
#     return random.choice(candidates)

def get_lowest_entropy_cell(grid, collapse_order, fiber_block_library):
    collapsed_cells = {(i, j) for i, j, _ in collapse_order}

    min_entropy = float('inf')
    candidates = []

    for i in range(len(grid)):
        for j in range(len(grid)):
            if (i, j) in collapsed_cells:
                continue
            cell_entropy = entropy(grid[i][j])
            # 排除已坍缩或熵为初始最大值（全可能状态）的单元格
            if cell_entropy == 0 or cell_entropy == len(fiber_block_library):
                continue

            if cell_entropy < min_entropy:
                min_entropy = cell_entropy
                candidates = [(i, j)]
            elif cell_entropy == min_entropy:
                candidates.append((i, j))

    return random.choice(candidates) if candidates else None

def collapse_cell(grid, cell, order):
    i, j = cell
    possible_states = grid[i][j]
    chosen_state = random.choice(possible_states)
    grid[i][j] = [chosen_state]
    order.append((i, j, chosen_state))
    return chosen_state

def is_state_compatible(state, i, j, grid, fiber_block_library):
    """
    检查给定状态是否与网格中所有相关单元格的状态兼容。

    参数：
        state (str): 当前单元格的状态。
        i (int): 当前单元格的行索引。
        j (int): 当前单元格的列索引。
        grid (list of list of list): 网格，grid[i][j] 是一个包含可能状态的列表。
        fiber_block_library (dict): 状态库，映射状态名称到 Prototype 对象。

    返回：
        bool: 如果状态兼容，返回 True，否则，返回 False。
    """
    proto = fiber_block_library[state]

    # 定义方向与对应邻居边界属性的映射
    direction_boundary_map = {
        "posX": "negX",
        "negX": "posX",
        "posY": "negY",
        "negY": "posY"
    }

    # 检查直接邻居
    for direction, neighbours in proto.neighbour_list.items():
        ni, nj = i, j
        if direction == "posX":
            nj += 1
        elif direction == "negX":
            nj -= 1
        elif direction == "posY":
            ni -= 1
        elif direction == "negY":
            ni += 1
        else:
            # 未知方向，跳过
            continue

        if 0 <= ni < len(grid) and 0 <= nj < len(grid[0]):
            neighbor_states = grid[ni][nj]
            if not neighbor_states:
                # 邻居单元格为空，可能需要处理
                continue

            # 获取邻居单元格相关边界的所有可能值
            boundary_property = direction_boundary_map.get(direction)
            if not boundary_property:
                # 如果没有定义对应的边界属性，跳过
                continue

            # 收集所有可能状态的相关边界值
            boundary_values = set()
            for neighbor_state in neighbor_states:
                neighbor_proto = fiber_block_library[neighbor_state]
                boundary_value = neighbor_proto.sockets[boundary_property]
                if boundary_value is not None:
                    boundary_values.add(boundary_value)

            # 如果所有边界值一致，则进行约束检查
            if len(boundary_values) == 1:
                consistent_boundary = boundary_values.pop()
                # 检查一致的边界值是否在允许的邻居状态中
                neighbor_state = grid[ni][nj][0]
                if neighbor_state not in neighbours:
                    #print(f"direct Constraint failed for direction {direction} at ({i}, {j}) with state {state}")
                    return False

    directions = ["posX", "negX", "posY", "negY"]

    # if i==2 and j==3:
    #     a=122
    # 对每个方向应用约束
    for direction in directions:
        # 获取当前方向相关的单元格位置
        if direction == "posX":
            required_cells = [
                (i, j + 2),  # 右隔一个单元格
                (i - 1, j + 1),  # 右上角单元格
                (i + 1, j + 1)  # 右下角单元格
            ]
        elif direction == "negX":
            required_cells = [
                (i, j - 2),  # 左隔一个单元格
                (i - 1, j - 1),  # 左上角单元格
                (i + 1, j - 1)  # 左下角单元格
            ]
        elif direction == "posY":
            required_cells = [
                (i - 2, j),  # 上隔一个单元格
                (i - 1, j - 1),  # 左上角单元格
                (i - 1, j + 1)  # 右上角单元格
            ]
        elif direction == "negY":
            required_cells = [
                (i + 2, j),  # 下隔一个单元格
                (i + 1, j - 1),  # 左下角单元格
                (i + 1, j + 1)  # 右下角单元格
            ]
        else:
            # 未知方向，跳过
            continue

        # 检查所有相关单元格是否在网格范围内
        all_within_bounds = all(
            0 <= ci < len(grid) and 0 <= cj < len(grid[0])
            for ci, cj in required_cells
        )

        if not all_within_bounds:
            # 如果有任意一个相关单元格超出网格范围，跳过该方向的约束检查
            continue

        # 如果所有相关单元格都在范围内，调用约束检查
        if not Position_constrain.check_direction_constraint(state, i, j, grid, fiber_block_library, direction):
            #print(f"three failed for direction {direction} at ({i}, {j}) with state {state}")
            b=state
            a=Position_constrain.check_direction_constraint(state, i, j, grid, fiber_block_library, direction)
            return False

    return True


def is_state_compatible_with_boundary(state, i, j, grid, fiber_block_library):
    proto = fiber_block_library[state]
    # 可以在这里添加对边界条件的检查逻辑
    return is_state_compatible(state, i, j, grid, fiber_block_library)

def get_neighbors(i, j, grid):
    """返回(i, j)单元格的四个相邻单元格坐标。"""
    neighbors = []
    directions = [("posX", 0, 1), ("negX", 0, -1), ("posY", -1, 0), ("negY", 1, 0)]
    for dir_name, di, dj in directions:
        ni, nj = i + di, j + dj
        if 0 <= ni < len(grid) and 0 <= nj < len(grid[0]):
            neighbors.append((ni, nj))
    return neighbors


def propagate_constraints(grid, start_i, start_j, fiber_block_library):
    queue = deque()
    queue.append((start_i, start_j))

    while queue:
        i, j = queue.popleft()

        possible_states = grid[i][j]
        if not possible_states:
            # 状态为空，发生冲突，根据需要处理
            continue

        # 检查并过滤不兼容的状态
        compatible_states = [
            state for state in possible_states
            if is_state_compatible_with_boundary(state, i, j, grid, fiber_block_library)
        ]

        if len(compatible_states) < len(possible_states):
            old_states = grid[i][j]
            grid[i][j] = compatible_states

            if not compatible_states:
                # 冲突出现，根据需要处理
                continue

            # 状态集合有变化才传播给邻居
            neighbors = get_neighbors(i, j, grid)
            for (ni, nj) in neighbors:
                queue.append((ni, nj))

def wfc_algorithm(grid, collapse_order, fiber_block_library):
    #print("[Inside] Initial grid address:", id(grid))  # 打印内存地址
    STEP=0
    while True:
        cell = get_lowest_entropy_cell(grid, collapse_order, fiber_block_library)
        if cell is None:
            # 所有格子都已经完成坍缩或无法再选择
            break

        chosen_state = collapse_cell(grid, cell, collapse_order)

        if len(collapse_order)==88:
            print("half")

        # 不从刚坍缩的单元格开始，而是从其邻居开始传播
        i, j = cell
        neighbors = get_neighbors(i, j, grid)
        for (ni, nj) in neighbors:
            visited = set()
            visited.add((i, j))
            propagate_constraints(grid, ni, nj, fiber_block_library)
        STEP += 1
        #visl.visualize_grid_states(grid, fiber_block_library, STEP)
    #print("[Inside] Final grid address:", id(grid))
    new_grid = deepcopy(grid)

    return new_grid

def draw_block(ax, block_type, rotation):
    """
    绘制特定类型和旋转角度的块，用于可视化。
    将原先的箭头改为等长直线。
    """

    # 为了与原来的颜色和风格保持一致，统一使用蓝色线条
    line_color = 'blue'
    line_width = 2

    if block_type == "Type1":
        if rotation == 0:
            # 原：ax.arrow(0.5, 0, -0.4, 0.5, ...)
            ax.plot([0.5, 0.5 - 0.5], [0, 0.5], color=line_color, linewidth=line_width)
        elif rotation == 90:
            # 原：ax.arrow(0, 0.5, 0.5, 0.4, ...)
            ax.plot([0, 0 + 0.5], [0.5, 0.5 + 0.5], color=line_color, linewidth=line_width)
        elif rotation == 180:
            # 原：ax.arrow(0.5, 1, 0.4, -0.5, ...)
            ax.plot([0.5, 0.5 + 0.5], [1, 1 - 0.5], color=line_color, linewidth=line_width)
        elif rotation == 270:
            # 原：ax.arrow(1, 0.5, -0.5, -0.4, ...)
            ax.plot([1, 1 - 0.5], [0.5, 0.5 - 0.5], color=line_color, linewidth=line_width)


    elif block_type == "Type3":
        if rotation == 0:
            # 原：ax.arrow(0, 0.5, 0.9, 0, ...)
            ax.plot([0, 0+1], [0.5, 0.5], color=line_color, linewidth=line_width)
            # 原：ax.arrow(0.5, 1, 0, -0.9, ...)
            ax.plot([0.5, 0.5], [1, 1-1], color=line_color, linewidth=line_width)

    elif block_type == "Type4":
        if rotation == 0:
            # 原：ax.arrow(0, 0.5, 0.9, 0, ...)
            ax.plot([0, 0+1], [0.5, 0.5], color=line_color, linewidth=line_width)
        elif rotation == 90:
            # 原：ax.arrow(0.5, 1, 0, -0.9, ...)
            ax.plot([0.5, 0.5], [1, 1-1], color=line_color, linewidth=line_width)



def visualize_all_types():
    """
    可视化所有类型和旋转角度的块。
    """
    fig, axes = plt.subplots(4, 4, figsize=(6, 6))

    block_types = ["Type1", "Type2", "Type3", "Type4"]
    rotations = [0, 90, 180, 270]

    for i, block_type in enumerate(block_types):
        for j, rotation in enumerate(rotations):
            ax = axes[i, j]
            ax.set_title(f"{block_type}_{rotation}")
            ax.add_patch(plt.Rectangle((0, 0), 1, 1, fill=None, edgecolor='black'))
            draw_block(ax, block_type, rotation)
            ax.set_xticks([])
            ax.set_yticks([])

    plt.tight_layout()
    plt.show()

def convert_to_matrix(block_type, rotation):
    """
    将特定的块类型和旋转角度转换为5x5的数值矩阵。
    0表示没有纤维，数字表示纤维的方向（相对于x轴的角度）。
    """
    matrix = np.ones((5, 5))

    if block_type == "Type1":
        if rotation == 0:
            matrix[2, 0] = 225
            matrix[3, 1] = 225
            matrix[4, 2] = 225
        elif rotation == 90:
            matrix[0, 2] = 315
            matrix[1, 1] = 315
            matrix[2, 0] = 315
        elif rotation == 180:
            matrix[0, 2] = 45
            matrix[1, 3] = 45
            matrix[2, 4] = 45
        elif rotation == 270:
            matrix[2, 4] = 135
            matrix[3, 3] = 135
            matrix[4, 2] = 135
    elif block_type == "Type2":
        if rotation == 0:
            matrix[2, 0] = 45
            matrix[3, 1] = 45
            matrix[4, 2] = 45
        elif rotation == 90:
            matrix[0, 2] = 135
            matrix[1, 1] = 135
            matrix[2, 0] = 135
        elif rotation == 180:
            matrix[2, 4] = 225
            matrix[1, 3] = 225
            matrix[0, 2] = 225
        elif rotation == 270:
            matrix[4, 2] = 315
            matrix[3, 3] = 315
            matrix[2, 4] = 315
    elif block_type == "Type3":
        if rotation == 0:
            matrix[2, :] = 0
            matrix[:, 2] = 90
        elif rotation == 90:
            matrix[:, 2] = 90
            matrix[2, :] = 180
        elif rotation == 180:
            matrix[2, :] = 180
            matrix[:, 2] = 270
        elif rotation == 270:
            matrix[:, 2] = 270
            matrix[2, :] = 0
    elif block_type == "Type4":
        if rotation == 0:
            matrix[2, :] = 0
        elif rotation == 90:
            matrix[:, 2] = 90
        elif rotation == 180:
            matrix[2, :] = 180
        elif rotation == 270:
            matrix[:, 2] = 270
    return matrix



def generate_fea_matrix(grid):
    """
    将任意 HxW 的块网格转换为 (H*5)x(W*5) 的有限元分析矩阵。
    每个单元格固定扩展为 5x5 的子矩阵。
    """
    # --- 主要改动点 1: 动态获取输入 grid 的尺寸 ---
    if not grid or not grid[0]:
        return np.array([])  # 处理空 grid 的情况
    H = len(grid)
    W = len(grid[0])

    # 将 grid 内容转换为字符串格式，这部分逻辑保持不变
    grid = [[str(cell[0]) if isinstance(cell, list) else str(cell) for cell in row] for row in grid]

    # --- 主要改动点 2: 根据动态尺寸创建 fea_matrix ---
    fea_matrix = np.zeros((H * 5, W * 5))

    # --- 主要改动点 3: 使用动态尺寸进行循环 ---
    for i in range(H):
        for j in range(W):
            block_str = grid[i][j]
            block_type, rotation = block_str.split('_')
            rotation = int(rotation)

            # convert_to_matrix 函数返回一个 5x5 的块
            block_matrix = convert_to_matrix(block_type, rotation)

            # 将 5x5 的块填充到 fea_matrix 的正确位置，这部分逻辑无需改动
            fea_matrix[i * 5:(i + 1) * 5, j * 5:(j + 1) * 5] = block_matrix

    return fea_matrix

def perform_FEA(grid):
    start = time.time()
    """
    基于完全坍缩的网格执行有限元分析。
    返回K_prime矩阵。
    """
    new_grid = []
    for row in grid:
        new_row = []
        for cell in row:
            if isinstance(cell, list):
                if not cell:  # 检查是否为空列表
                    break  # 中断当前操作
                new_row.append(str(cell[0]))
            else:
                new_row.append(str(cell))
        else:
            new_grid.append(new_row)
            continue  # 如果内层循环没有触发 break，继续外层循环
        break  # 如果内层循环触发 break，则中断外层循环
    fea_matrix = generate_fea_matrix(grid)

    # 将矩阵保存为文本文件 fiber_angles.txt
    #np.savetxt('fiber_angles.txt', fea_matrix, fmt='%d')
    # fea_matrix = np.zeros((50, 50))
    K_prime = Homoge.s(fea_matrix)

    # ...原有代码...
    #print(f"单次计算耗时: {time.time() - start:.4f}s")
    return K_prime


def generate_K_library(grid_size, fiber_block_library, num_samples=1):
    """
    生成K值库，用于有限元分析。
    """

    K_library = []
    for _ in range(num_samples):
        grid = [[list(fiber_block_library.keys()) for _ in range(grid_size[1])] for _ in range(grid_size[0])]
        collapse_order = []

        initial_state = "Type4_90"
        grid[0][0] = [initial_state]
        collapse_order.append((0, 0, initial_state))

        neighbors = get_neighbors(0, 0, grid)
        for (ni, nj) in neighbors:
            propagate_constraints(grid, ni, nj, fiber_block_library)

        #测试用，用后注释掉
        grid = [[["Type4_90"] for _ in range(5)] for _ in range(5)]


        # wfc_algorithm(grid, collapse_order, fiber_block_library)
        #
        # grid = [[str(cell[0]) if isinstance(cell, list) else str(cell) for cell in row] for row in grid]

        K = perform_FEA(grid)
        K_library.append(K)
    return K_library

def generate_random_K(K_library):
    """
    从K值库中随机选择一个K值。
    """
    return random.choice(K_library)

def visualize_grid(grid, collapse_order):
    """
    可视化最终的网格结果。
    """
    grid = [[str(cell[0]) if isinstance(cell, list) else str(cell) for cell in row] for row in grid]

    norm = mcolors.Normalize(vmin=0, vmax=len(collapse_order))
    cmap = plt.cm.get_cmap('coolwarm')

    collapse_dict = {(x, y): idx for idx, (x, y, _) in enumerate(collapse_order)}

    fig, axes = plt.subplots(5, 5, figsize=(5, 5))

    for i in range(5):
        for j in range(5):
            cell = grid[i][j]
            ax = axes[i, j]

            if (i, j) in collapse_dict:
                idx = collapse_dict[(i, j)]
                color = cmap(norm(idx))
                ax.add_patch(patches.Rectangle((0, 0), 1, 1, color=color, zorder=1))

            if cell == 'Type5':
                ax.axis('off')
            else:
                block_type, rotation = cell.split('_')
                draw_block(ax, block_type, int(rotation))

            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')

    plt.subplots_adjust(wspace=-2, hspace=-2)
    plt.show()

def main():
    # 生成原型
    #set_random_seed()
    prototypes = create_prototypes()
    # Display the results
    for proto in prototypes:
        print(proto)

        print("Sockets:", proto.sockets)
        print("Neighbour List:", proto.neighbour_list)
        print()

    # 构建fiber_block_library
    fiber_block_library = {f"{proto.mesh[:-4]}_{proto.rotation}": proto for proto in prototypes}

    # 初始化网格
    grid_size = (15, 15)
    grid = [[list(fiber_block_library.keys()) for _ in range(grid_size[1])] for _ in range(grid_size[0])]
    collapse_order = []
    # 初始化左上角的状态
    initial_state = "Type4_90"
    grid[0][0] = [initial_state]
    collapse_order.append((0, 0, initial_state))

    # 不再直接从 (0,0) 调用 propagate_constraints
    # 而是从 (0,0) 的邻居开始传播约束
    neighbors = get_neighbors(0, 0, grid)
    for (ni, nj) in neighbors:
        propagate_constraints(grid, ni, nj, fiber_block_library)

    # 执行波函数坍缩算法
    wfc_algorithm(grid, collapse_order, fiber_block_library)
    visl.visualize_grid_states(grid, fiber_block_library, 25)
    # 可视化所有类型的块
    visualize_all_types()
    grid = [[["Type3_0"] for _ in range(5)] for _ in range(5)]
    # 可视化最终的网格

    # 生成并存储K值库
    num_samples = 2
    K_library = generate_K_library(grid_size, fiber_block_library, num_samples)

    # 打印K值库信息
    print(f"Generated {len(K_library)} K values in the library.")
    for idx, K in enumerate(K_library[:5]):
        print(f"K-{idx + 1}:")
        print(K)

    # 随机选择一个K值作为目标
    target_K = generate_random_K(K_library)
    print("Randomly selected target K:")
    print(target_K)

if __name__ == "__main__":
    main()