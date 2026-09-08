# import torch
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
#
# class WFC_RL_Net(nn.Module):
#     """
#     定义用于波函数坍塌的强化学习神经网络结构。
#     """
#     def __init__(self, grid_height, grid_width, num_states, num_angles=6):
#         super(WFC_RL_Net, self).__init__()
#         self.grid_height = grid_height
#         self.grid_width = grid_width
#         self.num_states = num_states
#         self.num_angles = num_angles  # 新增，方向数
#
#         # 卷积神经网络用于处理 grid_constraints
#         self.cnn = nn.Sequential(
#             nn.Conv2d(in_channels=num_states, out_channels=32, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(32, 64, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.Flatten()
#         )
#
#         # 计算 CNN 输出的特征维度
#         cnn_output_size = 64 * grid_height * grid_width
#
#         # 全连接层用于处理弹性模量特征
#         self.elastic_fc = nn.Sequential(
#             nn.Linear(num_angles, 64),
#             nn.ReLU()
#         )
#
#         # 最后的全连接层，用于融合特征并输出动作概率
#         self.fc = nn.Sequential(
#             nn.Linear(cnn_output_size + 64, 1024),
#             nn.ReLU(),
#             nn.Linear(1024, grid_height * grid_width * num_states)
#         )
#
#     def forward(self, grid_constraints, elastic_modulus):
#         batch_size = grid_constraints.size(0)
#
#         # 处理 grid_constraints，使用 CNN
#         #x_grid = grid_constraints.permute(0, 3, 1, 2)  # 调整维度
#         x_grid = grid_constraints
#         x_grid = self.cnn(x_grid)  # 输出维度为 [batch_size, cnn_output_size]
#
#         # 处理弹性模量特征
#         # elastic_modulus 维度为 [batch_size, num_angles]
#         # 归一化处理
#         elastic_modulus_normalized = elastic_modulus / torch.max(elastic_modulus, dim=1, keepdim=True)[0]
#         x_elastic = self.elastic_fc(elastic_modulus_normalized)
#
#         # 融合特征
#         x = torch.cat((x_grid, x_elastic), dim=1)
#
#         # 前向传播，通过全连接层
#         x = self.fc(x)
#         scaled_x = x * 1000
#         # 输出动作概率
#         action_probs = F.softmax(scaled_x, dim=1)
#         return action_probs
# grid_height, grid_width = 10,10
# num_states = 17
# num_angles=6
# grid_constraints = torch.randn(1, num_states, grid_height, grid_width)
# elastic_features = torch.randn(1, num_angles)
# model = WFC_RL_Net(grid_height, grid_width, num_states)
# # 导出为 ONNX 文件
# torch.onnx.export(
#     model,
#     (grid_constraints, elastic_features),
#     "wfc_rl_net.onnx",
#     input_names=["grid_constraints", "elastic_features"],
#     output_names=["output"],
#     dynamic_axes={"grid_constraints": {0: "batch"}, "elastic_features": {0: "batch"}}
# )
import random
from collections import deque
import Position_constrain
import math
import random
from collections import deque
import numpy as np
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib import patches
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

def get_lowest_entropy_cell(grid, collapse_order, fiber_block_library):
    collapsed_cells = {(i, j) for i, j, _ in collapse_order}


    min_entropy = float('inf')
    min_entropy_cell = None
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            cell_entropy = entropy(grid[i][j])
            if (i, j) in collapsed_cells or cell_entropy == len(fiber_block_library) or cell_entropy == 0:
                continue
            if cell_entropy < min_entropy:
                min_entropy = cell_entropy
                min_entropy_cell = (i, j)
    return min_entropy_cell

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
    step = 0  # 增加一个计数器，用于标记迭代步骤
    while True:
        # 在尝试坍缩之前先可视化当前状态


        cell = get_lowest_entropy_cell(grid, collapse_order, fiber_block_library)
        if cell is None:
            break
        chosen_state = collapse_cell(grid, cell, collapse_order)

        i, j = cell
        neighbors = get_neighbors(i, j, grid)
        for (ni, nj) in neighbors:
            propagate_constraints(grid, ni, nj, fiber_block_library)

        # 传播后可视化
        visualize_grid_states(grid, fiber_block_library, step)
        step += 1



def draw_block(ax, block_type, rotation, x_offset=0, y_offset=0, scale=0.3):
    """
    在ax中绘制特定类型和旋转角度的块，可附加偏移和缩放因子，以便在同一个subplot中绘制多个状态。
    """
    line_color = 'blue'
    line_width = 1
    # 缩放坐标时可以将坐标乘scale再加偏移
    # 这里以(0,0)-(1,1)的单元格为基础，把绘制范围缩小到scale以内，并在x_offset,y_offset处绘制。
    # 比如原始坐标是[0,1], 缩小为scale后变成[0,scale], 然后再+ x_offset, y_offset
    def transform(coords):
        return [(x*scale + x_offset, y*scale + y_offset) for x, y in coords]

    if block_type == "Type1":
        if rotation == 0:
            coords = transform([(0.5,0), (0,0.5)])
        elif rotation == 90:
            coords = transform([(0,0.5), (0.5,1)])
        elif rotation == 180:
            coords = transform([(0.5,1), (1,0.5)])
        elif rotation == 270:
            coords = transform([(1,0.5),(0.5,0)])
        x_vals = [c[0] for c in coords]
        y_vals = [c[1] for c in coords]
        ax.plot(x_vals,y_vals,color=line_color,linewidth=line_width)

    elif block_type == "Type3":
        if rotation == 0:
            # 横线
            coords_h = transform([(0,0.5),(1,0.5)])
            # 竖线
            coords_v = transform([(0.5,0),(0.5,1)])
            ax.plot([c[0] for c in coords_h],[c[1] for c in coords_h],color=line_color,linewidth=line_width)
            ax.plot([c[0] for c in coords_v],[c[1] for c in coords_v],color=line_color,linewidth=line_width)

    elif block_type == "Type4":
        if rotation == 0:
            coords_h = transform([(0,0.5),(1,0.5)])
            ax.plot([c[0] for c in coords_h],[c[1] for c in coords_h],color=line_color,linewidth=line_width)
        elif rotation == 90:
            coords_v = transform([(0.5,0),(0.5,1)])
            ax.plot([c[0] for c in coords_v],[c[1] for c in coords_v],color=line_color,linewidth=line_width)

    elif block_type == "Type6":
        if rotation == 0:
            coords_1 = transform([(0.5,0), (0,0.5)])
            coords_2 = transform([(0.5, 1), (1,0.5)])
            ax.plot([c[0] for c in coords_1],[c[1] for c in coords_1],color=line_color,linewidth=line_width)
            ax.plot([c[0] for c in coords_2],[c[1] for c in coords_2],color=line_color,linewidth=line_width)
import math


def visualize_grid_states(grid, fiber_block_library, step):
    size = len(grid)  # 假设grid是5x5
    fig, axes = plt.subplots(size, size, figsize=(40,40))

    import math

    for i in range(size):
        for j in range(size):
            ax = axes[i, j]
            possible_states = grid[i][j]

            ax.clear()
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlim(0,1)
            ax.set_ylim(0,1)
            ax.add_patch(plt.Rectangle((0,0),1,1,fill=None,edgecolor='black'))

            n = len(possible_states)
            if n > 0:
                n_side = math.ceil(math.sqrt(n))
            else:
                n_side = 1  # 如果没有状态，也设为1，表示整格无状态。

            cell_scale = 1.0 / n_side

            # 绘制状态或X
            total_subcells = n_side * n_side
            for idx in range(total_subcells):
                # 子格坐标
                row_idx = idx // n_side
                col_idx = idx % n_side
                x_offset = col_idx * cell_scale
                y_offset = (n_side - 1 - row_idx) * cell_scale

                if idx < n:
                    # 有状态
                    state = possible_states[idx]
                    block_type, rotation = state.split('_')
                    rotation = int(rotation)
                    if block_type == "Type5":
                        # 用文本"T5"标记type5_0，而不画线条，这样区别无状态的X
                        ax.text(x_offset + cell_scale/2, y_offset + cell_scale/2,
                                "T5", ha='center', va='center', fontsize=8, color='green')
                    else:
                        # 正常绘制该状态
                        draw_block(ax, block_type, rotation, x_offset=x_offset, y_offset=y_offset, scale=cell_scale)
                else:
                    # 没有状态的位置，用X表示
                    ax.text(x_offset + cell_scale/2, y_offset + cell_scale/2,
                            "X", ha='center', va='center', fontsize=8, color='red')

            # 绘制内部的小区域格线
            if n_side > 1:
                for k in range(1, n_side):
                    # 竖线
                    x_line = k * cell_scale
                    ax.plot([x_line,x_line],[0,1], color='gray', linestyle='--', linewidth=0.5)
                    # 横线
                    y_line = k * cell_scale
                    ax.plot([0,1],[y_line,y_line], color='gray', linestyle='--', linewidth=0.5)

    plt.tight_layout()
    plt.suptitle(f"Step {step} - Current possible states")
    plt.show()



def main():
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
    grid_size = (5,5)
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


if __name__ == "__main__":
    main()