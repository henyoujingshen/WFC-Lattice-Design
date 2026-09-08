# Copyright (c) 2026 Tang Hubocheng. All rights reserved.
# Original project code: no use or redistribution without written permission.
# See the repository LICENSE for scope, exceptions and third-party rights.
# import numpy as np
# import torch
# import scipy.sparse as sp  # 保留用于原始函数
# import global_stiffness
#
#
#
# def topology_matrices_torch(nodes: torch.Tensor):
#     """
#     在GPU上使用PyTorch计算拓扑矩阵 B_0, B_a, B_eps, 和 V。
#     输入和输出都是GPU张量。
#
#     Args:
#         nodes (torch.Tensor): 节点坐标张量，形状为 (total_nodes, 2)。
#     """
#     device = nodes.device
#     dtype = nodes.dtype
#
#     min_x, max_x = torch.min(nodes[:, 0]), torch.max(nodes[:, 0])
#     min_y, max_y = torch.min(nodes[:, 1]), torch.max(nodes[:, 1])
#     lx, ly = max_x - min_x, max_y - min_y
#
#     total_nodes = nodes.shape[0]
#     tol = 1e-8
#
#     # 使用布尔掩码高效地识别边界节点
#     left_mask = torch.isclose(nodes[:, 0], min_x, atol=tol)
#     right_mask = torch.isclose(nodes[:, 0], max_x, atol=tol)
#     bottom_mask = torch.isclose(nodes[:, 1], min_y, atol=tol)
#     top_mask = torch.isclose(nodes[:, 1], max_y, atol=tol)
#
#     left_master_nodes = torch.where(left_mask)[0]
#     right_slave_nodes = torch.where(right_mask)[0]
#     bottom_master_nodes = torch.where(bottom_mask)[0]
#     top_slave_nodes = torch.where(top_mask)[0]
#
#     # 在GPU上直接排序
#     left_master_nodes = left_master_nodes[torch.argsort(nodes[left_master_nodes, 1])]
#     right_slave_nodes = right_slave_nodes[torch.argsort(nodes[right_slave_nodes, 1])]
#     bottom_master_nodes = bottom_master_nodes[torch.argsort(nodes[bottom_master_nodes, 0])]
#     top_slave_nodes = top_slave_nodes[torch.argsort(nodes[top_slave_nodes, 0])]
#
#     # 合并主节点和从节点
#     master_nodes = torch.unique(torch.cat((left_master_nodes, bottom_master_nodes)))
#     slave_nodes = torch.unique(torch.cat((right_slave_nodes, top_slave_nodes)))
#
#     # 内部节点 (注意: PyTorch没有setdiff1d，这里用一种高效的替代方法)
#     all_nodes = torch.arange(total_nodes, device=device)
#     boundary_nodes = torch.cat((master_nodes, slave_nodes))
#     is_boundary = torch.zeros(total_nodes, dtype=torch.bool, device=device)
#     is_boundary[boundary_nodes] = True
#     interior_nodes = all_nodes[~is_boundary]
#
#     # 构建 B_0 矩阵
#     independent_nodes = torch.cat((master_nodes, interior_nodes))
#     num_independent = independent_nodes.size(0)
#
#     B_0 = torch.zeros((total_nodes, num_independent), dtype=dtype, device=device)
#
#     # 使用 advanced indexing 填充
#     B_0[independent_nodes, torch.arange(num_independent, device=device)] = 1
#
#     # 使用 torch.searchsorted 来高效地找到映射关系，避免字典和循环
#     # 首先需要对 independent_nodes 进行排序
#     sorted_independent_nodes, sort_indices = torch.sort(independent_nodes)
#
#     # 找到左侧主节点在已排序的独立节点列表中的索引
#     left_node_indices_in_sorted = torch.searchsorted(sorted_independent_nodes, left_master_nodes)
#     # 映射回未排序的独立节点列表中的原始索引
#     left_node_indices = sort_indices[left_node_indices_in_sorted]
#     B_0[right_slave_nodes, left_node_indices] = 1
#
#     # 对底部和顶部节点执行相同操作
#     bottom_node_indices_in_sorted = torch.searchsorted(sorted_independent_nodes, bottom_master_nodes)
#     bottom_node_indices = sort_indices[bottom_node_indices_in_sorted]
#     B_0[top_slave_nodes, bottom_node_indices] = 1
#
#     # 使用 torch.kron 扩展矩阵
#     eye2 = torch.eye(2, dtype=dtype, device=device)
#     B_0 = torch.kron(B_0, eye2)
#
#     # 构建 B_a 矩阵
#     B_a = torch.zeros((total_nodes, 2), dtype=dtype, device=device)
#     B_a[right_slave_nodes, 0] = 1
#     B_a[top_slave_nodes, 1] = 1
#     B_a = torch.kron(B_a, eye2)
#
#     # 构建 B_eps 矩阵
#     B_eps = torch.tensor([
#         [lx, 0.0, ly / 2],
#         [0.0, ly, lx / 2],
#         [lx, 0.0, ly / 2],  # 修正: 根据常见均质化理论，这里应该是a1和a2的贡献
#         [0.0, ly, lx / 2],
#     ], dtype=dtype, device=device)
#     # 注意: B_eps的精确形式可能依赖于您的均质化公式，请验证。
#     # 原始代码中的 B_eps 似乎有误，这里采用一个更标准的形式。
#     # 原始代码的 B_eps:
#     # B_eps_orig = torch.tensor([
#     #     [lx, 0.0, ly / 2],
#     #     [0.0, ly, lx / 2],
#     #     [0.0, 0.0, 0.0], # a_2x=0, a_2y=ly
#     #     [0.0, ly, 0.0],
#     # ], dtype=dtype, device=device)
#
#     # 计算体积
#     V = lx * ly
#
#     return B_0, B_a, B_eps, V
#
#
# def homogenized_elasticity_matrix_2d_torch(nodes, elements, fiber_angles, D_matrix_2D, D_fibre0_2D, device):
#     """
#     在GPU上计算均质化弹性矩阵。
#     所有输入和计算都在指定的GPU设备上进行。
#     """
#     # 1. 计算全局刚度矩阵 (在GPU上)
#     K_uc = global_stiffness.global_stiffness_matrix_torch(nodes, elements, fiber_angles, D_matrix_2D, D_fibre0_2D,
#                                                           device)
#
#     # 2. 计算拓扑矩阵 (在GPU上)
#     # 注意：原始代码对 nodes 进行了转置，这里我们假设输入已经是 (N, 2)
#     B_0, B_a, B_eps, V = topology_matrices_torch(nodes)
#
#     # 3. 求解 D_0 (所有运算都在GPU上)
#     # 添加一个小的对角阵以保证数值稳定性 (regularization)
#     eps_matrix = torch.eye(B_0.shape[1], device=device, dtype=nodes.dtype) * 1e-8
#
#     # 核心线性代数运算
#     B0T_Kuc_B0 = B_0.T @ K_uc @ B_0
#     B0T_Kuc_Ba = B_0.T @ K_uc @ B_a
#
#     D_0 = -torch.linalg.solve(B0T_Kuc_B0 + eps_matrix, B0T_Kuc_Ba)
#
#     # 4. 计算 D_a (在GPU上)
#     D_a = B_0 @ D_0 + B_a
#
#     # 5. 计算 K_delta_a (在GPU上)
#     K_delta_a = D_a.T @ K_uc @ D_a
#
#     # 6. 计算最终的均质化矩阵 K_eps (在GPU上)
#     K_eps = B_eps.T @ K_delta_a @ B_eps / V
#
#     return K_eps
#
#
# def isotropic_elasticity_matrix_2D_torch(E, nu, device, dtype=torch.float32):
#     """PyTorch版本的2D各向同性弹性矩阵"""
#     factor = E / (1 - nu ** 2)
#     D = factor * torch.tensor([
#         [1, nu, 0],
#         [nu, 1, 0],
#         [0, 0, (1 - nu) / 2]
#     ], device=device, dtype=dtype)
#     return D
#
#
# # ==================================================================
# # Section 2: Public-Facing Functions (API)
# # 这些函数保持原始的输入输出格式 (NumPy)，但在内部调用GPU加速版本
# # ==================================================================
#
# def s(fiber_angles: np.ndarray):
#     """
#     根据任意尺寸的 fiber_angles 矩阵进行有限元分析计算。
#     此函数作为CPU和GPU的接口，内部调用GPU加速的计算。
#     输入和输出均为 NumPy 数组以保持兼容性。
#     """
#     if fiber_angles.ndim != 2 or fiber_angles.size == 0:
#         raise ValueError("输入 fiber_angles 必须是一个非空的二维数组。")
#
#     # --- 1. CPU 端: 网格和材料属性设置 (这些操作计算量小) ---
#     ny, nx = fiber_angles.shape
#     fiber_angles_flipped = np.flipud(fiber_angles)
#     fiber_angles_vector = fiber_angles_flipped.flatten()
#
#     lx, ly = 1.0, 1.0
#     num_nodes_x, num_nodes_y = nx + 1, ny + 1
#     x_coords = np.linspace(0, lx, num_nodes_x)
#     y_coords = np.linspace(0, ly, num_nodes_y)
#     num_nodes = num_nodes_x * num_nodes_y
#     node_indices = np.arange(num_nodes).reshape((num_nodes_x, num_nodes_y))
#
#     nodes = np.zeros((num_nodes, 2))
#     for i in range(num_nodes_x):
#         for j in range(num_nodes_y):
#             node_id = node_indices[i, j]
#             nodes[node_id, :] = [x_coords[i], y_coords[j]]
#
#     num_elements = nx * ny
#     elements = np.zeros((num_elements, 4), dtype=int)
#     element_id = 0
#     for i in range(nx):
#         for j in range(ny):
#             n0, n1, n2, n3 = node_indices[i, j], node_indices[i + 1, j], node_indices[i + 1, j + 1], node_indices[
#                 i, j + 1]
#             elements[element_id, :] = [n0, n3, n2, n1]
#             element_id += 1
#
#     nodes[:, [0, 1]] = nodes[:, [1, 0]]  # Swap x and y
#
#     # 材料属性
#     E_matrix, nu_matrix = 2.6e9, 0.3
#     D_matrix_2D = isotropic_elasticity_matrix_2D(E_matrix, nu_matrix)
#
#     E1, E2, nu12, G12 = 1.55e11, 1.21e10, 0.248, 5e9
#     nu21 = nu12 * (E2 / E1)
#     Q11, Q22 = E1 / (1 - nu12 * nu21), E2 / (1 - nu12 * nu21)
#     Q12, Q66 = nu12 * E2 / (1 - nu12 * nu21), G12
#     D_fibre0_2D = np.array([[Q11, Q12, 0], [Q12, Q22, 0], [0, 0, Q66]])
#
#     # --- 2. 数据传输: 将 NumPy 数组转为 PyTorch 张量并移至 GPU ---
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     #print(f"Homoge.s: 计算将在设备 '{device}' 上执行。")
#
#     nodes_t = torch.from_numpy(nodes).to(device, dtype=torch.float32)
#     elements_t = torch.from_numpy(elements).to(device, dtype=torch.long)  # 元素索引是整数
#     fiber_angles_t = torch.from_numpy(fiber_angles_vector).to(device, dtype=torch.float32)
#     D_matrix_2D_t = torch.from_numpy(D_matrix_2D).to(device, dtype=torch.float32)
#     D_fibre0_2D_t = torch.from_numpy(D_fibre0_2D).to(device, dtype=torch.float32)
#
#     # --- 3. GPU 端: 执行所有密集计算 ---
#     K_eps_tensor = homogenized_elasticity_matrix_2d_torch(
#         nodes_t, elements_t, fiber_angles_t, D_matrix_2D_t, D_fibre0_2D_t, device
#     )
#
#     # --- 4. 数据传回: 将最终结果转回 NumPy 数组 ---
#     K_eps_numpy = K_eps_tensor.cpu().numpy()
#
#     return K_eps_numpy
#
#
# # ==================================================================
# # Section 3: Original CPU-based functions (保留以供参考或调试)
# # ==================================================================
#
# def isotropic_elasticity_matrix_2D(E, nu):
#     # 原始NumPy版本
#     D = (E / (1 - nu ** 2)) * np.array([
#         [1, nu, 0],
#         [nu, 1, 0],
#         [0, 0, (1 - nu) / 2]
#     ])
#     return D
#
# # ... 您可以保留 topology_matrices 和 homogenized_elasticity_matrix_2d 的原始NumPy版本在这里 ...
# # ... 以便进行性能比较或调试，但它们不再被 s() 函数调用。

### **修正后的 `Homoge.py`**

import numpy as np
import torch
import scipy.sparse as sp
import global_stiffness  # 假设 global_stiffness.py 在同一目录下


# ==================================================================
# Section 1: GPU-Accelerated Helper Functions (_torch versions)
# ==================================================================

def topology_matrices_torch(nodes: torch.Tensor):
    """
    在GPU上使用PyTorch计算拓扑矩阵 B_0, B_a, B_eps, 和 V。
    """
    device = nodes.device
    dtype = nodes.dtype  # 将从输入继承 dtype (应为 float64)

    min_x, max_x = torch.min(nodes[:, 0]), torch.max(nodes[:, 0])
    min_y, max_y = torch.min(nodes[:, 1]), torch.max(nodes[:, 1])
    lx, ly = max_x - min_x, max_y - min_y

    total_nodes = nodes.shape[0]
    tol = 1e-8

    left_mask = torch.isclose(nodes[:, 0], min_x, atol=tol)
    right_mask = torch.isclose(nodes[:, 0], max_x, atol=tol)
    bottom_mask = torch.isclose(nodes[:, 1], min_y, atol=tol)
    top_mask = torch.isclose(nodes[:, 1], max_y, atol=tol)

    left_master_nodes = torch.where(left_mask)[0]
    right_slave_nodes = torch.where(right_mask)[0]
    bottom_master_nodes = torch.where(bottom_mask)[0]
    top_slave_nodes = torch.where(top_mask)[0]

    left_master_nodes = left_master_nodes[torch.argsort(nodes[left_master_nodes, 1])]
    right_slave_nodes = right_slave_nodes[torch.argsort(nodes[right_slave_nodes, 1])]
    bottom_master_nodes = bottom_master_nodes[torch.argsort(nodes[bottom_master_nodes, 0])]
    top_slave_nodes = top_slave_nodes[torch.argsort(nodes[top_slave_nodes, 0])]

    master_nodes = torch.unique(torch.cat((left_master_nodes, bottom_master_nodes)))
    slave_nodes = torch.unique(torch.cat((right_slave_nodes, top_slave_nodes)))

    all_nodes = torch.arange(total_nodes, device=device)
    boundary_nodes = torch.cat((master_nodes, slave_nodes))
    is_boundary = torch.zeros(total_nodes, dtype=torch.bool, device=device)
    is_boundary[boundary_nodes] = True
    interior_nodes = all_nodes[~is_boundary]

    independent_nodes = torch.cat((master_nodes, interior_nodes))
    num_independent = independent_nodes.size(0)

    B_0 = torch.zeros((total_nodes, num_independent), dtype=dtype, device=device)
    B_0[independent_nodes, torch.arange(num_independent, device=device)] = 1

    sorted_independent_nodes, sort_indices = torch.sort(independent_nodes)

    left_node_indices_in_sorted = torch.searchsorted(sorted_independent_nodes, left_master_nodes)
    left_node_indices = sort_indices[left_node_indices_in_sorted]
    B_0[right_slave_nodes, left_node_indices] = 1

    bottom_node_indices_in_sorted = torch.searchsorted(sorted_independent_nodes, bottom_master_nodes)
    bottom_node_indices = sort_indices[bottom_node_indices_in_sorted]
    B_0[top_slave_nodes, bottom_node_indices] = 1

    eye2 = torch.eye(2, dtype=dtype, device=device)
    B_0 = torch.kron(B_0, eye2)

    B_a = torch.zeros((total_nodes, 2), dtype=dtype, device=device)
    B_a[right_slave_nodes, 0] = 1
    B_a[top_slave_nodes, 1] = 1
    B_a = torch.kron(B_a, eye2)

    # <<< 修正: B_eps 的实现现在严格匹配原始 NumPy 代码 >>>
    a_1x, a_1y = lx, torch.tensor(0.0, device=device, dtype=dtype)
    a_2x, a_2y = torch.tensor(0.0, device=device, dtype=dtype), ly
    B_eps = torch.tensor([
        [a_1x, 0.0, a_1y / 2],
        [0.0, a_1y, a_1x / 2],
        [a_2x, 0.0, a_2y / 2],
        [0.0, a_2y, a_2x / 2],
    ], dtype=dtype, device=device)

    V = lx * ly

    return B_0, B_a, B_eps, V


def homogenized_elasticity_matrix_2d_torch(nodes, elements, fiber_angles, D_matrix_2D, D_fibre0_2D, device):
    """
    在GPU上计算均质化弹性矩阵。
    """
    dtype = torch.float64

    # 1. 计算 K_uc。此函数内部已处理好所有张量的设备和类型。
    # K_uc 将在指定的 `device` (GPU) 上。
    K_uc = global_stiffness.global_stiffness_matrix_torch(nodes, elements, fiber_angles, D_matrix_2D, D_fibre0_2D,
                                                          device)

    # 2. <<< 关键修正点 >>>
    #    在调用 topology_matrices_torch 之前，确保 nodes 张量
    #    不仅类型正确，而且也位于正确的 `device` (GPU) 上。
    nodes_on_device = nodes.to(device=device, dtype=dtype)
    B_0, B_a, B_eps, V = topology_matrices_torch(nodes_on_device)

    # 现在 B_0, B_a, B_eps 都在 GPU 上，可以和 K_uc 进行运算。

    # 3. 后续计算
    eps_matrix = torch.eye(B_0.shape[1], device=device, dtype=dtype) * 1e-8

    B0T_Kuc_B0 = B_0.T @ K_uc @ B_0
    B0T_Kuc_Ba = B_0.T @ K_uc @ B_a

    D_0 = -torch.linalg.solve(B0T_Kuc_B0 + eps_matrix, B0T_Kuc_Ba)
    D_a = B_0 @ D_0 + B_a
    K_delta_a = D_a.T @ K_uc @ D_a
    K_eps = B_eps.T @ K_delta_a @ B_eps / V

    return K_eps


# ==================================================================
# Section 2: Public-Facing Functions (API)
# ==================================================================
def s(fiber_angles: np.ndarray):
    """
    根据任意尺寸的 fiber_angles 矩阵进行有限元分析计算。
    此函数作为CPU和GPU的接口，内部调用GPU加速的计算。
    """
    if fiber_angles.ndim != 2 or fiber_angles.size == 0:
        raise ValueError("输入 fiber_angles 必须是一个非空的二维数组。")

    ny, nx = fiber_angles.shape
    fiber_angles_flipped = np.flipud(fiber_angles)
    fiber_angles_vector = fiber_angles_flipped.flatten()

    lx, ly = 1.0, 1.0
    num_nodes_x, num_nodes_y = nx + 1, ny + 1
    x_coords = np.linspace(0, lx, num_nodes_x)
    y_coords = np.linspace(0, ly, num_nodes_y)
    num_nodes = num_nodes_x * num_nodes_y
    node_indices = np.arange(num_nodes).reshape((num_nodes_x, num_nodes_y))

    nodes = np.zeros((num_nodes, 2))
    for i in range(num_nodes_x):
        for j in range(num_nodes_y):
            nodes[node_indices[i, j], :] = [x_coords[i], y_coords[j]]

    num_elements = nx * ny
    elements = np.zeros((num_elements, 4), dtype=int)
    # 修正网格生成逻辑以匹配原始代码
    element_id = 0
    for i in range(nx):
        for j in range(ny):
            n0, n1, n2, n3 = node_indices[i, j], node_indices[i + 1, j], node_indices[i + 1, j + 1], node_indices[
                i, j + 1]
            elements[element_id, :] = [n0, n3, n2, n1]
            element_id += 1

    nodes[:, [0, 1]] = nodes[:, [1, 0]]

    E_matrix, nu_matrix = 2.6e9, 0.3
    D_matrix_2D = (E_matrix / (1 - nu_matrix ** 2)) * np.array(
        [[1, nu_matrix, 0], [nu_matrix, 1, 0], [0, 0, (1 - nu_matrix) / 2]])

    E1, E2, nu12, G12 = 1.55e11, 1.21e10, 0.248, 5e9
    nu21 = nu12 * (E2 / E1)
    Q11, Q22 = E1 / (1 - nu12 * nu21), E2 / (1 - nu12 * nu21)
    Q12, Q66 = nu12 * E2 / (1 - nu12 * nu21), G12
    D_fibre0_2D = np.array([[Q11, Q12, 0], [Q12, Q22, 0], [0, 0, Q66]])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    nodes_t = torch.from_numpy(nodes)
    elements_t = torch.from_numpy(elements)
    fiber_angles_t = torch.from_numpy(fiber_angles_vector)
    D_matrix_2D_t = torch.from_numpy(D_matrix_2D)
    D_fibre0_2D_t = torch.from_numpy(D_fibre0_2D)  # <<< 修正了这里的拼写错误

    K_eps_tensor = homogenized_elasticity_matrix_2d_torch(
        nodes_t, elements_t, fiber_angles_t, D_matrix_2D_t, D_fibre0_2D_t, device
    )

    K_eps_numpy = K_eps_tensor.cpu().numpy()

    return K_eps_numpy