# Copyright (c) 2026 Tang Hubocheng. All rights reserved.
# Original project code: no use or redistribution without written permission.
# See the repository LICENSE for scope, exceptions and third-party rights.
import numpy as np
import scipy.sparse as sp
import torch


# ==================================================================
# Section 1: GPU-Accelerated Helper Functions (_torch versions)
# ==================================================================

def element_stiffness_matrix_torch(vx, vy, D_batch, device, dtype=torch.float64):
    """
    在GPU上批量计算所有单元的单元刚度矩阵。
    """
    num_elements = vx.shape[0]
    Wgt = 1.0
    gauss_coords = torch.tensor([-1, 1], device=device, dtype=dtype) / torch.sqrt(torch.tensor(3.0, device=device))
    r_nodes = torch.tensor([-1, 1, 1, -1], device=device, dtype=dtype)
    s_nodes = torch.tensor([-1, -1, 1, 1], device=device, dtype=dtype)
    ke_batch = torch.zeros((num_elements, 8, 8), device=device, dtype=dtype)
    for r in gauss_coords:
        for s in gauss_coords:
            dNdr = 0.25 * r_nodes * (1 + s_nodes * s)
            dNds = 0.25 * s_nodes * (1 + r_nodes * r)
            dxdr = torch.einsum('p,ep->e', dNdr, vx)
            dydr = torch.einsum('p,ep->e', dNdr, vy)
            dxds = torch.einsum('p,ep->e', dNds, vx)
            dyds = torch.einsum('p,ep->e', dNds, vy)
            j_batch = dxdr * dyds - dxds * dydr
            inv_j = 1.0 / j_batch
            dNdx = inv_j.unsqueeze(1) * (dNdr.unsqueeze(0) * dyds.unsqueeze(1) - dNds.unsqueeze(0) * dydr.unsqueeze(1))
            dNdy = inv_j.unsqueeze(1) * (dNds.unsqueeze(0) * dxdr.unsqueeze(1) - dNdr.unsqueeze(0) * dxds.unsqueeze(1))
            B_batch = torch.zeros(num_elements, 3, 8, device=device, dtype=dtype)
            B_batch[:, 0, 0::2] = dNdx
            B_batch[:, 1, 1::2] = dNdy
            B_batch[:, 2, 0::2] = dNdy
            B_batch[:, 2, 1::2] = dNdx
            integrand = torch.bmm(B_batch.transpose(1, 2), torch.bmm(D_batch, B_batch))
            ke_batch += integrand * j_batch.view(-1, 1, 1) * Wgt
    return ke_batch


def rotation_matrix_3x3_torch(angle_deg, dtype=torch.float64):
    theta = torch.deg2rad(angle_deg)
    cos_theta, sin_theta = torch.cos(theta), torch.sin(theta)
    c2, s2, cs = cos_theta ** 2, sin_theta ** 2, cos_theta * sin_theta
    R_batch = torch.stack([
        torch.stack([c2, s2, 2 * cs], dim=-1),
        torch.stack([s2, c2, -2 * cs], dim=-1),
        torch.stack([-cs, cs, c2 - s2], dim=-1)
    ], dim=1).to(dtype=dtype)
    return R_batch


def transform_stiffness_matrix_3x3_torch(C_batch, R_batch):
    term1 = torch.bmm(R_batch.transpose(1, 2), C_batch)
    C_transformed = torch.bmm(term1, R_batch)
    return C_transformed


def global_stiffness_matrix_torch(nodes, elements, fiber_angles, D_matrix_2D, D_fiber0_2D, device):
    dtype = torch.float64

    nodes = nodes.to(device=device, dtype=dtype)

    # <<< 关键修正点 >>>
    # 显式地将 elements 转换为 torch.long 类型
    elements = elements.to(device=device, dtype=torch.long)

    fiber_angles = fiber_angles.to(device=device, dtype=dtype)
    D_matrix_2D = D_matrix_2D.to(device=device, dtype=dtype)
    D_fiber0_2D = D_fiber0_2D.to(device=device, dtype=dtype)

    num_nodes = nodes.shape[0]
    num_dofs = 2 * num_nodes
    num_elements = elements.shape[0]

    el_nodes_coords = nodes[elements]  # 现在这个索引操作是合法的
    vx = el_nodes_coords[:, :, 0]
    vy = el_nodes_coords[:, :, 1]

    D_batch = torch.zeros((num_elements, 3, 3), device=device, dtype=dtype)
    matrix_mask = (fiber_angles == 1)
    fiber_mask = ~matrix_mask

    if torch.any(matrix_mask):
        D_batch[matrix_mask] = D_matrix_2D

    if torch.any(fiber_mask):
        fiber_angles_val = fiber_angles[fiber_mask]
        R_batch = rotation_matrix_3x3_torch(fiber_angles_val, dtype=dtype)
        num_fibers = R_batch.shape[0]
        D_fiber_batch = D_fiber0_2D.unsqueeze(0).expand(num_fibers, -1, -1)
        D_rotated = transform_stiffness_matrix_3x3_torch(D_fiber_batch, R_batch)
        D_batch[fiber_mask] = D_rotated

    ke_batch = element_stiffness_matrix_torch(vx, vy, D_batch, device, dtype=dtype)

    dofs_x = 2 * elements
    dofs_y = 2 * elements + 1
    elem_dofs = torch.stack((dofs_x, dofs_y), dim=-1).view(num_elements, 8)

    i_indices = elem_dofs.unsqueeze(2).expand(-1, -1, 8).flatten()
    j_indices = elem_dofs.unsqueeze(1).expand(-1, 8, -1).flatten()
    values = ke_batch.flatten()

    K_uc_sparse = torch.sparse_coo_tensor(
        indices=torch.stack([i_indices, j_indices]),
        values=values,
        size=(num_dofs, num_dofs)
    )

    K_uc = K_uc_sparse.to_dense()

    return K_uc