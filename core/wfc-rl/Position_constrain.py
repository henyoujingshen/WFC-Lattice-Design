# In Position_constrain.py

class ConstraintViolation(Exception):
    """
    当 WFC 算法在约束传播过程中检测到冲突时抛出的自定义异常。

    一个冲突意味着某个单元格的可能性列表变为空，表明没有有效的
    状态可以放置在该位置，从而使得当前的布局无效。

    Attributes:
        message (str): 描述错误的字符串。
        position (tuple, optional): 一个 (row, col) 元组，指示冲突发生的位置。
    """

    def __init__(self, message, position=None):
        super().__init__(message)
        self.position = position


# ==================================================================
# >>>>>>>>>> REFACTORED WITH CONFIGURATION DICTIONARY <<<<<<<<<<
# ==================================================================

# 使用配置字典来消除 if/elif 重复代码，提高可维护性
DIRECTION_CONFIG = {
    "posX": {"offset": (0, 2), "diag_up": (-1, 1), "diag_down": (1, 1),
             "R_edge": "posX", "RS_L_edge": "negX", "RU_B_edge": "negY", "RD_T_edge": "posY"},
    "negX": {"offset": (0, -2), "diag_up": (-1, -1), "diag_down": (1, -1),
             "R_edge": "negX", "RS_L_edge": "posX", "RU_B_edge": "negY", "RD_T_edge": "posY"},
    "posY": {"offset": (-2, 0), "diag_up": (-1, -1), "diag_down": (-1, 1),
             "R_edge": "posY", "RS_L_edge": "negY", "RU_B_edge": "posX", "RD_T_edge": "negX"},
    "negY": {"offset": (2, 0), "diag_up": (1, -1), "diag_down": (1, 1),
             "R_edge": "negY", "RS_L_edge": "posY", "RU_B_edge": "posX", "RD_T_edge": "negX"},
}


def check_direction_constraint(state, i, j, grid, fiber_block_library, direction):
    """
    检查给定状态在特定方向上是否满足复杂的“间隔对角”约束。

    此约束基于一个奇偶校验规则：一个块边界(R)的状态必须等于其
    间隔邻居(RS_L)和两个对角邻居(RU_B, RD_T)的边界状态之和模2。
    (R = (RS_L + RU_B + RD_T) % 2)

    如果任何相关邻居有多种可能性（未坍缩），则约束检查会推迟，
    函数返回 True，允许当前状态作为一种可能性保留。

    Args:
        state (str): 要检查其有效性的状态名称 (e.g., "Type1_90")。
        i (int): 状态所在单元格的行索引。
        j (int): 状态所在单元格的列索引。
        grid (list[list[list[str]]]): 当前的 WFC 网格。
        fiber_block_library (dict): 从状态名称到 Prototype 对象的映射。
        direction (str): 要检查的方向 ("posX", "negX", "posY", "negY")。

    Returns:
        bool: 如果状态满足约束或约束被推迟，则返回 True。
              如果状态明确违反了约束，则返回 False。

    Raises:
        ConstraintViolation: 如果在查找邻居状态时发生意外错误。
    """
    # 1. 从配置字典中获取该方向的所有坐标偏移和边界名称
    config = DIRECTION_CONFIG.get(direction)
    if not config:
        raise ValueError(f"无效的方向: {direction}")

    # 2. 获取当前状态在该方向上的边界类型 (0 或 1)
    proto = fiber_block_library[state]
    R = proto.sockets.get(config["R_edge"], 0)

    # --- 内部辅助函数，用于安全地获取邻居的边界状态 ---
    def get_boundary_states(offset_i, offset_j, edge):
        """安全地获取指定偏移处单元格的所有可能边界状态。"""
        cell_i, cell_j = i + offset_i, j + offset_j
        if 0 <= cell_i < len(grid) and 0 <= cell_j < len(grid[0]):
            neighbor_states = grid[cell_i][cell_j]
            if not neighbor_states:  # 如果邻居的可能性列表为空
                return []
            return [fiber_block_library[s].sockets.get(edge, 0) for s in neighbor_states]
        else:
            return [0]  # 网格外部的边界默认为 0

    # 3. 计算相关邻居的坐标并获取它们的边界状态
    RS_L_states = get_boundary_states(*config["offset"], config["RS_L_edge"])
    RU_B_states = get_boundary_states(*config["diag_up"], config["RU_B_edge"])
    RD_T_states = get_boundary_states(*config["diag_down"], config["RD_T_edge"])

    # 4. 检查是否有任何邻居的状态列表为空（这是一个错误条件）
    if not (RS_L_states and RU_B_states and RD_T_states):
        raise ConstraintViolation(
            f"约束检查失败，因为在({i},{j})的邻居处发现了一个空的候选列表 (方向: {direction})",
            position=(i, j)
        )

    # 5. 检查邻居是否已完全坍缩（即只有一个确定的边界状态）
    # 如果任何一个邻居仍有多种可能性，我们就推迟约束检查。
    if any(len(set(states)) > 1 for states in [RS_L_states, RU_B_states, RD_T_states]):
        return True  # 推迟检查，暂时认为其兼容

    # 6. 如果所有邻居都已坍缩，执行核心的奇偶校验规则
    # R_expected = (RS_L + RU_B + RD_T) mod 2
    RS_L = RS_L_states[0]
    RU_B = RU_B_states[0]
    RD_T = RD_T_states[0]

    expected_R = (RS_L + RU_B + RD_T) % 2

    # 如果当前状态的边界与期望值匹配，则约束满足
    return R == expected_R