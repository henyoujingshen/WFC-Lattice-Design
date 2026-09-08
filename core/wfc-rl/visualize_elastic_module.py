# Copyright (c) 2026 Tang Hubocheng. All rights reserved.
# Original project code: no use or redistribution without written permission.
# See the repository LICENSE for scope, exceptions and third-party rights.
import numpy as np

import matplotlib.pyplot as plt
def calculate_modulus_2D(CH):
    """
    计算二维方向的弹性模量。
    输入:
        CH: 3x3 弹性矩阵
    输出:
        E: 包含三个模量的列表
    """
    S = np.linalg.inv(CH)  # 顺应性矩阵
    E = [1 / S[0, 0],  # 第一主方向模量
         1 / S[1, 1],  # 第二主方向模量
         1 / S[2, 2]]  # 剪切模量
    return E


def generate_2D(CH):
    """
    转换 3x3 二维弹性矩阵 CH 到 2x2x2x2 张量。
    """
    C = np.zeros((2, 2, 2, 2))
    for i in range(3):
        for j in range(3):
            a, b = change_2D(i + 1)
            c, d = change_2D(j + 1)
            C[a - 1, b - 1, c - 1, d - 1] = CH[i, j]

    # 填充对称性
    for i in range(2):
        for j in range(2):
            for k in range(2):
                for l in range(2):
                    C[j, i, k, l] = C[i, j, k, l]
                    C[i, j, l, k] = C[i, j, k, l]
                    C[j, i, l, k] = C[i, j, k, l]
    return C


def to_matrix_2D(C):
    """
    将 2x2x2x2 张量转换为 3x3 矩阵形式。
    """
    CH = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            a, b = change_2D(i + 1)
            c, d = change_2D(j + 1)
            CH[i, j] = C[a - 1, b - 1, c - 1, d - 1]
    return CH


def change_2D(w):
    """
    映射 1, 2, 3 -> (1,1), (2,2), (1,2)
    """
    if w == 1:
        return 1, 1
    elif w == 2:
        return 2, 2
    elif w == 3:
        return 1, 2


def transform_2D(tensor, trans):
    """
    二维情况下弹性张量的坐标变换。
    """
    N_tensor = np.zeros_like(tensor)
    for i in range(2):
        for j in range(2):
            for k in range(2):
                for l in range(2):
                    for m in range(2):
                        for n in range(2):
                            for p in range(2):
                                for q in range(2):
                                    N_tensor[i, j, k, l] += (
                                            trans[i, m] * trans[j, n] *
                                            trans[k, p] * trans[l, q] *
                                            tensor[m, n, p, q]
                                    )
    return N_tensor


def calculate_stiffness_directions(CH):
    """
    计算每 15 度方向的弹性模量，返回 6 个方向的模量值。
    输入:
        CH: 3x3 弹性矩阵
    输出:
        stiffness_values: 长度为6的方向刚度值列表
    """
    tensor = generate_2D(CH)
    theta = np.linspace(np.pi/6, np.pi, 6)  # 每 15 度 (π/12)，共 6 个方向

    stiffness_values = []
    for angle in theta:
        # 构建旋转矩阵
        trans = np.array([[np.cos(angle), -np.sin(angle)],
                          [np.sin(angle), np.cos(angle)]])

        # 计算旋转后的张量
        N_tensor = transform_2D(tensor, trans)

        # 转换成 3x3 矩阵形式
        N_CH = to_matrix_2D(N_tensor)

        # 计算弹性模量
        E = calculate_modulus_2D(N_CH)

        # 保存第一主方向模量
        stiffness_values.append(E[0])

    return stiffness_values


import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

import numpy as np
import matplotlib.pyplot as plt


import numpy as np
from scipy.interpolate import Akima1DInterpolator
import matplotlib.pyplot as plt


def generate_smooth_stiffness_data(stiffness_vector):
    """
    根据给定的6个方向的刚度值，生成一个用于绘制平滑360度刚度圆的数据点集。

    输入:
        stiffness_vector (list or np.array): 长度为6的刚度值向量。
            这6个点分别对应 30, 60, 90, 120, 150, 180 度。
            (坐标系: 9点钟方向为起点，顺时针)

    输出:
        tuple: (angles_for_plot_rad, stiffness_smooth)
            - angles_for_plot_rad: 用于绘图的361个平滑角度点 (单位: 弧度)
            - stiffness_smooth: 对应的361个平滑刚度值
    """
    if len(stiffness_vector) != 6:
        raise ValueError("输入向量的长度必须为 6。")

    # --- 1. 定义原始角度并进行对称扩展 ---
    # 用户的角度定义: 30, 60, 90, 120, 150, 180 度
    user_angles_deg = np.array([30, 60, 90, 120, 150, 180])

    # 根据对称性 E(theta) = E(theta + 180°)，扩展数据以覆盖360度
    # 新的角度为: 210, 240, 270, 300, 330, 360
    full_user_angles_deg = np.concatenate([user_angles_deg, user_angles_deg + 180])
    full_stiffness = np.concatenate([stiffness_vector, stiffness_vector])

    # 注意：用户的 360 度和 0 度是同一点。我们将 360 度改为 0 度以方便处理。
    full_user_angles_deg[full_user_angles_deg == 360] = 0

    # --- 2. 坐标系转换 ---
    # Matplotlib的极坐标系：0度在3点钟(东方)，逆时针增长。
    # 用户的坐标系：    0度在9点钟(西方)，顺时针增长。
    # 转换公式: matplotlib_angle = 270 - user_angle (或者 3*pi/2 - user_angle_rad)
    # 让我们验证一下：
    # user=0 (9点) -> plot=270 (下方) -> 不对
    # 转换公式2: plot_angle_rad = pi - user_angle_rad
    # user=0 (9点, pi) -> plot=pi-pi = 0 -> 不对
    # 正确的转换：
    # 将用户角度从度数转换为弧度
    full_user_angles_rad = np.deg2rad(full_user_angles_deg)

    # --- 3. Akima 插值 (实现平缓平滑) ---
    # 为了让插值器理解周期性，我们需要对数据进行排序和封装
    # 将角度和刚度数据对齐并按角度排序
    sort_indices = np.argsort(full_user_angles_rad)
    sorted_angles_rad = full_user_angles_rad[sort_indices]
    sorted_stiffness = full_stiffness[sort_indices]

    # 添加一个周期点，将第一个点(0 rad)的数据附加到末尾(2*pi rad)
    # 这能确保曲线在起点和终点完美闭合
    periodic_angles = np.append(sorted_angles_rad, sorted_angles_rad[0] + 2 * np.pi)
    periodic_stiffness = np.append(sorted_stiffness, sorted_stiffness[0])

    # 创建 Akima 插值器
    akima_interpolator = Akima1DInterpolator(periodic_angles, periodic_stiffness)

    # 在一个密集的点集上生成平滑数据 (361个点以包含起点和终点)
    angles_for_plot_rad = np.linspace(0, 2 * np.pi, 361)
    stiffness_smooth = akima_interpolator(angles_for_plot_rad)

    return angles_for_plot_rad, stiffness_smooth


def plot_stiffness_circle(smooth_angles, smooth_stiffness, original_stiffness_vector):
    """根据生成的平滑数据点绘制刚度圆。"""

    # --- 准备原始数据点用于绘图 ---
    user_angles_deg = np.array([30, 60, 90, 120, 150, 180])
    full_user_angles_deg = np.concatenate([user_angles_deg, user_angles_deg + 180])
    full_user_angles_deg[full_user_angles_deg == 360] = 0
    original_angles_rad = np.deg2rad(full_user_angles_deg)
    original_stiffness = np.concatenate([original_stiffness_vector, original_stiffness_vector])

    # --- 绘图 ---
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={'projection': 'polar'})

    # 绘制平滑曲线
    ax.plot(smooth_angles, smooth_stiffness, '-', linewidth=2.5, label='Akima Smoothed Stiffness')

    # 绘制原始数据点
    ax.plot(original_angles_rad, original_stiffness, 'o', color='red', markersize=8, label='Original Data Points')

    # 填充区域
    ax.fill(smooth_angles, smooth_stiffness, alpha=0.2)

    # --- 设置坐标系以匹配用户定义 ---
    # 将 0 度设置在9点钟方向 (West)
    ax.set_theta_zero_location('W')
    # 将角度增长方向设置为顺时针
    ax.set_theta_direction('clockwise')

    # 美化图表
    ax.set_title('360-Degree Stiffness Visualization', fontsize=16, pad=20)
    ax.set_rlabel_position(90)  # 将半径标签放在90度位置
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    ax.grid(True)

    plt.show()


# --- 主程序入口 ---
if __name__ == '__main__':
    # 您的实际1x6输入向量
    # (30°, 60°, 90°, 120°, 150°, 180°)
    my_stiffness_vector = [13350748894.12578, 19394979215.591995, 33040842837.173363, 19242335660.441475, 13202262899.165756, 12922593189.579687]
    # 1. 生成平滑的数据点
    smooth_angles_rad, smooth_stiffness_values = generate_smooth_stiffness_data(my_stiffness_vector)

    # 2. 输出连接的数据点 (这里只展示部分，因为总共有361个点)
    print("--- 用于绘图的平滑数据点 (角度[弧度], 刚度值) ---")
    for i in range(0, 361, 30):  # 每隔30个点打印一次作为示例
        print(f"Angle: {smooth_angles_rad[i]:.4f}, Stiffness: {smooth_stiffness_values[i]:.4f}")

    print("\n总共生成了 {} 个平滑数据点。".format(len(smooth_stiffness_values)))

    # 3. 可视化结果
    plot_stiffness_circle(smooth_angles_rad, smooth_stiffness_values, my_stiffness_vector)