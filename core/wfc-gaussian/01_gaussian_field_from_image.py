# Copyright (c) 2026 Tang Hubocheng. All rights reserved.
# Original project code: no use or redistribution without written permission.
# See the repository LICENSE for scope, exceptions and third-party rights.
import cv2
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.cluster import SpectralClustering
import json
import math
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os


def get_dominant_colors(image, k=3):
    """
    使用K-Means算法找出图像中的主要颜色。
    """
    pixels = image.reshape((-1, 3)).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    centers = np.uint8(centers)
    return centers


def decompose_covariance(covariance, image_shape):
    """
    将协方差矩阵分解为缩放(scale)和旋转角度(angle)。
    """
    eigenvalues, eigenvectors = np.linalg.eig(covariance)
    # 确保特征值和特征向量是实数，以避免复数计算问题
    eigenvalues = np.real(eigenvalues)
    eigenvectors = np.real(eigenvectors)

    # 计算旋转角度
    angle = math.atan2(eigenvectors[1, 0], eigenvectors[0, 0])

    # 根据图像尺寸归一化尺度
    # 使用特征值的绝对值的平方根来计算标准差
    scale_x = np.sqrt(np.abs(eigenvalues[0])) / image_shape[1]
    scale_y = np.sqrt(np.abs(eigenvalues[1])) / image_shape[0]

    # 避免尺度为零，设置一个最小值
    scale_x = max(scale_x, 0.01)
    scale_y = max(scale_y, 0.01)

    return [scale_x, scale_y], angle


def generate_gaussian_field_from_image_v6(image_path, output_path,
                                          components_per_color,
                                          k_colors=2,
                                          # ▼▼▼ 新增超参数，并带有默认值 ▼▼▼
                                          scale_multiplier=1.0,
                                          opacity_regulator=1.0,
                                          # ▲▲▲ 新增超参数 ▲▲▲
                                          visualize_results=False):
    """
    使用“分割优先”策略，并允许通过超参数调控高斯场的生成。

    :param image_path: 输入图像的文件路径。
    :param output_path: 输出JSON文件的路径。
    :param components_per_color: 一个列表，每个元素代表对应主色调要拟合的分量数量。
    :param k_colors: 要从图像中提取的主色调数量。
    :param scale_multiplier: 尺度缩放因子，控制椭球大小和重叠度 (>1更大, <1更小)。
    :param opacity_regulator: 不透明度调节器，控制单元格偏好强度。
    :param visualize_results: 如果为True，则显示每个颜色组的拟合结果图。
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"错误: 无法读取图像文件 '{image_path}'。")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    height, width, _ = img.shape

    non_white_mask = np.all(img != [255, 255, 255], axis=-1)
    pixels_for_color_detection = img[non_white_mask]
    if len(pixels_for_color_detection) == 0:
        print("警告: 图像中未找到非白色像素，无法提取主色调。")
        return

    dominant_colors = get_dominant_colors(pixels_for_color_detection, k=k_colors)

    if len(components_per_color) != k_colors:
        raise ValueError(
            f"参数 'components_per_color' 的长度 ({len(components_per_color)}) 必须与 'k_colors' ({k_colors}) 的值相等。")

    gaussian_list = []
    color_names = ["red", "green", "blue", "yellow", "cyan", "magenta"]

    for color_index, color in enumerate(dominant_colors):
        assigned_color_name = color_names[color_index % len(color_names)]
        print(f"处理颜色组 {color_index + 1} (RGB: {color}), 分配名称: '{assigned_color_name}'")

        n_components = components_per_color[color_index]
        print(f"  -> 将使用 {n_components} 个分量进行拟合。")

        # 使用更宽容的颜色范围来查找像素点
        color_diff = 15
        lower_bound = np.clip(color.astype(int) - color_diff, 0, 255).astype(np.uint8)
        upper_bound = np.clip(color.astype(int) + color_diff, 0, 255).astype(np.uint8)
        mask = cv2.inRange(img, lower_bound, upper_bound)
        points = np.argwhere(mask > 0)

        if len(points) == 0:
            print(f"  -> 警告: 未找到颜色 {color} 的像素点，跳过。")
            continue

        points = points[:, [1, 0]]  # 转换为 (x, y) 坐标

        if len(points) < n_components:
            print(f"  -> 警告: 像素点数量 ({len(points)}) 过少，无法聚类为 {n_components} 个部分。")
            continue

        clustering = SpectralClustering(n_clusters=n_components, affinity='nearest_neighbors', random_state=0)
        labels = clustering.fit_predict(points)

        if visualize_results:
            fig, ax = plt.subplots(1, figsize=(8, 8))
            color_cycle = plt.get_cmap('viridis')(np.linspace(0, 1, n_components))

        for i in range(n_components):
            points_of_component = points[labels == i]
            if len(points_of_component) < 2:
                continue

            gmm_single = GaussianMixture(n_components=1, covariance_type='full', n_init=3, random_state=0)
            gmm_single.fit(points_of_component)

            mean = gmm_single.means_[0]
            covariance = gmm_single.covariances_[0]

            mu_normalized = [mean[0] / width, 1.0 - (mean[1] / height)]

            # --- 参数提取与超参数应用 ---
            raw_opacity = min((len(points_of_component) / len(points)) * 1.5, 1.0)
            opacity = min(raw_opacity * opacity_regulator, 1.0)

            scale_normalized, angle_for_json = decompose_covariance(covariance, (height, width))
            scale_normalized = [s * scale_multiplier for s in scale_normalized]

            gaussian = {
                "color": assigned_color_name, "mu": mu_normalized, "scale": scale_normalized,
                "opacity": opacity, "angle": angle_for_json, "rgb_color": [c / 255.0 for c in color.tolist()]
            }
            gaussian_list.append(gaussian)

            if visualize_results:
                component_color = color_cycle[i]
                ax.scatter(points_of_component[:, 0], points_of_component[:, 1], c=[component_color], s=5,
                           label=f'Component {i + 1}')
                eigenvalues, eigenvectors = np.linalg.eig(covariance)
                angle_rad = np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0])
                angle_deg = np.rad2deg(angle_rad)
                n_std = 2.0
                width_ellipse = 2 * n_std * np.sqrt(np.abs(eigenvalues[0]))
                height_ellipse = 2 * n_std * np.sqrt(np.abs(eigenvalues[1]))
                ellipse = patches.Ellipse(xy=mean, width=width_ellipse, height=height_ellipse, angle=angle_deg,
                                          edgecolor=component_color, facecolor='none', linewidth=2, linestyle='--')
                ax.add_patch(ellipse)

        if visualize_results:
            ax.set_title(f'Clustered Points and Fitted Ellipses for Color "{assigned_color_name}"')
            ax.set_xlabel("X coordinate");
            ax.set_ylabel("Y coordinate")
            ax.invert_yaxis();
            ax.axis('equal');
            ax.legend()
            plt.show()

    output_data = {
        "version": 1.0, "bounds": [0.0, 1.0, 0.0, 1.0], "grid_size": 250.0, "field": {"gaussians": gaussian_list}
    }
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    print(f"\n高斯场已成功生成并保存至: {output_path}")
    print(f"  - 使用参数: scale_multiplier={scale_multiplier}, opacity_regulator={opacity_regulator}")



import json
import numpy as np
import matplotlib.pyplot as plt


def visualize_gaussian_field(json_path, resolution=600):
    """
    读取描述高斯场参数的JSON文件，并将其可视化为一张柔和的图像。

    :param json_path: 包含高斯椭球参数的JSON文件路径。
    :param resolution: 输出图像的正方形分辨率（像素）。
    """
    # 1. 加载JSON数据
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        gaussians = data['field']['gaussians']
        print(f"成功加载 {len(gaussians)} 个高斯椭球的数据。")
    except FileNotFoundError:
        print(f"错误: JSON文件未找到，请确认路径 '{json_path}' 是否正确。")
        return
    except (json.JSONDecodeError, KeyError):
        print(f"错误: JSON文件格式不正确或缺少必要的字段。")
        return

    # 2. 创建画布和坐标网格
    # 创建一个纯白色的背景画布 (R=1, G=1, B=1)
    canvas = np.ones((resolution, resolution, 3))

    # 创建一个从 [0, 1] 范围的坐标网格，用于计算
    # linspace生成等差数列，meshgrid将其扩展为二维网格
    x = np.linspace(0, 1, resolution)
    y = np.linspace(0, 1, resolution)
    xx, yy = np.meshgrid(x, y)

    # 3. 逐个绘制并混合高斯椭球
    print("开始在画布上渲染高斯椭球...")
    for i, g in enumerate(gaussians):
        # 提取参数
        mu_x, mu_y = g['mu']
        scale_x, scale_y = g['scale']
        angle = g['angle']
        rgb_color = np.array(g['rgb_color'])
        opacity = g['opacity']

        # --- 核心数学计算 ---
        # a. 坐标系转换：将网格坐标原点移至高斯中心
        #    注意：JSON的y坐标是数学坐标系（y=0在下），而图像数组是计算机坐标系（y=0在上）
        #    因此我们需要反转 mu_y
        x_prime = xx - mu_x
        y_prime = yy - (1.0 - mu_y)

        # b. 反向旋转坐标系，角度要取负值
        cos_a = np.cos(-angle)
        sin_a = np.sin(-angle)
        x_rot = x_prime * cos_a - y_prime * sin_a
        y_rot = x_prime * sin_a + y_prime * cos_a

        # c. 计算2D高斯函数值 (强度)
        #    这是标准的高斯分布公式，决定了椭球的形状和模糊程度
        #    增加一个极小值避免除以零
        sigma_x_sq = (scale_x + 1e-6) ** 2
        sigma_y_sq = (scale_y + 1e-6) ** 2

        intensity = np.exp(-((x_rot ** 2) / (2 * sigma_x_sq) + (y_rot ** 2) / (2 * sigma_y_sq)))

        # d. 应用不透明度，并将其维度扩展以便与颜色相乘
        alpha = opacity * intensity[..., np.newaxis]  # Shape: (res, res, 1)

        # e. Alpha混合 (Compositing)
        #    公式: C_out = C_layer * alpha + C_bg * (1 - alpha)
        #    C_layer 是当前高斯椭球的颜色
        #    C_bg 是当前画布的颜色
        canvas = rgb_color * alpha + canvas * (1 - alpha)
        print(f"  - 已渲染第 {i + 1}/{len(gaussians)} 个椭球 (颜色: {g['color']})")

    # 4. 显示最终图像
    print("渲染完成，正在显示图像...")
    plt.figure(figsize=(8, 8))
    # imshow可以直接显示Numpy数组格式的图像
    plt.imshow(canvas)
    plt.axis('off')  # 关闭坐标轴，使图像更纯粹
    plt.title(f'可视化渲染: {json_path}')
    plt.show()



def automate_experiment_2():
    """
    自动化脚本，用于系统性地生成一系列用于实验二的JSON配置文件。
    本实验的核心是研究 'scale_multiplier' (椭球重叠度) 对结果的影响。
    """
    # --- 1. 定义实验参数 ---

    # 输入文件：定义两种材料区域的简单图像
    input_file = "Monica_2026-01-07_21-39-27.png"

    # 输出文件夹：所有生成的JSON文件将保存在这里
    output_folder = "experiment_2_outputs"

    # 固定的参数：在本次实验中，我们保持这些参数不变，以隔离变量
    K_COLORS = 2  # 两种材料
    COMPONENTS_PER_COLOR = [2, 3]  # 将每个区域分解为4个小椭球，以创建复杂的互锁边界

    # 核心变量：我们要测试的一系列尺度缩放因子
    # 0.9: 椭球略微收缩，边界清晰
    # 1.0: 基准状态，根据像素分布自然拟合
    # 1.2: 椭球放大20%，开始显著重叠
    # 1.5: 椭球放大50%，形成非常宽泛的高度混合区
    scale_multipliers_to_test = [0.9, 1.0, 1.2, 1.5]

    # --- 2. 执行自动化流程 ---

    # 如果输出文件夹不存在，则创建它
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"已创建输出文件夹: {output_folder}")

    print("\n" + "=" * 50)
    print("      开始自动化执行实验二：调控界面重叠度")
    print("=" * 50 + "\n")

    # 循环遍历每一个要测试的尺度因子
    for i, sm_value in enumerate(scale_multipliers_to_test):
        # 构造一个清晰的输出文件名，包含参数信息
        output_json_path = os.path.join(output_folder, f"config_scale_{sm_value:.2f}.json")

        print(f"--- 正在生成第 {i + 1}/{len(scale_multipliers_to_test)} 个配置文件 (scale = {sm_value}) ---")

        # 调用核心函数生成JSON文件
        generate_gaussian_field_from_image_v6(
            image_path=input_file,
            output_path=output_json_path,
            k_colors=K_COLORS,
            components_per_color=COMPONENTS_PER_COLOR,
            scale_multiplier=sm_value,  # <-- 在这里传入变量
            opacity_regulator=1.0,  # 保持其他参数不变
            visualize_results=False  # 在批量生成时关闭实时绘图
        )

        # (可选) 如果你想立即看到每个生成的场的模糊效果，可以取消下面的注释
        print(f"  -> 可视化渲染 {output_json_path}...")
        visualize_gaussian_field(output_json_path)

    print("\n" + "=" * 50)
    print("实验二的配置文件已全部生成！")
    print(f"所有文件均保存在 '{output_folder}' 文件夹中。")
    print("=" * 50)


def automate_experiment_3():
    """
    自动化脚本，用于系统性地生成一系列用于实验三的JSON配置文件。
    本实验的核心是研究 'opacity_regulator' (单元格偏好强度) 对结果的影响。
    """
    # --- 1. 定义实验参数 ---

    # 输入文件和输出文件夹保持不变
    input_file = "image.png"
    output_folder = "experiment_3_outputs"  # 为新实验创建新的输出文件夹

    # 固定的参数：保持几何形状的基准状态不变
    K_COLORS = 2
    COMPONENTS_PER_COLOR = [2, 3]
    SCALE_MULTIPLIER = 1.0  # <-- 固定尺度因子，这是与实验二的关键区别

    # 核心变量：我们要测试的一系列不透明度调节器
    # 1.2:  增强偏好，让边界更“确定”，过渡可能更陡峭
    # 1.0:  基准状态，不作干预
    # 0.8:  减弱偏好20%，给予WFC算法更多自由度，鼓励混合
    # 0.6:  显著减弱偏好，可能形成一个非常宽且随机的“软”融合区
    opacity_regulators_to_test = [1.2, 1.0, 0.8, 0.6]

    # --- 2. 执行自动化流程 ---

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"已创建输出文件夹: {output_folder}")

    print("\n" + "=" * 50)
    print("      开始自动化执行实验三：调控界面偏好强度")
    print("=" * 50 + "\n")

    # 循环遍历每一个要测试的不透明度调节器
    for i, op_value in enumerate(opacity_regulators_to_test):
        # 构造清晰的输出文件名
        output_json_path = os.path.join(output_folder, f"config_opacity_{op_value:.2f}.json")

        print(f"--- 正在生成第 {i + 1}/{len(opacity_regulators_to_test)} 个配置文件 (opacity_reg = {op_value}) ---")

        # 调用核心函数生成JSON文件
        generate_gaussian_field_from_image_v6(
            image_path=input_file,
            output_path=output_json_path,
            k_colors=K_COLORS,
            components_per_color=COMPONENTS_PER_COLOR,
            scale_multiplier=SCALE_MULTIPLIER,  # <-- 固定此参数
            opacity_regulator=op_value,  # <-- 在这里传入变量
            visualize_results=False
        )
        print(f"  -> 可视化渲染 {output_json_path}...")
        visualize_gaussian_field(output_json_path)

    print("\n" + "=" * 50)
    print("实验三的配置文件已全部生成！")
    print(f"所有文件均保存在 '{output_folder}' 文件夹中。")
    print("=" * 50)


# --- 运行主函数 ---
if __name__ == "__main__":
    automate_experiment_3()

# # --- 使用示例 ---
# input_file = "image.png"
# output_file = "generated_gaussian_field.json"
#
# # 预设 k_colors=2，表示要找2个主色调
# # 预设 components_per_color=[2, 3]，表示：
# # - 为第一个主色调拟合2个高斯分量（椭圆）
# # - 为第二个主色调拟合3个高斯分量（椭圆）
# generate_gaussian_field_from_image_v5(
#     input_file,
#     output_file,
#     k_colors=2,
#     components_per_color=[2, 3], # <<<<< 在这里设置每个颜色的分量数
#     visualize_results=True
# )
#
# # --- 使用示例 ---
# # 将 'generated_gaussian_field_v4_final.json' 替换为你的JSON文件名
# input_json_file = "generated_gaussian_field.json"
# visualize_gaussian_field(input_json_file)
# # 另一个例子：如果你想为3个主色调分别拟合1, 2, 3个分量
# # generate_gaussian_field_from_image_v5(
# #     input_file,
# #     output_file,
# #     k_colors=3,
# #     components_per_color=[1, 2, 3],
# #     visualize_results=True
# # )