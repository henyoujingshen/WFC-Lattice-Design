# WFC-Gaussian 核心代码

## 文件说明

- `01_gaussian_field_from_image.py`：从图片生成 Gaussian field 配置文件。
- `02_wfc_gaussian_stl.py`：基于 Gaussian/WFC 结果生成 2.5D STL 结构。
- `03_stiffness_gradient_stl.py`：基于三刚度梯度配置生成带夹持结构的 STL。
- `visualize_new_library.py`：WFC-Gaussian 结果可视化辅助函数。
- `image.png`、`image1.png`：Gaussian field 示例输入图片。
- `saveTheRabbit.json`：`02_wfc_gaussian_stl.py` 的示例输入配置。
- `stiffness_experiments/2_1_interlocking_gradient.json`：由输入图色块拟合 GaussianMixture 后得到的 WFC 示例输入配置。

## 运行依赖

代码依赖的第三方库主要包括 `numpy`、`matplotlib`、`opencv-python`、`scikit-learn`、`trimesh`、`shapely`。
