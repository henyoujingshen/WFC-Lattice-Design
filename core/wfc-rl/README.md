# WFC-RL 核心代码

## 入口文件

`GPU_hope.py`

## 文件说明

- `GPU_hope.py`：WFC-RL 主流程代码。
- `WFC_library_undirected.py`：WFC 约束传播、网格生成和相关计算函数。
- `Position_constrain.py`：位置约束与约束异常定义。
- `Homoge.py`、`global_stiffness.py`：等效刚度与有限元相关计算。
- `visualize_elastic_module.py`、`visl.py`：结果可视化辅助函数。
- `02 消融实验代码`：RL-WFC 对照/消融实验脚本。

## 运行依赖

代码依赖的第三方库主要包括 `numpy`、`scipy`、`matplotlib`、`torch`、`tqdm`。
