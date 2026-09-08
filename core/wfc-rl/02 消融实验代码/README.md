# RL-WFC 消融实验代码

本文件夹包含用于说明 RL-WFC 方法设计选择的关键对照脚本。

- `01_two_head_policy_ablation.py`：两步动作/双头策略网络版本，用于和其他动作空间设计对照。
- `02_flat_action_space_ablation.py`：扁平动作空间/单输出头版本，用于动作空间消融。
- `03_conditional_input_ablation.py`：带目标刚度条件输入的版本，用于条件化输入对照。
- `04_random_search_baseline.py`：随机搜索/贪心基线，用于非 RL 对照。
- `05_ablation_result_plot.py`：消融实验结果曲线绘制脚本。

这些脚本共享上一级目录中的 WFC-RL 依赖模块。
