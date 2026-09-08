# WFC Lattice Design

硕士阶段非周期混杂点阵结构生成研究与设计工具箱。

**公开展示 · 保留所有权利。** 未经书面许可，禁止使用、运行、调用、转载、修改、再分发及商业使用；学术及非商业用途亦须授权。公开访问及 GitHub 平台条款允许的浏览、Fork 不等于获得其他使用授权。第三方组件适用其原许可证，完整范围与例外见 [LICENSE](LICENSE)。

- **WFC-Gaussian**：高斯场引导的波函数坍缩生成、梯度点阵与 STL 导出。
- **WFC-RL**：强化学习结合波函数坍缩，面向目标等效刚度的结构设计，含消融与对照实验。
- **设计工具箱**：保留英文 APP 源码版和中文网页可视化版。

| 内容 | 目录 |
| --- | --- |
| WFC-RL 核心代码与消融实验 | [core/wfc-rl](core/wfc-rl) |
| WFC-Gaussian 核心代码与示例 | [core/wfc-gaussian](core/wfc-gaussian) |
| APP 版本源码 | [toolbox/app](toolbox/app) |
| 网页可视化版本源码 | [toolbox/web](toolbox/web) |

## 启动工具箱

以下运行说明仅供权利人及已获得相应书面使用授权的人员使用。在独立 Python 环境中，从仓库根目录运行：

```bash
cd toolbox/web
python -m pip install -r requirements.txt
python app.py
```

浏览器访问 `http://127.0.0.1:7860`；若端口占用，查看终端提示。APP 源码版将目录换为 `toolbox/app`。两个版本均由本地 Python/Gradio 服务运行。

原始依赖声明与算法逻辑保留；源码仅增加版权注释，尚未完成运行环境兼容性与端到端验证。核心脚本入口和实验说明见各目录 README；归档范围和版本差异见 [版本说明](docs/VERSIONS.md)。

研究对应：唐呼博成，南方科技大学，2026，硕士学位论文《基于波函数坍缩的非周期混杂点阵结构生成方法》。本仓库归档研究代码，不包含论文全文和答辩材料。
