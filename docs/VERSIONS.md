# 版本与归档范围

## v2026.09.08

2026-09-08 整理的研究归档快照；版本号表示本次整理日期，并非原软件发布版本，也不表示原始开发先后顺序。源目录未提供 Git 历史，未重建或虚构历史版本。

- `core/wfc-rl`：主入口 `GPU_hope.py`，保留均质化、位置约束、WFC、可视化和五份消融/对照脚本。
- `core/wfc-gaussian`：保留三个方法脚本、可视化模块、两张示例输入图和两份 JSON 配置。
- `toolbox/app`：英文界面、PyInstaller 资源路径适配与自动打开浏览器的启动逻辑。
- `toolbox/web`：中文操作说明与界面，通过本地 Gradio 服务提供可视化。

APP 与网页版本的 `wfc_toolbox` 模块内容一致，`app.py` 存在界面文字、布局和启动方式差异；为保留两套可独立使用的原始版本，分别完整存放。原始文件校验、加入版权注释后的归档校验及来源相对目录见 `source-manifest.json`。

## Windows 程序附件

`WFC_Lattice_Designer-Windows-v2026.09.08.zip` 是原 APP 发布文件夹的完整归档。解压后运行 `WFC_Lattice_Designer/WFC_Lattice_Designer.exe`，保留同目录 `_internal` 文件夹。附件保留已有构建，未重新编译，未执行启动验证；归档日期不代表二进制编译日期。

## 运行提示

- 在对应核心代码目录运行脚本；实验参数、循环次数和输出路径位于脚本内，运行前按实验需求检查。
- 消融脚本导入上级目录模块，需让 `core/wfc-rl` 位于 `PYTHONPATH`。
- WFC-RL 的第三方依赖包括 numpy、scipy、matplotlib、torch、tqdm、pandas；原资料未提供精确锁定版本。
- WFC-Gaussian 的第三方依赖包括 numpy、matplotlib、opencv-python、scikit-learn、trimesh、shapely；STL 三角化可能需安装相应可选后端。原资料未提供精确锁定版本。
- 工具箱保留原 `requirements.txt`；版本区间不等同于已验证的环境锁文件。
- 本次执行源码语法、文件完整性与明显凭据模式检查；不将其等同于完整训练、论文结果复现或界面/STL 全流程验证。

## 保留与排除

源码、同目录模块、输入样例、原依赖声明和运行脚本均保留。Windows 运行库单独放在程序附件中。论文全文、展示文稿、展示 PDF、重复总压缩包不纳入精简代码仓库。原始 NAS 文件保持不变。

使用根目录 `LICENSE` 中的专有版权声明，保留所有权利；仓库现已按作者要求改为公开，供简历链接访问；版权声明仍然适用。既有归档附件中的“私有”描述反映归档时状态，不代表当前访问限制。所有 Python、JavaScript 和批处理源码均新增版权注释，算法逻辑保持原样。第三方运行库保留原许可证。Windows 附件增加同一版权声明，已有二进制内容未修改。
