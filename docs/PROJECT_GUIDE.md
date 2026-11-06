# WFC Lattice Design — Repository Guide

Research code and design tools for non-periodic hybrid lattice structures using wave function collapse (WFC), Gaussian fields, and reinforcement learning.

This repository accompanies Tang Hubocheng's 2026 master's research at Southern University of Science and Technology, titled 《基于波函数坍缩的非周期混杂点阵结构生成方法》. The thesis text and defense materials are not included. See the [main README](../README.md) for the Chinese overview.

## Start with your area of interest

| Area | What is included | Entry point |
| --- | --- | --- |
| Reinforcement learning for lattice design | WFC generation, target equivalent stiffness, homogenization utilities, and comparison/ablation scripts | [WFC-RL guide](../core/wfc-rl/README.md); `GPU_hope.py` |
| Gaussian-field-guided design | Image-to-field configuration, WFC generation, gradient lattices, and STL export | [WFC-Gaussian guide](../core/wfc-gaussian/README.md) |
| Interactive design toolbox | Chinese interface for field editing, lattice visualization, and STL export | [Web toolbox](../toolbox/web/README.md); `app.py` |
| English interface / APP source | English interface with packaged-application resource handling | [APP source](../toolbox/app/README.md); `app.py` |

Both toolbox variants use a local Python/Gradio service. The APP directory contains source code; it is not a separate native implementation. The two versions preserve their original interface and launch differences. See [version and archive notes](VERSIONS.md) for details.

## Reading the research code

- **WFC-RL:** Start with `core/wfc-rl/GPU_hope.py`. Local modules separate constraint propagation, position constraints, equivalent stiffness calculations, and visualization. The `02 消融实验代码` folder contains policy/action-space comparisons, a conditional-input comparison, a random-search baseline, and a plotting script.
- **WFC-Gaussian:** `01_gaussian_field_from_image.py` builds a field configuration from an image. `02_wfc_gaussian_stl.py` and `03_stiffness_gradient_stl.py` contain separate structure-generation/export workflows. Example images and JSON configurations are included in the same directory.
- **Toolbox:** Each `app.py` connects the interface to its local `wfc_toolbox` modules. The corresponding `requirements.txt` records the original dependency ranges.

## Environment and reproducibility

For the rights holder and users with the required written authorization, installation and launch instructions are in the [main README](../README.md). Review script parameters, output paths, and the working directory before running experiments. The ablation scripts also require `core/wfc-rl` to be available on `PYTHONPATH`.

The archived dependency ranges are not a tested environment lockfile. Exact environments for the core research scripts are not pinned. Full training, thesis-result reproduction, and end-to-end interface/STL workflows have not been validated by this documentation update. The existing [version notes](VERSIONS.md) explain the validation scope and optional STL triangulation dependencies.

## Rights and attribution

**Publicly viewable; all rights reserved.** This repository is not released under an open-source license. Public availability does not grant permission to run, reuse, modify, or redistribute its original code or materials. Academic and non-commercial uses also require prior written permission. Third-party materials remain subject to their own licenses. The authoritative scope and exceptions are set out in [LICENSE](../LICENSE).

