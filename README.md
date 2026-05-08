# Sheep Herding RL

羊群引导强化学习：高层编队策略 + 可选 **MAPPO 低层导航**（[`InforMARL/`](InforMARL/)）对接。

## 文档

- **[分层环境与 MAPPO 低层对接：修改说明与使用教程（中文）](docs/hierarchical_mappo_integration_zh.md)** — 包含全部改动文件列表、参数说明、命令行/Python 示例、FAQ 与训练阶段建议。

## 快速开始（高层 + 低层动力学）

需安装 PyTorch；低层权重目录中应有 `actor.pt`（或与训练一致的 checkpoint）。

```bash
python train_hierarchical.py \
  --low-level-model-dir /path/to/mappo_run/models \
  --num-herders 3 --world-size 100 100 \
  # ... 其余参数同 train_ppo.py
```

仅训练高层、不用低层时，不传 `--low-level-model-dir` 即可，行为与原先一致。

## 子项目

- [`InforMARL/`](InforMARL/)：Graph MAPPO / InforMARL 导航训练代码；经主仓库 patch 支持 `--external_goals` 与宿主写入 landmark。
