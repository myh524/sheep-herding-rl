#!/usr/bin/env bash
# 与 scripts/visualize.sh 一样先 cd 到子项目根目录，避免 sys.path / 相对路径错乱
export INFORMARL_MPL_BLOCK_END=1
cd "$(dirname "$0")"

# 与仓库根目录 scripts/visualize.sh 一致：不默认 MPLBACKEND=TkAgg（无 python3-tk 时会整段回退 Agg 无窗口）。
# 需要指定后端时再: export MPLBACKEND=Qt5Agg

# 入口见 visualize.py（默认 Matplotlib；与根目录 visualize.py 参数风格对齐）
python visualize.py \
  --model_dir "/home/standard/workspace/sheep-herding-rl/InforMARL/onpolicy/results/GraphMPE/navigation_graph/rmappo/informarl/run5/models" \
  --num_agents 5 \
  --num_obstacles 3 \
  --seed 0 \
  --render_delay 10
