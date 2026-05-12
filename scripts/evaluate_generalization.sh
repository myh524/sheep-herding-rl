#!/bin/bash
# 泛化评估：在羊只数 (5,10,15,20,25) × 机械狗数 (3,4,5,6) 网格上评估同一 checkpoint。
# 输出成功率、完成步数、结束时羊群扩散度等；可选写入 JSON。
#
# 用法:
#   bash scripts/evaluate_generalization.sh <model_path> [num_episodes] [output_json]
#
# 默认与训练对齐（Python 内建）：无圆盘边界、--high_level_interval 3、势场运动学（不传 herder_teleport）。
#
# 环境变量（可选，传给 evaluate_generalization.py）:
#   GEN_SEED=0              传给 --seed
#   GEN_PER_COMBO_SEED=1    设置后加 --per_combo_seed（每格 seed+羊*1000+狗）
#   GEN_RUN_FULL_HORIZON=1  设置后加 --run_full_horizon（不因到达目标提前结束）
#   GEN_FORMATION_DELTA=1   强制加 --formation_delta（通常不必；Python 会从 ckpt 自动对齐 10/13 维）
#   GEN_DISK_BOUNDARY=1     加 --disk-boundary（启用圆盘约束；默认不启用）
#   GEN_NO_DISK_BOUNDARY=1  加 --no-disk-boundary（显式兼容参数，与默认等价）
#   GEN_HERDER_TELEPORT=1   加 --herder_teleport（默认不加，保持势场动力学）
#   GEN_HIGH_LEVEL_INTERVAL=N  覆盖默认 3
#   GEN_OUTPUT_FIGURES=figures/gen_plots  折线图输出目录（传 --output_figures）
#   GEN_FIGURE_PREFIX=run1_       PNG 文件名前缀（传 --figure_prefix）

set -euo pipefail

MODEL_PATH="${1:-results/sheep_herding/gpu_train/ppo/seed1/20260410_171812/models/model_3602400.pt}"
MODEL_PATH="${1:-results/sheep_herding/gpu_train/ppo/seed1/20260511_112928/models/model_8402400.pt}"
NUM_EPISODES="${2:-50}"
OUTPUT_JSON="${3:-}"

if [ -z "$MODEL_PATH" ]; then
    echo "错误: 未指定模型路径。"
    echo "用法: $0 <model_path> [num_episodes] [output_json]"
    exit 1
fi

if [ ! -f "$MODEL_PATH" ]; then
    echo "错误: 模型文件不存在: $MODEL_PATH"
    echo "用法: $0 <model_path> [num_episodes] [output_json]"
    exit 1
fi

cd "$(dirname "$0")/.."

EXTRA=()
if [ -n "${GEN_SEED:-}" ]; then
    EXTRA+=(--seed "$GEN_SEED")
fi
if [ -n "${GEN_PER_COMBO_SEED:-}" ]; then
    EXTRA+=(--per_combo_seed)
fi
if [ -n "${GEN_RUN_FULL_HORIZON:-}" ]; then
    EXTRA+=(--run_full_horizon)
fi
if [ -n "${GEN_FORMATION_DELTA:-}" ]; then
    EXTRA+=(--formation_delta)
fi
if [ -n "${GEN_DISK_BOUNDARY:-}" ]; then
    EXTRA+=(--disk-boundary)
fi
if [ -n "${GEN_NO_DISK_BOUNDARY:-}" ]; then
    EXTRA+=(--no-disk-boundary)
fi
if [ -n "${GEN_HERDER_TELEPORT:-}" ]; then
    EXTRA+=(--herder_teleport)
fi
if [ -n "${GEN_HIGH_LEVEL_INTERVAL:-}" ]; then
    EXTRA+=(--high_level_interval "$GEN_HIGH_LEVEL_INTERVAL")
fi
if [ -n "${GEN_OUTPUT_FIGURES:-}" ]; then
    EXTRA+=(--output_figures "$GEN_OUTPUT_FIGURES")
fi
if [ -n "${GEN_FIGURE_PREFIX:-}" ]; then
    EXTRA+=(--figure_prefix "$GEN_FIGURE_PREFIX")
fi

OUT_ARGS=()
if [ -n "$OUTPUT_JSON" ]; then
    OUT_ARGS+=(--output_json "$OUTPUT_JSON")
fi

echo "=========================================="
echo "泛化网格评估 (evaluate_generalization.py)"
echo "=========================================="
echo "模型: $MODEL_PATH"
echo "每格 episodes: $NUM_EPISODES"
if [ -n "$OUTPUT_JSON" ]; then
    echo "JSON 输出: $OUTPUT_JSON"
else
    echo "JSON 输出: (未指定，仅打印终端)"
fi
echo "=========================================="

python3 evaluate_generalization.py \
    --model_path "$MODEL_PATH" \
    --num_episodes "$NUM_EPISODES" \
    --episode_length 250
    --no-disk-boundary \
    "${OUT_ARGS[@]}" \
    "${EXTRA[@]}"
    # --herder_teleport