#!/bin/bash
# 可视化运行脚本
# 加载训练好的模型并实时渲染环境

# 默认参数
MODEL_PATH="${1:-results/sheep_herding/default/ppo/seed1/models/model_1000.pt}"
NUM_EPISODES="${2:-5}"
NUM_SHEEP="${3:-6}"
NUM_HERDERS="${4:-3}"
WORLD_SIZE_X="${5:-30.0}"
WORLD_SIZE_Y="${6:-30.0}"

# 检查模型文件是否存在
if [ ! -f "$MODEL_PATH" ]; then
    echo "错误: 模型文件不存在: $MODEL_PATH"
    echo "用法: $0 <model_path> [num_episodes] [num_sheep] [num_herders] [world_size_x] [world_size_y]"
    exit 1
fi

echo "=========================================="
echo "羊群引导模型可视化"
echo "=========================================="
echo "模型路径: $MODEL_PATH"
echo "Episode数: $NUM_EPISODES"
echo "羊数量: $NUM_SHEEP"
echo "机械狗数量: $NUM_HERDERS"
echo "世界大小: ${WORLD_SIZE_X} x ${WORLD_SIZE_Y}"
echo "=========================================="

cd "$(dirname "$0")/.."

python visualize.py \
    --model_path "$MODEL_PATH" \
    --num_episodes $NUM_EPISODES \
    --num_sheep $NUM_SHEEP \
    --num_herders $NUM_HERDERS \
    --world_size $WORLD_SIZE_X $WORLD_SIZE_Y \
    --render_delay 50
