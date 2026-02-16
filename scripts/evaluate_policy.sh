#!/bin/bash
# 策略评估脚本
# 运行多个episode并输出性能统计

# 默认参数
MODEL_PATH="${1:-results/sheep_herding/default/ppo/seed1/models/model_1000.pt}"
NUM_EPISODES="${2:-100}"
NUM_SHEEP="${3:-6}"
NUM_HERDERS="${4:-3}"
WORLD_SIZE_X="${5:-30.0}"
WORLD_SIZE_Y="${6:-30.0}"
VERBOSE="${7:---verbose}"

# 检查模型文件是否存在
if [ ! -f "$MODEL_PATH" ]; then
    echo "错误: 模型文件不存在: $MODEL_PATH"
    echo "用法: $0 <model_path> [num_episodes] [num_sheep] [num_herders] [world_size_x] [world_size_y] [--verbose|--no-verbose]"
    exit 1
fi

echo "=========================================="
echo "策略性能评估"
echo "=========================================="
echo "模型路径: $MODEL_PATH"
echo "评估Episode数: $NUM_EPISODES"
echo "羊数量: $NUM_SHEEP"
echo "机械狗数量: $NUM_HERDERS"
echo "世界大小: ${WORLD_SIZE_X} x ${WORLD_SIZE_Y}"
echo "详细模式: $VERBOSE"
echo "=========================================="

cd "$(dirname "$0")/.."

python evaluate_policy.py \
    --model_path "$MODEL_PATH" \
    --num_episodes $NUM_EPISODES \
    --num_sheep $NUM_SHEEP \
    --num_herders $NUM_HERDERS \
    --world_size $WORLD_SIZE_X $WORLD_SIZE_Y \
    $VERBOSE
