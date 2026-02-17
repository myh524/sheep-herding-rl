#!/bin/bash
# 可视化运行脚本
# 加载训练好的模型并实时渲染环境
# 默认参数
MODEL_PATH="${1:-results/sheep_herding/gpu_train/ppo/seed1/20260217_115051/models/model_225150.pt}"

# 检查模型文件是否存在
if [ ! -f "$MODEL_PATH" ]; then
    echo "错误: 模型文件不存在: $MODEL_PATH"
    echo "用法: $0 <model_path> [num_episodes] [num_sheep] [num_herders] [world_size_x] [world_size_y]"
    exit 1
fi

cd "$(dirname "$0")/.."
python visualize.py \
    --model_path "$MODEL_PATH" \
    --num_episodes 5 \
    --num_sheep 30 \
    --num_herders 5 \
    --world_size 60 60 \
    --episode_length 150 \
    --render_delay 50
