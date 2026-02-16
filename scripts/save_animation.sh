#!/bin/bash
# 保存可视化动画脚本
# 将模型运行过程保存为GIF或MP4

# 参数
MODEL_PATH="${1:?用法: $0 <model_path> [output_file] [num_episodes]}"
OUTPUT="${2:-visualization_demo.gif}"
NUM_EPISODES="${3:-1}"
NUM_SHEEP="${4:-6}"
NUM_HERDERS="${5:-3}"
WORLD_SIZE_X="${6:-30.0}"
WORLD_SIZE_Y="${7:-30.0}"

# 检查模型文件是否存在
if [ ! -f "$MODEL_PATH" ]; then
    echo "错误: 模型文件不存在: $MODEL_PATH"
    exit 1
fi

# 检查imageio库
python -c "import imageio" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "警告: 需要安装imageio库才能保存动画"
    echo "运行: pip install imageio"
    exit 1
fi

echo "=========================================="
echo "保存可视化动画"
echo "=========================================="
echo "模型路径: $MODEL_PATH"
echo "输出文件: $OUTPUT"
echo "Episode数: $NUM_EPISODES"
echo "=========================================="

cd "$(dirname "$0")/.."

# 根据输出文件扩展名选择保存方式
if [[ "$OUTPUT" == *.gif ]]; then
    python visualize.py \
        --model_path "$MODEL_PATH" \
        --num_episodes $NUM_EPISODES \
        --num_sheep $NUM_SHEEP \
        --num_herders $NUM_HERDERS \
        --world_size $WORLD_SIZE_X $WORLD_SIZE_Y \
        --save_gif "$OUTPUT"
elif [[ "$OUTPUT" == *.mp4 ]]; then
    python visualize.py \
        --model_path "$MODEL_PATH" \
        --num_episodes $NUM_EPISODES \
        --num_sheep $NUM_SHEEP \
        --num_herders $NUM_HERDERS \
        --world_size $WORLD_SIZE_X $WORLD_SIZE_Y \
        --save_video "$OUTPUT"
else
    echo "错误: 不支持的输出格式，请使用 .gif 或 .mp4"
    exit 1
fi

echo "动画已保存到: $OUTPUT"
