#!/bin/bash
# 训练分析脚本
# 解析训练日志并绘制训练曲线

# 默认参数
LOG_DIR="${1:-results/sheep_herding/default/ppo/seed1}"
SMOOTH="${2:-10}"
OUTPUT="${3:-}"

echo "=========================================="
echo "训练日志分析"
echo "=========================================="
echo "日志目录: $LOG_DIR"
echo "平滑窗口: $SMOOTH"
echo "=========================================="

cd "$(dirname "$0")/.."

if [ -n "$OUTPUT" ]; then
    python analyze_training.py \
        --log_dir "$LOG_DIR" \
        --smooth $SMOOTH \
        --output "$OUTPUT"
else
    python analyze_training.py \
        --log_dir "$LOG_DIR" \
        --smooth $SMOOTH
fi
