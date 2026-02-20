#!/bin/bash
# Training Analysis Script

LOG_DIR="results/sheep_herding/default/ppo/seed1/xxx"
SMOOTH_WINDOW=10

echo "=========================================="
echo "Training Analysis"
echo "=========================================="
echo "Log directory: $LOG_DIR"
echo "Smooth window: $SMOOTH_WINDOW"
echo "=========================================="

python -m tools.analyze_training \
    --log_dir "$LOG_DIR" \
    --smooth $SMOOTH_WINDOW

echo "=========================================="
echo "Analysis completed!"
echo "=========================================="
