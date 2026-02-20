#!/bin/bash
# 可视化训练过程脚本

echo "=========================================="
echo "可视化训练过程"
echo "=========================================="
echo "目的: 演示训练过程，不是真正训练"
echo "总步数: 3000 (约100个episodes)"
echo "=========================================="

python visualize_training.py \
    --num_sheep 2 \
    --num_herders 3 \
    --episode_length 30 \
    --num_env_steps 3000 \
    --seed 1 \
    --hidden_size 128 \
    --layer_N 2 \
    --lr 3e-4 \
    --ppo_epoch 3 \
    --num_mini_batch 2 \
    --render_interval 1 \
    --update_interval 1

echo "=========================================="
echo "可视化训练完成！"
echo "=========================================="
