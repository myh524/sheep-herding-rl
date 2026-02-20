#!/bin/bash
# Quick Training Script for Testing (Short run with discrete actions)

echo "=========================================="
echo "Quick Training Test (Discrete Actions)"
echo "=========================================="

python train_ppo.py \
    --num_sheep 5 \
    --num_herders 3 \
    --episode_length 30 \
    --num_env_steps 3000 \
    --seed 1 \
    --hidden_size 64 \
    --discrete_actions \
    --lr 3e-4 \
    --ppo_epoch 3 \
    --num_mini_batch 2 \
    --log_interval 1 \
    --save_interval 1000

echo "=========================================="
echo "Quick test completed!"
echo "=========================================="
