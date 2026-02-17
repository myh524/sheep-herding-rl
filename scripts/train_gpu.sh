#!/bin/bash
# ============================================================================
# GPU Accelerated Training Script (Improved Version)
# ============================================================================
# Features:
#   - Auto GPU detection
#   - Larger network (256 hidden, 3 layers)
#   - Curriculum learning enabled (auto-manages sheep/herders count)
#   - Reward normalization
#   - Improved hyperparameters
# ============================================================================
# Curriculum Learning Stages (auto-managed):
#   Stage 0: 3 sheep, 3 herders, 60x60 world
#   Stage 1: 5 sheep, 3 herders, 60x60 world
#   Stage 2: 10 sheep, 3 herders, 60x60 world (target)
# ============================================================================

python train_ppo.py \
    --env_name sheep_herding \
    --scenario_name gpu_train \
    --use_curriculum \
    --start_stage 0 \
    --seed 1 \
    --num_env_steps 3000000 \
    --n_rollout_threads 4 \
    --lr 3e-4 \
    --ppo_epoch 10 \
    --num_mini_batch 4 \
    --clip_param 0.2 \
    --value_loss_coef 0.5 \
    --entropy_coef 0.05 \
    --gamma 0.99 \
    --gae_lambda 0.95 \
    --max_grad_norm 0.5 \
    --hidden_size 256 \
    --layer_N 3 \
    --log_interval 100 \
    --save_interval 500 \
    --use_ReLU \
    --use_orthogonal \
    --use_feature_normalization \
    --use_gae \
    --use_linear_lr_decay \
    --use_clipped_value_loss \
    --use_reward_normalization \
    2>&1
