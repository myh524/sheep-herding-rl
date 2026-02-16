#!/bin/bash
# ============================================================================
# Fast Training Script - CPU Optimized (Improved Version)
# ============================================================================
# Features:
#   - Reduced training steps (500K)
#   - Improved hyperparameters
#   - Smaller network for faster training
#   - Curriculum learning recommended
# ============================================================================

python train_ppo.py \
    --env_name sheep_herding \
    --scenario_name fast_train \
    --num_sheep 6 \
    --num_herders 3 \
    --world_size 30.0 30.0 \
    --episode_length 100 \
    --seed 1 \
    --num_env_steps 500000 \
    --lr 3e-4 \
    --ppo_epoch 10 \
    --num_mini_batch 4 \
    --clip_param 0.2 \
    --value_loss_coef 0.5 \
    --entropy_coef 0.01 \
    --gamma 0.99 \
    --gae_lambda 0.95 \
    --max_grad_norm 0.5 \
    --hidden_size 128 \
    --layer_N 2 \
    --log_interval 50 \
    --save_interval 200 \
    --use_ReLU \
    --use_orthogonal \
    --use_feature_normalization \
    --use_gae \
    --use_linear_lr_decay \
    --use_clipped_value_loss \
    --use_reward_normalization \
    2>&1
