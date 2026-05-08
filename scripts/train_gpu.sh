#!/bin/bash
# ============================================================================
# GPU Accelerated Training Script (Improved Version)
# ============================================================================
# 注意：续行符 \ 的下一行不能以 # 开头，否则 Bash 会在此处结束命令，后面参数不会传给 Python。
# Features:
#   - Auto GPU detection
#   - Larger network (256 hidden, 3 layers)
#   - Curriculum learning enabled (auto-manages sheep/herders count)
#   - Reward normalization
#   - Improved hyperparameters
# ============================================================================
# Curriculum 阶段在代码里定义：envs/defaults.py → DEFAULT_CURRICULUM_STAGE_SPECS
# （默认）各阶段 world 100×100、ep_len 150；dt 与 SheepFlockEnv 默认一致（2.0）
#   --herder_teleport: 狗直接到编队目标，无运动学（先看队形；低层运动后续再训）
# ============================================================================

# 使用 --device cuda：若 PyTorch 无法初始化 CUDA 会立即报错，避免长时间在 CPU 上静默训练。
# 稠密奖励见 envs/sheep_flock.py _compute_reward（reward_config 可调）。
# 梯度仍异常时可：略降 --lr（如 1e-4）、减小 --skip_update_grad_norm、或 --value_huber_beta 20
python3 train_ppo.py \
    --device cuda \
    --env_name sheep_herding \
    --scenario_name gpu_train \
    --herder_teleport \
    --use_curriculum \
    --start_stage 0 \
    --seed 1 \
    --num_env_steps 10000000 \
    --n_rollout_threads 8 \
    --lr 3e-4 \
    --ppo_epoch 10 \
    --num_mini_batch 4 \
    --clip_param 0.2 \
    --value_loss_coef 0.5 \
    --entropy_coef 0.05 \
    --gamma 0.99 \
    --gae_lambda 0.95 \
    --max_grad_norm 0.5 \
    --ppo_log_ratio_clip 20 \
    --skip_update_grad_norm 200 \
    --value_huber_beta 10 \
    --hidden_size 512 \
    --layer_N 3 \
    --log_interval 100 \
    --use_tensorboard \
    --save_interval 50 \
    --use_ReLU \
    --use_orthogonal \
    --use_feature_normalization \
    --use_gae \
    --use_linear_lr_decay \
    --use_clipped_value_loss \
    --use_reward_normalization \
    --formation_delta 2>&1
