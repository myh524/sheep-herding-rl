#!/bin/bash
# ============================================================================
# Sheep Herding RL Training Script (Improved Version)
# ============================================================================
# Usage:
#   1. Direct run: bash scripts/train.sh
#   2. Modify parameters below and run
# ============================================================================

# ============================================================================
# Part 1: Environment Basic Parameters
# ============================================================================

ENV_NAME="sheep_herding"
SCENARIO_NAME="default"

# Number of sheep (recommended: 5-20)
NUM_SHEEP=6

# Number of herders (recommended: 2-5)
NUM_HERDERS=3

# Episode length (recommended: 100-200)
EPISODE_LENGTH=150

# World size [width, height] (meters)
WORLD_SIZE="30.0 30.0"

# ============================================================================
# Part 2: Training Mode Selection
# ============================================================================

# Mode 1: Standard training (fixed parameters)
USE_CURRICULUM=false
USE_RANDOMIZED=false

# Mode 2: Curriculum learning (recommended for new training)
# - Start from simple scenario, gradually increase difficulty
# USE_CURRICULUM=true
# USE_RANDOMIZED=false

# Mode 3: Randomized training (for generalization)
# USE_CURRICULUM=false
# USE_RANDOMIZED=true

# Curriculum learning start stage
START_STAGE=0

# ============================================================================
# Part 3: PPO Hyperparameters (Improved)
# ============================================================================

# Total training steps (recommended: 1M-5M)
NUM_ENV_STEPS=2000000

# Learning rate (improved: 5e-4 -> 3e-4)
LR=3e-4

# PPO epochs (improved: 15 -> 10)
PPO_EPOCH=10

# Mini-batch count (improved: 1 -> 4)
NUM_MINI_BATCH=4

# PPO clip parameter
CLIP_PARAM=0.2

# Value loss coefficient (improved: 1.0 -> 0.5)
VALUE_LOSS_COEF=0.5

# Entropy coefficient
ENTROPY_COEF=0.01

# Discount factor
GAMMA=0.99

# GAE lambda
GAE_LAMBDA=0.95

# Gradient clipping (improved: 10.0 -> 0.5)
MAX_GRAD_NORM=0.5

# ============================================================================
# Part 4: Network Architecture (Improved)
# ============================================================================

# Hidden layer size (improved: 64 -> 256)
HIDDEN_SIZE=256

# Network layers (improved: 1 -> 3)
LAYER_N=3

USE_RELU=true
USE_ORTHOGONAL=true
USE_FEATURE_NORMALIZATION=true

# ============================================================================
# Part 5: Advanced Training Options
# ============================================================================

USE_POPART=false
USE_VALUENORM=false
USE_GAE=true
USE_LINEAR_LR_DECAY=true
USE_CLIPPED_VALUE_LOSS=true

# Reward normalization (new feature)
USE_REWARD_NORMALIZATION=true

# ============================================================================
# Part 6: Logging and Saving
# ============================================================================

SEED=1
LOG_INTERVAL=10
SAVE_INTERVAL=100
EXPERIMENT_NAME=""
USE_WANDB=false
WANDB_PROJECT="sheep_herding"

# ============================================================================
# Part 7: Build Training Command
# ============================================================================

CMD="python train_ppo.py \
    --env_name ${ENV_NAME} \
    --scenario_name ${SCENARIO_NAME} \
    --num_sheep ${NUM_SHEEP} \
    --num_herders ${NUM_HERDERS} \
    --episode_length ${EPISODE_LENGTH} \
    --world_size ${WORLD_SIZE} \
    --seed ${SEED} \
    --num_env_steps ${NUM_ENV_STEPS} \
    --lr ${LR} \
    --ppo_epoch ${PPO_EPOCH} \
    --num_mini_batch ${NUM_MINI_BATCH} \
    --clip_param ${CLIP_PARAM} \
    --value_loss_coef ${VALUE_LOSS_COEF} \
    --entropy_coef ${ENTROPY_COEF} \
    --gamma ${GAMMA} \
    --gae_lambda ${GAE_LAMBDA} \
    --max_grad_norm ${MAX_GRAD_NORM} \
    --hidden_size ${HIDDEN_SIZE} \
    --layer_N ${LAYER_N} \
    --log_interval ${LOG_INTERVAL} \
    --save_interval ${SAVE_INTERVAL}"

if [ "$USE_WANDB" = true ]; then
    CMD="${CMD} --use_wandb --wandb_project ${WANDB_PROJECT}"
fi

if [ "$USE_RELU" = true ]; then
    CMD="${CMD} --use_ReLU"
fi

if [ "$USE_ORTHOGONAL" = true ]; then
    CMD="${CMD} --use_orthogonal"
fi

if [ "$USE_FEATURE_NORMALIZATION" = true ]; then
    CMD="${CMD} --use_feature_normalization"
fi

if [ "$USE_POPART" = true ]; then
    CMD="${CMD} --use_popart"
fi

if [ "$USE_VALUENORM" = true ]; then
    CMD="${CMD} --use_valuenorm"
fi

if [ "$USE_GAE" = true ]; then
    CMD="${CMD} --use_gae"
fi

if [ "$USE_LINEAR_LR_DECAY" = true ]; then
    CMD="${CMD} --use_linear_lr_decay"
fi

if [ "$USE_CLIPPED_VALUE_LOSS" = true ]; then
    CMD="${CMD} --use_clipped_value_loss"
fi

if [ "$USE_REWARD_NORMALIZATION" = true ]; then
    CMD="${CMD} --use_reward_normalization"
fi

if [ "$USE_CURRICULUM" = true ]; then
    CMD="${CMD} --use_curriculum --start_stage ${START_STAGE}"
fi

if [ "$USE_RANDOMIZED" = true ]; then
    CMD="${CMD} --use_randomized"
fi

if [ -n "$EXPERIMENT_NAME" ]; then
    CMD="${CMD} --experiment_name ${EXPERIMENT_NAME}"
fi

# ============================================================================
# Part 8: Execute Training
# ============================================================================

echo "=========================================="
echo "Starting Training (Improved Version)"
echo "=========================================="
echo "Mode: $(if [ "$USE_CURRICULUM" = true ]; then echo 'Curriculum Learning'; elif [ "$USE_RANDOMIZED" = true ]; then echo 'Randomized Training'; else echo 'Standard Training'; fi)"
echo "Sheep: ${NUM_SHEEP}, Herders: ${NUM_HERDERS}"
echo "World: ${WORLD_SIZE}"
echo "Steps: ${NUM_ENV_STEPS}, LR: ${LR}"
echo "Hidden: ${HIDDEN_SIZE}, Layers: ${LAYER_N}"
echo "=========================================="
echo ""

eval ${CMD}

echo ""
echo "=========================================="
echo "Training Complete!"
echo "=========================================="
