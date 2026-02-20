#!/bin/bash
# PPO Training Script for Sheep Herding (Simplified Version)

# Default parameters
NUM_SHEEP=2
NUM_HERDERS=3
EPISODE_LENGTH=35
NUM_ENV_STEPS=200000
SEED=1
HIDDEN_SIZE=128
ACTION_REPEAT=5

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --num_sheep)
            NUM_SHEEP="$2"
            shift 2
            ;;
        --num_herders)
            NUM_HERDERS="$2"
            shift 2
            ;;
        --episode_length)
            EPISODE_LENGTH="$2"
            shift 2
            ;;
        --num_env_steps)
            NUM_ENV_STEPS="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        --hidden_size)
            HIDDEN_SIZE="$2"
            shift 2
            ;;
        --action_repeat)
            ACTION_REPEAT="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "=========================================="
echo "PPO Training for Sheep Herding (Simplified)"
echo "=========================================="
echo "Number of sheep: $NUM_SHEEP"
echo "Number of herders: $NUM_HERDERS"
echo "Episode length: $EPISODE_LENGTH"
echo "Total steps: $NUM_ENV_STEPS"
echo "Seed: $SEED"
echo "Hidden size: $HIDDEN_SIZE"
echo "Action repeat: $ACTION_REPEAT"
echo "=========================================="

# Build command
CMD="python train_ppo.py \
    --num_sheep $NUM_SHEEP \
    --num_herders $NUM_HERDERS \
    --episode_length $EPISODE_LENGTH \
    --num_env_steps $NUM_ENV_STEPS \
    --seed $SEED \
    --hidden_size $HIDDEN_SIZE \
    --action_repeat $ACTION_REPEAT \
    --lr 3e-4 \
    --ppo_epoch 5 \
    --num_mini_batch 4 \
    --clip_param 0.2 \
    --value_loss_coef 0.5 \
    --entropy_coef 0.02 \
    --gamma 0.99 \
    --gae_lambda 0.95 \
    --max_grad_norm 0.5 \
    --layer_N 2 \
    --use_ReLU \
    --use_gae \
    --use_clipped_value_loss \
    --log_interval 10 \
    --save_interval 100"

eval $CMD

echo "=========================================="
echo "Training completed!"
echo "=========================================="
