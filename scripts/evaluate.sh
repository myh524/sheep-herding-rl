#!/bin/bash
# Policy Evaluation Script

MODEL_PATH="results/sheep_herding/default/ppo/seed1/xxx/models/model_100000.pt"
NUM_EPISODES=100

echo "=========================================="
echo "Policy Evaluation"
echo "=========================================="
echo "Model path: $MODEL_PATH"
echo "Number of episodes: $NUM_EPISODES"
echo "=========================================="

python -m tools.evaluate_policy \
    --model_path "$MODEL_PATH" \
    --num_episodes $NUM_EPISODES \
    --num_sheep 10 \
    --num_herders 3 \
    --episode_length 60

echo "=========================================="
echo "Evaluation completed!"
echo "=========================================="
