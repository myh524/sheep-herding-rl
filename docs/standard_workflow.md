# Training and Evaluation Standard Workflow

This document outlines the standard workflow for training and evaluating the sheep herding high-level controller.

## 1. Environment Setup

### 1.1 Dependencies Installation

```bash
# Install required dependencies
pip install -r requirements.txt

# Verify installation
pip list | grep -E "numpy|torch|gym|matplotlib|scipy|pillow|opencv-python"
```

### 1.2 Configuration Files

The project uses command-line arguments for configuration. Key configuration files:

- `scripts/train.sh`: Standard training script
- `scripts/train_fast.sh`: Fast CPU training for testing
- `scripts/train_gpu.sh`: GPU training with curriculum learning

## 2. Training Workflow

### 2.1 Pre-training Checks

Before starting training, perform the following checks:

1. **Hardware Check**: Ensure sufficient GPU memory (at least 4GB) for training
2. **Dataset Check**: Verify environment parameters are set correctly
3. **Configuration Check**: Review hyperparameters in training script

### 2.2 Training Modes

#### Fast Testing Mode

```bash
# Fast CPU training for testing (50k steps)
bash scripts/train_fast.sh

# Expected output directory: results/sheep_herding/fast_train/ppo/seed1/
```

#### Standard Training Mode

```bash
# Standard training with curriculum learning
bash scripts/train.sh

# Expected output directory: results/sheep_herding/standard/ppo/seed1/
```

#### GPU Curriculum Learning Mode (Recommended)

```bash
# GPU training with curriculum learning
bash scripts/train_gpu.sh

# Expected output directory: results/sheep_herding/gpu_curriculum/ppo/seed1/
```

### 2.3 Custom Training

For custom training configurations, use the `train_ppo.py` script directly:

```bash
python train_ppo.py \
    --use_curriculum \
    --num_env_steps 2000000 \
    --lr 3e-4 \
    --lr_warmup_steps 1000 \
    --lr_min 1e-5 \
    --hidden_size 256 \
    --layer_N 3 \
    --ppo_epoch 10 \
    --num_mini_batch 4 \
    --clip_param 0.2 \
    --clip_param_final 0.1 \
    --use_clip_annealing \
    --gae_lambda 0.95 \
    --max_grad_norm 0.5 \
    --use_reward_normalization \
    --num_sheep 10 \
    --num_herders 3 \
    --world_size 50.0 50.0 \
    --episode_length 150
```

### 2.4 Training Monitoring

During training, monitor the following metrics:

| Metric | Description | Target |
|--------|-------------|--------|
| `avg_reward` | Average episode reward | >50 |
| `success_rate` | Episode success rate | >70% |
| `kl_divergence` | Policy update size | 0.01-0.02 |
| `train_loss` | Training loss | Decreasing |
| `value_loss` | Value function loss | Decreasing |

### 2.5 Training Analysis

Use the `analyze_training.py` tool to analyze training results:

```bash
# Analyze training curves
python analyze_training.py --log_dir results/sheep_herding/gpu_curriculum/ppo/seed1/

# Expected output: training_analysis.png with reward and loss curves
```

## 3. Evaluation Workflow

### 3.1 Model Evaluation

After training, evaluate the model using the `evaluate_policy.py` script:

```bash
# Evaluate trained model
python evaluate_policy.py \
    --model_path results/sheep_herding/gpu_curriculum/ppo/seed1/models/model_2000000.pt \
    --num_episodes 100 \
    --num_sheep 10 \
    --num_herders 3 \
    --world_size 50.0 50.0

# Expected output: Evaluation metrics including success rate, average reward, etc.
```

### 3.2 Evaluation Metrics

Key evaluation metrics:

| Metric | Calculation | Target |
|--------|-------------|--------|
| Success Rate | Success episodes / Total episodes | >70% |
| Average Reward | Mean episode reward | >50 |
| Average Steps | Mean steps per episode | <120 |
| Reward Std | Standard deviation of episode rewards | <20 |
| Distance to Target | Mean final distance to target | <5.0 |

### 3.3 Robustness Evaluation

Evaluate model robustness across different configurations:

```bash
# Evaluate with different sheep counts
python evaluate_policy.py \
    --model_path results/sheep_herding/gpu_curriculum/ppo/seed1/models/model_2000000.pt \
    --num_episodes 50 \
    --num_sheep 5 \
    --num_herders 3

# Evaluate with different herder counts
python evaluate_policy.py \
    --model_path results/sheep_herding/gpu_curriculum/ppo/seed1/models/model_2000000.pt \
    --num_episodes 50 \
    --num_sheep 10 \
    --num_herders 2
```

## 4. Visualization Workflow

### 4.1 Basic Visualization

```bash
# Visualize trained model
python visualize.py \
    --model_path results/sheep_herding/gpu_curriculum/ppo/seed1/models/model_2000000.pt \
    --num_episodes 5 \
    --num_sheep 10 \
    --num_herders 3

# Expected output: Interactive visualization window
```

### 4.2 Output Formats

#### Save as GIF

```bash
# Save visualization as GIF
python visualize.py \
    --model_path results/sheep_herding/gpu_curriculum/ppo/seed1/models/model_2000000.pt \
    --num_episodes 3 \
    --save_gif output.gif

# Expected output: output.gif file with animation
```

#### Save as Video

```bash
# Save visualization as MP4
python visualize.py \
    --model_path results/sheep_herding/gpu_curriculum/ppo/seed1/models/model_2000000.pt \
    --num_episodes 3 \
    --save_video output.mp4

# Expected output: output.mp4 file with animation
```

### 4.3 Real-time Visualization

```bash
# Real-time visualization with performance metrics
python realtime_visualizer.py \
    --model_path results/sheep_herding/gpu_curriculum/ppo/seed1/models/model_2000000.pt

# Expected output: Real-time visualization with performance metrics
```

## 5. Model Inspection

### 5.1 Model Analysis

Use the `model_inspector.py` tool to analyze model weights and biases:

```bash
# Inspect model
python model_inspector.py \
    --model_path results/sheep_herding/gpu_curriculum/ppo/seed1/models/model_2000000.pt

# Expected output: Model architecture, weight statistics, and layer analysis
```

### 5.2 Performance Monitoring

Use the `performance_monitor.py` tool to monitor model performance:

```bash
# Monitor model performance
python performance_monitor.py \
    --model_path results/sheep_herding/gpu_curriculum/ppo/seed1/models/model_2000000.pt \
    --num_episodes 20

# Expected output: Performance metrics and bottleneck analysis
```

## 6. Demo Generation

### 6.1 Generate Demo

Use the `demo_generator.py` tool to generate demonstration videos:

```bash
# Generate demonstration
python demo_generator.py \
    --model_path results/sheep_herding/gpu_curriculum/ppo/seed1/models/model_2000000.pt \
    --output_dir demos/ \
    --num_demos 5

# Expected output: Demo videos in demos/ directory
```

### 6.2 Demo Configuration

The demo generator supports the following configurations:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--num_sheep` | Number of sheep | 10 |
| `--num_herders` | Number of herders | 3 |
| `--world_size` | World size | 50.0 50.0 |
| `--episode_length` | Episode length | 150 |
| `--save_format` | Output format (gif/mp4) | mp4 |

## 7. Post-training Analysis

### 7.1 Result Visualization

Use the `result_visualizer.py` tool to visualize training results:

```bash
# Visualize training results
python result_visualizer.py \
    --log_dir results/sheep_herding/gpu_curriculum/ppo/seed1/

# Expected output: Comprehensive training analysis plots
```

### 7.2 Model Comparison

To compare different models:

```bash
# Compare multiple models
python result_visualizer.py \
    --log_dirs \
    results/sheep_herding/fast_train/ppo/seed1/ \
    results/sheep_herding/standard/ppo/seed1/ \
    results/sheep_herding/gpu_curriculum/ppo/seed1/

# Expected output: Comparative analysis of different models
```

## 8. Troubleshooting

### 8.1 Common Issues

| Issue | Possible Cause | Solution |
|-------|---------------|----------|
| Training crashes with OOM | Insufficient GPU memory | Reduce batch size or use CPU training |
| Reward not increasing | Learning rate too high/low | Adjust lr to 1e-4~1e-3 |
| Actions all zero | Network initialization issue | Check use_orthogonal parameter |
| Reward explosion | No reward clipping | Add reward clipping |
| Unstable training | PPO clipping parameter issue | Adjust clip_param to 0.1~0.3 |
| KL divergence negative | Numerical instability | Ensure proper KL divergence calculation |

### 8.2 Debugging Tips

1. **Check Observation Range**:
   ```python
   # Print observation range
   obs = env.reset()
   print(f"Obs range: [{obs.min():.3f}, {obs.max():.3f}]")
   ```

2. **Verify Action Decoding**:
   ```python
   # Check action decoding
   decoder = HighLevelAction()
   raw = np.random.uniform(-1, 1, 4)
   decoded = decoder.decode_action(raw, target_direction=0)
   print(f"Raw action: {raw}")
   print(f"Decoded params: {decoded}")
   ```

3. **Monitor Reward Distribution**:
   ```python
   # Record reward distribution
   rewards = []
   for _ in range(100):
       obs, reward, done, info = env.step(env.action_space.sample())
       rewards.append(reward)
   print(f"Reward: mean={np.mean(rewards):.2f}, std={np.std(rewards):.2f}")
   ```

## 9. Best Practices

1. **Hyperparameter Tuning**: Start with default parameters, then adjust based on training curves
2. **Curriculum Learning**: Always use curriculum learning for better generalization
3. **Model Selection**: Select models based on success rate and reward stability, not just peak reward
4. **Evaluation**: Evaluate models on different flock sizes to ensure generalization
5. **Documentation**: Document all training runs with parameters and results

## 10. Summary

This workflow provides a structured approach to training and evaluating the sheep herding high-level controller. By following these steps, you can ensure consistent and reliable results across different training runs.

For more detailed information, refer to the following documentation:

- [Project README](../README.md)
- [API Documentation](./api/README.md)
- [Environment Design Document](./02_牧羊场景RL环境设计文档.md)
- [Algorithm Implementation Document](./03_算法实现详解文档.md)
