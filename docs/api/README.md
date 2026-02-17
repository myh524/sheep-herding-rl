# API Documentation Index

Welcome to the API documentation for the Sheep Herding High-Level Controller project. This documentation provides detailed information about the core modules and classes used in the project.

## Core Modules

### Environment Modules

- [SheepFlockEnv](./envs_sheep_flock.md): Main environment class implementing Gym interface
- [HighLevelAction](./envs_high_level_action.md): Action decoder for high-level control
- [HerderEntity](./envs_herder_entity.md): Physical model for herder movement
- [SheepEntity](./envs_sheep_entity.md): Boids model for sheep behavior
- [CurriculumEnv](./envs_curriculum_env.md): Curriculum learning environment

### Algorithm Modules

- [PPOActorCritic](./onpolicy_ppo_actor_critic.md): PPO Actor-Critic network architecture
- [PPOBuffer](./onpolicy_ppo_buffer.md): Experience replay buffer for PPO
- [RewardNormalizer](./onpolicy_reward_normalizer.md): Reward normalization for stable training

## Getting Started

### Installation

```bash
# Clone the repository
git clone https://github.com/your-username/sheep-herding.git
cd sheep-herding

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

```python
# Import necessary modules
from envs.sheep_flock import SheepFlockEnv
from onpolicy.algorithms.ppo_actor_critic import PPOActorCritic
import torch

# Create environment
env = SheepFlockEnv(
    world_size=(50, 50),
    num_sheep=10,
    num_herders=3,
    episode_length=150
)

# Create network
args = type('Args', (), {
    'hidden_size': 256,
    'layer_N': 3,
    'use_orthogonal': True,
    'use_ln': True
})()
network = PPOActorCritic(args)

# Reset environment
obs = env.reset()

# Run one episode
done = False
while not done:
    # Convert observation to tensor
    obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
    
    # Get action from network
    action, _, _ = network.get_actions(obs_tensor, None, None)
    action = action.detach().numpy()[0]
    
    # Step environment
    obs, reward, done, info = env.step(action)
    
    # Render environment
    env.render()
```

## Configuration

### Environment Configuration

| Parameter | Description | Default |
|-----------|-------------|---------|
| `world_size` | World dimensions (width, height) | (50, 50) |
| `num_sheep` | Number of sheep | 10 |
| `num_herders` | Number of herders | 3 |
| `episode_length` | Maximum steps per episode | 150 |
| `dt` | Time step | 0.1 |

### Algorithm Configuration

| Parameter | Description | Default |
|-----------|-------------|---------|
| `lr` | Learning rate | 3e-4 |
| `ppo_epoch` | PPO update epochs | 10 |
| `num_mini_batch` | Number of mini-batches | 4 |
| `hidden_size` | Network hidden size | 256 |
| `layer_N` | Number of network layers | 3 |
| `gamma` | Discount factor | 0.99 |
| `gae_lambda` | GAE lambda parameter | 0.95 |
| `clip_param` | PPO clipping parameter | 0.2 |
| `max_grad_norm` | Maximum gradient norm | 0.5 |

## Training

### Standard Training

```bash
# GPU training with curriculum learning (recommended)
bash scripts/train_gpu.sh

# Fast CPU training for testing
bash scripts/train_fast.sh

# Custom training
python train_ppo.py \
    --use_curriculum \
    --num_env_steps 2000000 \
    --lr 3e-4 \
    --hidden_size 256 \
    --layer_N 3 \
    --use_reward_normalization
```

### Evaluation

```bash
# Evaluate trained model
python evaluate_policy.py --model_path results/xxx/model.pt --num_episodes 100
```

### Visualization

```bash
# Visualize trained model
python visualize.py --model_path results/xxx/model.pt --num_episodes 5

# Save as GIF
python visualize.py --model_path results/xxx/model.pt --save_gif output.gif
```

## Troubleshooting

### Common Issues

| Issue | Possible Cause | Solution |
|-------|---------------|----------|
| Reward not increasing | Learning rate too high/low | Adjust lr to 1e-4~1e-3 |
| Actions all zero | Network initialization issue | Check use_orthogonal parameter |
| Reward explosion | No reward clipping | Add reward clipping |
| Unstable training | PPO clipping parameter issue | Adjust clip_param |
| KL divergence negative | Numerical instability | Ensure proper KL divergence calculation |

## See Also

- [Project README](../../README.md): Main project documentation
- [Chinese README](../../README_CN.md): Chinese project documentation
- [Project Architecture](../../docs/01_项目架构白皮书.md): Project architecture documentation
- [Environment Design](../../docs/02_牧羊场景RL环境设计文档.md): Environment design documentation
- [Algorithm Implementation](../../docs/03_算法实现详解文档.md): Algorithm implementation documentation
- [Development Guide](../../docs/04_二次开发指南.md): Development guide
