# PPOBuffer API Documentation

## Overview

`PPOBuffer` is the experience replay buffer class for the PPO algorithm. It stores transitions collected during rollouts and prepares them for training updates.

## Class Definition

```python
class PPOBuffer:
    """
    PPO experience replay buffer
    
    Stores transitions and computes advantages for PPO updates
    """
```

## Key Methods

### insert(obs, action, reward, mask, value, action_log_prob)

```python
def insert(self, obs, action, reward, mask, value, action_log_prob):
    """
    Insert transition into buffer
    
    Args:
        obs: Observation
        action: Action
        reward: Reward
        mask: Done mask
        value: Value estimate
        action_log_prob: Action log probability
    """
```

### compute_returns_and_advantage(last_value, gamma, gae_lambda)

```python
def compute_returns_and_advantage(self, last_value, gamma=0.99, gae_lambda=0.95):
    """
    Compute returns and advantages using GAE
    
    Args:
        last_value: Value estimate for last state
        gamma: Discount factor
        gae_lambda: GAE lambda parameter
    """
```

### get()

```python
def get(self):
    """
    Get all transitions as batched tensors
    
    Returns:
        dict: Batched transitions
    """
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `obs_shape` | tuple | (12,) | Observation shape |
| `action_shape` | tuple | (4,) | Action shape |
| `size` | int | 1000 | Buffer size |
| `device` | torch.device | 'cpu' | Device to use |

## Buffer Structure

The buffer stores the following tensors:

| Tensor | Shape | Description |
|--------|-------|-------------|
| `obs_buf` | (size, *obs_shape) | Observations |
| `action_buf` | (size, *action_shape) | Actions |
| `reward_buf` | (size,) | Rewards |
| `mask_buf` | (size,) | Done masks |
| `value_buf` | (size,) | Value estimates |
| `action_log_prob_buf` | (size,) | Action log probabilities |
| `return_buf` | (size,) | Discounted returns |
| `adv_buf` | (size,) | Advantage estimates |

## GAE Calculation

The buffer uses Generalized Advantage Estimation (GAE) to compute advantages:

```python
def compute_returns_and_advantage(self, last_value, gamma=0.99, gae_lambda=0.95):
    # Last value if not done
    value_preds = torch.cat([self.value_buf, last_value.unsqueeze(0)], dim=0)
    
    # Compute deltas
    deltas = self.reward_buf + gamma * value_preds[1:] * self.mask_buf - value_preds[:-1]
    
    # Compute advantages using GAE
    self.adv_buf = torch.zeros_like(self.reward_buf)
    gae = 0
    for t in reversed(range(len(self.adv_buf))):
        gae = deltas[t] + gamma * gae_lambda * self.mask_buf[t] * gae
        self.adv_buf[t] = gae
    
    # Compute returns
    self.return_buf = self.adv_buf + self.value_buf
```

## Usage Example

```python
from onpolicy.utils.ppo_buffer import PPOBuffer
import torch

# Create buffer
buffer = PPOBuffer(
    obs_shape=(12,),
    action_shape=(4,),
    size=1000,
    device=torch.device('cpu')
)

# Insert transition
buffer.insert(
    obs=torch.randn(12),
    action=torch.randn(4),
    reward=torch.tensor(1.0),
    mask=torch.tensor(1.0),
    value=torch.tensor(0.0),
    action_log_prob=torch.tensor(0.0)
)

# Compute returns and advantages
last_value = torch.tensor(0.0)
buffer.compute_returns_and_advantage(last_value)

# Get batched transitions
data = buffer.get()
print(f"Batch size: {data['obs'].shape[0]}")
```

## Dependencies

- `torch`: PyTorch library for deep learning
- `numpy`: For numerical operations

## See Also

- [PPOActorCritic](./onpolicy_ppo_actor_critic.md): Actor-Critic network
- [Reward Normalizer](./onpolicy_reward_normalizer.md): Reward normalization
