# RewardNormalizer API Documentation

## Overview

`RewardNormalizer` is the reward normalization class that stabilizes training by normalizing rewards. It uses running mean and standard deviation to normalize rewards during training.

## Class Definition

```python
class RewardNormalizer:
    """
    Reward normalizer using running mean and standard deviation
    
    Stabilizes training by normalizing rewards
    """
```

## Key Methods

### __init__(shape, clip_range=10.0, epsilon=1e-8)

```python
def __init__(self, shape, clip_range=10.0, epsilon=1e-8):
    """
    Initialize reward normalizer
    
    Args:
        shape: Shape of reward
        clip_range: Clipping range for extreme values
        epsilon: Small value to avoid division by zero
    """
```

### normalize(reward)

```python
def normalize(self, reward):
    """
    Normalize reward
    
    Args:
        reward: Raw reward
    
    Returns:
        Normalized reward
    """
```

### update(reward)

```python
def update(self, reward):
    """
    Update running mean and std
    
    Args:
        reward: Raw reward
    """
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `shape` | tuple | () | Shape of reward |
| `clip_range` | float | 10.0 | Clipping range for extreme values |
| `epsilon` | float | 1e-8 | Small value to avoid division by zero |

## Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `mean` | np.ndarray | Running mean of rewards |
| `std` | np.ndarray | Running standard deviation of rewards |
| `count` | int | Number of updates |

## Algorithm

The reward normalizer uses the following algorithm:

1. **Clipping**: Clip extreme reward values to [-clip_range, clip_range]
2. **Running Statistics**: Maintain running mean and standard deviation
3. **Normalization**: Normalize rewards using (reward - mean) / (std + epsilon)

```python
def normalize(self, reward):
    # Clip extreme values
    clipped_reward = np.clip(reward, -self.clip_range, self.clip_range)
    # Normalize
    normalized_reward = (clipped_reward - self.mean) / (self.std + self.epsilon)
    return normalized_reward

def update(self, reward):
    # Clip extreme values
    clipped_reward = np.clip(reward, -self.clip_range, self.clip_range)
    # Update running mean and std
    self.count += 1
    if self.count == 1:
        self.mean = np.mean(clipped_reward)
        self.std = np.std(clipped_reward)
    else:
        old_mean = self.mean
        old_std = self.std
        self.mean = old_mean + (np.mean(clipped_reward) - old_mean) / self.count
        self.std = np.sqrt(old_std**2 * (self.count - 2) / (self.count - 1) + 
                          np.var(clipped_reward) / self.count)
```

## Usage Example

```python
from onpolicy.utils.reward_normalizer import RewardNormalizer
import numpy as np

# Create reward normalizer
normalizer = RewardNormalizer(shape=())

# Simulate rewards
rewards = np.random.normal(5.0, 2.0, 100)

# Normalize and update
normalized_rewards = []
for reward in rewards:
    normalized = normalizer.normalize(reward)
    normalized_rewards.append(normalized)
    normalizer.update(reward)

print(f"Original rewards: mean={np.mean(rewards):.2f}, std={np.std(rewards):.2f}")
print(f"Normalized rewards: mean={np.mean(normalized_rewards):.2f}, std={np.std(normalized_rewards):.2f}")
print(f"Running mean: {normalizer.mean:.2f}, running std: {normalizer.std:.2f}")
```

## Dependencies

- `numpy`: For numerical operations

## See Also

- [PPOBuffer](./onpolicy_ppo_buffer.md): Experience replay buffer for PPO
- [PPOActorCritic](./onpolicy_ppo_actor_critic.md): Actor-Critic network
