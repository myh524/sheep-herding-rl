# PPOActorCritic API Documentation

## Overview

`PPOActorCritic` is the neural network class implementing the PPO algorithm's actor-critic architecture. It handles policy learning and value estimation for the sheep herding task.

## Class Definition

```python
class PPOActorCritic(nn.Module):
    """
    PPO Actor-Critic network
    
    Architecture:
        obs (12D) -> MLP(256, 256, 256) -> Actor -> action (4D)
                                          -> Critic -> value (1D)
    """
```

## Key Methods

### get_actions(obs, rnn_states, masks)

```python
def get_actions(self, obs, rnn_states, masks) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Get actions from observations
    
    Args:
        obs: Observations
        rnn_states: RNN states (not used in this implementation)
        masks: Masks for RNN states (not used in this implementation)
    
    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: (actions, action_log_probs, values)
    """
```

### evaluate_actions(obs, actions)

```python
def evaluate_actions(self, obs, actions) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Evaluate actions for PPO update
    
    Args:
        obs: Observations
        actions: Actions to evaluate
    
    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: (action_log_probs, dist_entropy, values)
    """
```

## Network Architecture

The network consists of:

1. **Shared Backbone**: 3-layer MLP with 256 hidden units per layer
   - Input: 12D observation vector
   - Output: 256D feature vector

2. **Actor Head**: Linear layer mapping features to action mean
   - Input: 256D feature vector
   - Output: 4D action mean
   - Uses Gaussian distribution with learned standard deviation

3. **Critic Head**: Linear layer mapping features to value estimate
   - Input: 256D feature vector
   - Output: 1D value estimate

## Hyperparameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `hidden_size` | int | 256 | Hidden layer size |
| `layer_N` | int | 3 | Number of hidden layers |
| `use_orthogonal` | bool | True | Use orthogonal initialization |
| `use_ln` | bool | True | Use layer normalization |
| `dropout` | float | 0.0 | Dropout probability |

## Initialization

The network uses orthogonal initialization for weights and zero initialization for biases:

```python
def init(module):
    if isinstance(module, nn.Linear):
        nn.init.orthogonal_(module.weight)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
```

## Action Distribution

The actor uses a Gaussian distribution for continuous action space:

```python
action_mean = self.actor_features
action_log_std = self.action_log_std.expand_as(action_mean)
action_std = torch.exp(action_log_std)
dist = Normal(action_mean, action_std)
action = dist.sample()
action_log_probs = dist.log_prob(action).sum(-1, keepdim=True)
```

## Usage Example

```python
from onpolicy.algorithms.ppo_actor_critic import PPOActorCritic
import torch

# Create network
args = type('Args', (), {
    'hidden_size': 256,
    'layer_N': 3,
    'use_orthogonal': True,
    'use_ln': True,
    'dropout': 0.0
})()

network = PPOActorCritic(args)

# Generate sample observation
obs = torch.randn(1, 12)

# Get actions
actions, action_log_probs, values = network.get_actions(obs, None, None)

print(f"Actions: {actions}")
print(f"Action log probs: {action_log_probs}")
print(f"Values: {values}")

# Evaluate actions
action_log_probs_eval, dist_entropy, values_eval = network.evaluate_actions(obs, actions)

print(f"Evaluation - Action log probs: {action_log_probs_eval}")
print(f"Evaluation - Dist entropy: {dist_entropy}")
print(f"Evaluation - Values: {values_eval}")
```

## Dependencies

- `torch`: PyTorch library for deep learning
- `torch.nn`: Neural network modules
- `torch.distributions`: Probability distributions

## See Also

- [PPO Buffer](./onpolicy_ppo_buffer.md): Experience replay buffer for PPO
- [SheepFlockEnv](./envs_sheep_flock.md): Main environment class
