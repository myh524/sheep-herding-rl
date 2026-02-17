# SheepFlockEnv API Documentation

## Overview

`SheepFlockEnv` is the main environment class for sheep herding reinforcement learning, implementing the OpenAI Gym interface. It manages the simulation of sheep flock behavior and herder interactions.

## Class Definition

```python
class SheepFlockEnv:
    """
    Sheep herding reinforcement learning environment
    
    Args:
        world_size: World size (width, height)
        num_sheep: Number of sheep
        num_herders: Number of herders
        episode_length: Maximum steps per episode
        dt: Time step
        reward_config: Reward weight configuration
    """
```

## Key Methods

### reset()

```python
def reset(self) -> np.ndarray:
    """
    Reset environment to initial state
    
    Returns:
        np.ndarray: Initial observation
    """
```

### step(action)

```python
def step(self, action) -> Tuple[np.ndarray, float, bool, dict]:
    """
    Execute one step in the environment
    
    Args:
        action: Action to execute
    
    Returns:
        Tuple[np.ndarray, float, bool, dict]: (observation, reward, done, info)
    """
```

### render(mode)

```python
def render(self, mode='human'):
    """
    Render the environment
    
    Args:
        mode: Render mode
    """
```

## Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `observation_space` | gym.Space | Observation space (Box(10,)) |
| `action_space` | gym.Space | Action space (Box(4,)) |
| `world_size` | Tuple[float, float] | World dimensions |
| `num_sheep` | int | Number of sheep |
| `num_herders` | int | Number of herders |
| `episode_length` | int | Maximum steps per episode |

## Observation Space

The observation space is a 10-dimensional vector with normalized values:

| Index | Observation | Normalization | Description |
|-------|-------------|---------------|-------------|
| [0] | `d_goal / r_max` | Distance / world diagonal | Flock to target distance |
| [1:3] | `cos φ, sin φ` | Unit vector | Flock→target direction |
| [3] | `flock_speed / max_speed` | Speed / max speed | Flock speed magnitude |
| [4:6] | `cos θ_vel, sin θ_vel` | Unit vector | Flock velocity direction (relative to target) |
| [6] | `spread / r_max` | Spread / r_max | Flock spread |
| [7:9] | `cos θ_main, sin θ_main` | Unit vector | Flock main direction (relative to target) |
| [9] | `num_sheep / 30.0` | Count / 30 | Flock size |

## Action Space

The action space is a 4-dimensional continuous vector:

| Index | Parameter | Range | Description |
|-------|-----------|-------|-------------|
| [0] | `wedge_center` | [-π, π] | Formation center angle (relative to target direction) |
| [1] | `wedge_width` | [0, 1] | Formation spread (0=push, 1=surround) |
| [2] | `radius` | [R_min, R_max] | Station radius |
| [3] | `asymmetry` | [-0.5, 0.5] | Flank offset (for steering control) |

## Reward Function

The reward function includes multiple components:

1. **Potential-based reward**: Dense guidance signal based on distance to target
2. **Distance reward**: Encourages flock to move closer to target
3. **Compactness reward**: Incentivizes tight flock formation
4. **Direction alignment reward**: Rewards flock movement toward target
5. **Success bonus**: Large reward for reaching target
6. **Time penalty**: Small penalty per step to encourage efficiency

## Usage Example

```python
from envs.sheep_flock import SheepFlockEnv

# Create environment
env = SheepFlockEnv(
    world_size=(50, 50),
    num_sheep=10,
    num_herders=3,
    episode_length=150,
    dt=0.1
)

# Reset environment
obs = env.reset()

# Run one episode
done = False
while not done:
    # Sample random action
    action = env.action_space.sample()
    # Step environment
    obs, reward, done, info = env.step(action)
    # Render environment
    env.render()
```

## Dependencies

- `numpy`: For numerical operations
- `gym`: For reinforcement learning environment interface
- `scipy`: For spatial calculations
- `matplotlib`: For rendering

## See Also

- [HighLevelAction](./envs_high_level_action.md): Action decoder for high-level control
- [HerderEntity](./envs_herder_entity.md): Physical model for herder movement
- [SheepEntity](./envs_sheep_entity.md): Boids model for sheep behavior
