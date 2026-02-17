# HighLevelAction API Documentation

## Overview

`HighLevelAction` is the action decoder class that converts neural network output to station parameters for herders. It handles the mapping from continuous action space to physical herder positions.

## Class Definition

```python
class HighLevelAction:
    """
    High-level action decoder
    
    Args:
        R_ref: Reference radius (default world diagonal/6)
        R_min: Minimum station radius
        R_max: Maximum station radius
    """
```

## Key Methods

### decode_action(raw_action, target_direction)

```python
def decode_action(self, raw_action, target_direction) -> dict:
    """
    Decode raw action from neural network to station parameters
    
    Args:
        raw_action: Raw action from network [-1, 1]
        target_direction: Direction to target
    
    Returns:
        dict: Decoded station parameters
    """
```

### sample_herder_positions(num_herders, params)

```python
def sample_herder_positions(self, num_herders, params) -> list:
    """
    Sample target positions for herders
    
    Args:
        num_herders: Number of herders
        params: Station parameters
    
    Returns:
        list: List of target positions
    """
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `R_ref` | float | None | Reference radius (calculated from world size if None) |
| `R_min` | float | 5.0 | Minimum station radius |
| `R_max` | float | 20.0 | Maximum station radius |

## Action Decoding

The raw action from the neural network is decoded as follows:

1. **wedge_center**: [-1, 1] → [-π, π] (formation center angle)
2. **wedge_width**: [-1, 1] → [0, 1] (formation spread)
3. **radius**: [-1, 1] → [R_min, R_max] (station radius)
4. **asymmetry**: [-1, 1] → [-0.5, 0.5] (flank offset)

## Formation Modes

Based on the decoded parameters, different formation modes are created:

| Mode | wedge_width | Description |
|------|-------------|-------------|
| PUSH | ≈ 0 | All herders at same angle |
| NARROW | ≈ 0.3 | Narrow wedge formation |
| WIDE | ≈ 0.6 | Wide wedge formation |
| SURROUND | ≈ 1 | 360° surround formation |

## Position Calculation

The herder positions are calculated deterministically based on the decoded parameters:

1. Calculate formation center with asymmetry offset
2. Distribute herders evenly across the wedge width
3. Position herders at the specified radius

```python
# Formation center calculation
formation_center = wedge_center + asymmetry * wedge_width * π

# Angle distribution for each herder
total_spread = wedge_width * π
for i in range(num_herders):
    fraction = i / (num_herders - 1)
    angle = formation_center + (fraction - 0.5) * total_spread
    position = flock_center + radius * [cos(angle), sin(angle)]
```

## Usage Example

```python
from envs.high_level_action import HighLevelAction

# Create action decoder
decoder = HighLevelAction(R_ref=15.0)

# Decode raw action
raw_action = [-0.2, 0.3, 0.1, 0.0]
target_direction = 0.0
params = decoder.decode_action(raw_action, target_direction)

# Sample herder positions
num_herders = 3
positions = decoder.sample_herder_positions(num_herders, params)

print(f"Decoded params: {params}")
print(f"Herder positions: {positions}")
```

## Dependencies

- `numpy`: For numerical operations
- `math`: For trigonometric functions

## See Also

- [SheepFlockEnv](./envs_sheep_flock.md): Main environment class
- [HerderEntity](./envs_herder_entity.md): Physical model for herder movement
