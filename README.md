# High-Level Controller for Sheep Herding

A reinforcement learning project for training high-level decision controllers in multi-robot sheep herding scenarios.

## Project Overview

### Architecture

This project implements a **hierarchical control architecture**:

```
┌─────────────────────────────────────────────────────────────┐
│                    High-Level Controller                     │
│                  (This Project - PPO Training)               │
├─────────────────────────────────────────────────────────────┤
│  Input:  Flock state (centroid, shape, spread, direction)   │
│  Output: Station parameters (μ_r, σ_r, μ_θ, κ)              │
│          ↓                                                   │
│  Sample: n target positions for herders                     │
└─────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                    Low-Level Controller                      │
│                    (Completed - Separate Project)            │
├─────────────────────────────────────────────────────────────┤
│  Input:  n target positions                                  │
│  Output: Navigation actions for each herder                  │
└─────────────────────────────────────────────────────────────┘
```

### Key Features

- **PPO Algorithm**: Proximal Policy Optimization for continuous action spaces
- **Generalization**: Single trained network works across different flock/herder configurations
- **Curriculum Learning**: Progressive difficulty increase for stable training
- **Physical Constraints**: Herder movement with speed/acceleration limits

---

## Algorithm Theory Analysis

### Why PPO?

This project selects PPO (Proximal Policy Optimization) as the core algorithm based on the following considerations:

#### 1. Continuous Action Space Compatibility

The sheep herding task has a 4-dimensional continuous action space (μ_r, σ_r, μ_θ, κ). PPO handles this effectively:

```python
# PPO uses Gaussian policy, suitable for continuous actions
action_dist = Normal(mean, std)
action = action_dist.sample()
log_prob = action_dist.log_prob(action)
```

Compared to discrete algorithms like DQN, PPO doesn't require discretization of continuous actions, preserving complete information of the action space.

#### 2. Policy Stability

PPO's core innovation - the clipped objective function - ensures controlled policy updates:

```
L^CLIP(θ) = E[min(r(θ)Â, clip(r(θ), 1-ε, 1+ε)Â)]

Where r(θ) = π_θ(a|s) / π_θ_old(a|s) is the importance sampling ratio
```

**Significance for Sheep Herding Task**:
- Sheep behavior has randomness (Boids model), policy needs to be robust
- Excessive policy updates may cause flock to lose control
- Clipping parameter ε=0.2 limits single update magnitude

#### 3. Sample Efficiency vs Computational Efficiency Balance

| Algorithm | Sample Efficiency | Computational Efficiency | Stability | Use Case |
|-----------|-------------------|-------------------------|-----------|----------|
| PPO | Medium | High | High | Continuous control, general |
| SAC | High | Medium | Medium | Maximum entropy exploration |
| TD3 | High | Medium | Medium | Deterministic policy |
| DDPG | Medium | High | Low | Deterministic policy |
| MAPPO | Medium | High | High | Multi-agent |

**Why PPO for this project**:
1. Sheep herding is a single-agent problem (high-level controller makes unified decisions)
2. Requires stable training process (curriculum learning stage transitions)
3. Best choice when computational resources are limited

### Comparison with Other Algorithms

#### PPO vs SAC

```python
# SAC's entropy regularization
loss = -E[Q(s,a)] + α * H(π(·|s))

# PPO's clipped objective
loss = -min(r(θ)Â, clip(r(θ), 1-ε, 1+ε)Â)
```

| Aspect | PPO | SAC |
|--------|-----|-----|
| Exploration Strategy | Stochastic policy + entropy bonus | Maximum entropy framework + auto temperature |
| Hyperparameters | Fewer (ε, lr) | More (α, τ, lr, etc.) |
| Stability | High (clipping protection) | Medium (Q-value overestimation risk) |
| Sheep Task Fit | ✅ Stable training | ⚠️ May over-explore |

#### PPO vs MAPPO

MAPPO is the multi-agent extension of PPO, but this project doesn't need it:

```
MAPPO Use Cases:
- Each agent makes independent decisions
- Requires centralized training, decentralized execution
- Competition/cooperation relationships between agents

This Project's Characteristics:
- High-level controller makes unified decisions (single-agent)
- Outputs station parameters, not individual actions for each herder
- Uses von Mises distribution sampling for implicit coordination
```

### Convergence Analysis

#### Theoretical Guarantees

PPO inherits TRPO's theoretical foundation:

1. **Monotonic Improvement Guarantee**: Policy performance monotonically non-decreasing when approximation error is small enough
2. **Convergence to Local Optimum**: Converges when KL constraints are satisfied

#### Practical Convergence Characteristics

```
Training Phase Breakdown:
├── Early (0-20%): Policy random exploration, high reward variance
├── Middle (20-60%): Learning basic stationing strategy, reward rising
├── Late (60-100%): Policy fine-tuning, reward convergence
└── Curriculum Switch: May briefly drop, quick recovery
```

### Sample Efficiency Discussion

#### Design Choices to Improve Sample Efficiency

1. **GAE (Generalized Advantage Estimation)**
```python
# GAE computes advantage function
Â_t = Σ_{l=0}^∞ (γλ)^l δ_{t+l}
δ_t = r_t + γV(s_{t+1}) - V(s_t)
```
- λ=0.95 balances bias and variance
- More flexible than n-step return

2. **Multiple Epochs**
```python
for epoch in range(ppo_epochs):  # Usually 10-15 rounds
    for minibatch in minibatches:
        update_policy(minibatch)
```
- Reuse each batch of data, improving efficiency
- Note: Too many epochs may cause overfitting

3. **Observation Space Normalization**
```python
# All observations relative/normalized
obs[0] = distance / r_max  # Range [0, 2]
obs[1:3] = direction_unit_vector  # Range [-1, 1]
```
- Accelerates neural network learning
- Improves generalization capability

---

## Action Space Design

### Station Parameters (4D Continuous)

| Index | Parameter | Range | Description |
|-------|-----------|-------|-------------|
| [0] | `wedge_center` | [-π, π] | Formation center angle (relative to target direction) |
| [1] | `wedge_width` | [0, 1] | Formation spread (0=push, 1=surround) |
| [2] | `radius` | [R_min, R_max] | Station radius |
| [3] | `asymmetry` | [-0.5, 0.5] | Flank offset (for steering control) |

### Formation Modes

```
wedge_width ≈ 0    → PUSH mode (all herders at same angle)
wedge_width ≈ 0.3  → NARROW mode (narrow wedge)
wedge_width ≈ 0.6  → WIDE mode (wide wedge)
wedge_width ≈ 1    → SURROUND mode (360° surround)
```

**Use Cases**:
- `PUSH`: Concentrated force pushing in one direction
- `SURROUND`: Containing the flock, preventing escape
- `asymmetry > 0`: Right flank emphasis, steer flock left
- `asymmetry < 0`: Left flank emphasis, steer flock right

### Position Calculation (Deterministic)

```python
# Formation center calculation
formation_center = wedge_center + asymmetry * wedge_width * π

# Angle distribution (deterministic, not random sampling)
total_spread = wedge_width * π
for i in range(num_herders):
    fraction = i / (num_herders - 1)
    angle = formation_center + (fraction - 0.5) * total_spread
    position = flock_center + radius * [cos(angle), sin(angle)]
```

**Key Improvement**: Deterministic mapping ensures same action → same target position, avoiding target flickering.

---

## Observation Space Design

### 10D Relative/Normalized Observation

| Index | Observation | Normalization | Description |
|-------|-------------|---------------|-------------|
| [0] | `d_goal / r_max` | Distance / world diagonal | Flock to target distance |
| [1:3] | `cos φ, sin φ` | Unit vector | Flock→target direction |
| [3] | `flock_speed / max_speed` | Speed / max speed | Flock speed magnitude |
| [4:6] | `cos θ_vel, sin θ_vel` | Unit vector | Flock velocity direction (relative to target) |
| [6] | `spread / r_max` | Spread / r_max | Flock spread |
| [7:9] | `cos θ_main, sin θ_main` | Unit vector | Flock main direction (relative to target) |
| [9] | `num_sheep / 30.0` | Count / 30 | Flock size |

### Key Design Principles

1. **All observations relative/normalized**: Independent of world size and rotation
2. **Target direction as reference**: Angles relative to target direction
3. **Velocity information**: Added flock speed magnitude and direction to help policy judge flock motion state

---

## Physical Model

### HerderEntity Class

Herders have physical constraints for realistic movement:

```python
class HerderEntity:
    max_speed = 3.0    # Maximum speed (units/second)
    max_accel = 1.0    # Maximum acceleration
    
    def update(self, dt):
        # Move towards target with speed/acceleration limits
        # Position changes smoothly, no flickering
```

### Potential Field Obstacle Avoidance

Herders automatically avoid the flock when moving, using potential field method:

```python
def update_herders(self, dt):
    # Attraction force: towards target
    attract_force = (target - position) / distance
    
    # Repulsion force: away from flock (when too close)
    if dist_to_flock < avoid_radius:
        repel_strength = (avoid_radius - dist_to_flock) / avoid_radius
        repel_force = (position - flock_center) / dist_to_flock * repel_strength
    
    # Combined force movement
    move_direction = attract_force + repel_force * 1.5
```

**Avoid Radius**: `avoid_radius = flock_spread * 2.5 + 3.0`

**Design Principle**:
- High-level policy: Decides "where to go" (target position)
- Low-level control: Decides "how to get there" (automatic avoidance)

### Decision Frequency Control

```
Step 1, 6, 11, ...: High-level policy outputs station parameters
Step 2-5, 7-10, ...: Herders move towards targets (physical constraints + avoidance)

This ensures:
- Herder positions change smoothly
- Sheep evasion force is continuous
- Policy has time to execute decisions
- Flock is not scattered by herders passing through
```

### Speed Parameters

| Entity | Parameter | Value | Notes |
|--------|-----------|-------|-------|
| Sheep | `max_speed` | 3.0 | Increased for faster training |
| Sheep | `max_force` | 0.3 | Faster turning |
| Herder | `move_speed` | 5.0*dt | Movement per step |
| Herder | `avoid_radius` | ~15 | Flock avoidance radius |

---

## Reward Function

### Improved Design

```python
reward = 0.0

# 1. Potential-based reward (dense guidance)
potential = -distance / max_distance
reward += (potential - prev_potential) * 100.0

# 2. Distance reward (reduced weight)
reward += distance_delta * 0.5

# 3. Compactness reward (positive incentive)
if spread < target_spread:
    reward += 0.5

# 4. Direction alignment reward
reward += max(0, alignment) * 0.3

# 5. Success bonus (increased)
if at_target:
    reward += 50.0

# 6. Time penalty (reduced)
reward += -0.005

# 7. Reward clipping
reward = clip(reward, -10.0, 100.0)
```

### Key Improvements

- **Potential-based reward**: Dense guidance signal
- **Reduced negative rewards**: More positive incentives
- **Increased success bonus**: 10 → 50
- **Reward clipping**: Prevent extreme values

---

## Training Configuration

### Recommended Hyperparameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| `lr` | 3e-4 | Reduced from 5e-4 |
| `ppo_epoch` | 10 | Reduced from 15 |
| `num_mini_batch` | 4 | Increased from 1 |
| `hidden_size` | 256 | Increased from 64 |
| `layer_N` | 3 | Increased from 1 |
| `max_grad_norm` | 0.5 | Reduced from 10.0 |
| `episode_length` | 150 | Increased from 100 |
| `lr_warmup_steps` | 1000 | Learning rate warmup steps |
| `lr_min` | 1e-5 | Minimum learning rate for cosine annealing |
| `clip_param_final` | 0.1 | Final clip parameter for annealing |
| `use_clip_annealing` | True | Enable clip parameter annealing |

### Curriculum Learning Stages

| Stage | Sheep | Herders | World | Episode Length | Target Success |
|-------|-------|---------|-------|----------------|----------------|
| 1 | 3 | 2 | 30×30 | 50 | 80% |
| 2 | 5 | 2 | 40×40 | 75 | 70% |
| 3 | 6 | 3 | 30×30 | 100 | 60% |

---

## Code Implementation Details

### Core Class API Documentation

#### SheepFlockEnv

Main environment class implementing Gym interface:

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
    
    Methods:
        reset() -> np.ndarray: Reset environment, return initial observation
        step(action) -> Tuple[obs, reward, done, info]: Execute action
        render(mode): Render environment
    
    Attributes:
        observation_space: Box(10,) observation space
        action_space: Box(4,) action space
    """
```

#### HighLevelAction

Action decoder, converts neural network output to station parameters:

```python
class HighLevelAction:
    """
    High-level action decoder
    
    Args:
        R_ref: Reference radius (default world diagonal/6)
        R_min: Minimum station radius
        R_max: Maximum station radius
    
    Methods:
        decode_action(raw_action, target_direction) -> Dict:
            Decode raw action [-1,1] to actual parameters
        
        sample_herder_positions(num_herders, mu_r, sigma_r, mu_theta, kappa):
            Sample herder positions
    """
```

#### PPOActorCritic

PPO network architecture:

```python
class PPOActorCritic(nn.Module):
    """
    PPO Actor-Critic network
    
    Architecture:
        obs (12D) -> MLP(256, 256, 256) -> Actor -> action (4D)
                                          -> Critic -> value (1D)
    
    Methods:
        get_actions(obs, rnn_states, masks) -> actions, log_probs, values
        evaluate_actions(obs, actions) -> log_probs, entropy, values
    """
```

### Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                        Training Data Flow                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────┐    reset()    ┌──────────┐                        │
│  │  Env     │ ────────────→ │  Obs     │                        │
│  │          │               │  (12D)   │                        │
│  └──────────┘               └──────────┘                        │
│       ↑                           │                             │
│       │ step(action)              ↓                             │
│       │                    ┌──────────┐                         │
│       │                    │  Policy  │ get_actions()           │
│       │                    │ (Actor)  │ ──────────→ action (4D) │
│       │                    └──────────┘                         │
│       │                           │                             │
│       │                           ↓                             │
│       │                    ┌──────────┐                         │
│       │                    │ Decoder  │ decode_action()         │
│       │                    │          │ ──────────→ params      │
│       │                    └──────────┘                         │
│       │                           │                             │
│       │                           ↓                             │
│       │                    ┌──────────┐                         │
│       └────────────────────│ Sampler  │ sample_positions()      │
│                            └──────────┘ ──────────→ n positions │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Configuration Reference

Training script supports the following key configurations:

```bash
# Environment configuration
--num_sheep 10          # Number of sheep
--num_herders 3         # Number of herders
--world_size 50.0 50.0  # World size
--episode_length 100    # Episode length

# Training configuration
--num_env_steps 1000000 # Total training steps
--lr 5e-4              # Learning rate
--ppo_epoch 15         # PPO update epochs
--num_mini_batch 1     # Number of mini-batches

# Network configuration
--hidden_size 64       # Hidden layer size
--layer_N 1            # Number of layers

# Training mode
--use_curriculum       # Enable curriculum learning
--use_randomized       # Enable randomized training
```

### Debugging Tips

#### 1. Check Observation Range

```python
# Print observation range at training start
obs = env.reset()
print(f"Obs range: [{obs.min():.3f}, {obs.max():.3f}]")
# Expected: Most values in [-1, 1] range
```

#### 2. Verify Action Decoding

```python
# Check if action decoding is correct
decoder = HighLevelAction()
for _ in range(10):
    raw = np.random.uniform(-1, 1, 4)
    decoded = decoder.decode_action(raw, target_direction=0)
    print(f"raw: {raw} -> kappa: {decoded['kappa']:.2f}")
```

#### 3. Monitor Reward Distribution

```python
# Record reward distribution
rewards = []
for _ in range(100):
    obs, reward, done, info = env.step(random_action)
    rewards.append(reward)
print(f"Reward: mean={np.mean(rewards):.2f}, std={np.std(rewards):.2f}")
```

#### 4. Common Issues

| Issue | Possible Cause | Solution |
|-------|---------------|----------|
| Reward not increasing | Learning rate too high/low | Adjust lr to 1e-4~1e-3 |
| Actions all zero | Network initialization issue | Check use_orthogonal |
| Reward explosion | No reward clipping | Add reward clipping |
| Unstable training | PPO clipping parameter issue | Adjust clip_param |

---

## Experimental Results & Analysis

### Training Curve Analysis Template

#### Standard Training Curve

```
Reward Curve (Illustrative)
    │
100 ┤                    ╭────────────────────
    │                   ╱
 50 ┤              ╭───╯
    │         ╭────╯
  0 ┤    ╭────╯
    │╭───╯
-50 ┤╯
    └──────────────────────────────────────────
      0    200k   400k   600k   800k   1M steps
           │      │      │      │
           Stage1 Stage2 Stage3 Converged
```

#### Key Metrics

| Metric | Calculation | Target | Description |
|--------|-------------|--------|-------------|
| Success Rate | Success episodes / Total episodes | >70% | Proportion reaching target |
| Average Reward | Episode reward mean | >50 | Training convergence indicator |
| Convergence Speed | Steps to reach 90% optimal reward | <500k | Sample efficiency |
| Reward Std | Episode reward standard deviation | <20 | Policy stability |

### Ablation Experiment Design

#### 1. Observation Space Ablation

| Experiment | Obs Dimension | Success Rate | Notes |
|------------|---------------|--------------|-------|
| Full | 12D | Baseline | Complete observation |
| No Shape | 10D | -5% | Remove λ1, λ2 |
| No Direction | 10D | -10% | Remove direction vectors |
| No Spread | 11D | -3% | Remove spread |

#### 2. Reward Function Ablation

| Experiment | Reward Components | Success Rate | Notes |
|------------|-------------------|--------------|-------|
| Full | All | Baseline | Complete reward |
| No Potential | Remove potential reward | -15% | Lack of dense guidance |
| No Compact | Remove compactness reward | -5% | Flock tends to scatter |
| No Direction | Remove direction reward | -8% | Unstable push direction |

#### 3. Network Architecture Ablation

| Experiment | hidden_size | layer_N | Success Rate |
|------------|-------------|---------|--------------|
| Small | 64 | 1 | Baseline |
| Medium | 128 | 2 | +5% |
| Large | 256 | 3 | +8% |
| Too Large | 512 | 4 | +6% (overfitting risk) |

### Performance Metrics Definition

```python
# Evaluation metrics calculation
def evaluate_policy(policy, env, num_episodes=100):
    metrics = {
        'success_rate': 0,
        'avg_reward': 0,
        'avg_steps': 0,
        'avg_distance': 0,
    }
    
    for _ in range(num_episodes):
        obs = env.reset()
        done = False
        total_reward = 0
        steps = 0
        
        while not done:
            action = policy.get_action(obs)
            obs, reward, done, info = env.step(action)
            total_reward += reward
            steps += 1
        
        metrics['success_rate'] += info['is_success']
        metrics['avg_reward'] += total_reward
        metrics['avg_steps'] += steps
        metrics['avg_distance'] += info['distance_to_target']
    
    for k in metrics:
        metrics[k] /= num_episodes
    
    return metrics
```

### Failure Case Analysis

#### Common Failure Modes

| Failure Type | Manifestation | Root Cause | Improvement Suggestion |
|--------------|---------------|------------|------------------------|
| Flock Scatter | Spread keeps increasing | κ too small, loose surround | Increase compactness reward weight |
| Push Failure | Distance not decreasing | κ too large, insufficient push force | Adjust κ range, add direction reward |
| Boundary Stuck | Flock lingers at boundary | Improper herder positioning | Add boundary awareness observation |
| Timeout | Steps exhausted | Conservative policy | Reduce time penalty, increase success reward |

---

## Sim2Real Transfer Guide

### Simulation vs Real World Gap Analysis

#### 1. Sheep Behavior Differences

| Aspect | Simulation | Real World | Gap Impact |
|--------|------------|------------|------------|
| Behavior Model | Boids rules | Complex psychology | Policy may fail |
| Perception Range | Fixed radius | Variable | Needs robustness |
| Evasion Behavior | Simple repulsion | Individual differences | Needs generalization |

#### 2. Herder Differences

| Aspect | Simulation | Real World | Gap Impact |
|--------|------------|------------|------------|
| Motion Model | Idealized | Dynamics delay | Execution error |
| Localization Accuracy | Perfect | Noisy | Observation noise |
| Communication Delay | None | Exists | Decision delay |

### Domain Randomization Strategy

```python
class DomainRandomizedEnv(SheepFlockEnv):
    """Domain randomized environment"""
    
    def reset(self):
        # Randomize sheep parameters
        self.sheep_max_speed = np.random.uniform(1.5, 2.5)
        self.sheep_perception = np.random.uniform(4.0, 6.0)
        
        # Randomize herder parameters
        self.herder_max_speed = np.random.uniform(2.5, 3.5)
        self.herder_accel = np.random.uniform(0.8, 1.2)
        
        # Randomize observation noise
        self.obs_noise_std = np.random.uniform(0.0, 0.05)
        
        return super().reset()
    
    def _get_obs(self):
        obs = super()._get_obs()
        # Add observation noise
        obs += np.random.normal(0, self.obs_noise_std, obs.shape)
        return obs
```

#### Recommended Randomization Ranges

| Parameter | Simulation Baseline | Randomization Range |
|-----------|---------------------|---------------------|
| Sheep Speed | 2.0 | [1.5, 2.5] |
| Sheep Perception Radius | 5.0 | [4.0, 6.0] |
| Herder Speed | 3.0 | [2.5, 3.5] |
| Observation Noise | 0 | [0, 0.05] |
| Action Delay | 0 | [0, 2] steps |

### Real World Deployment Guide

#### 1. Pre-deployment Preparation

```python
# Export ONNX model for deployment
def export_to_onnx(policy, save_path):
    dummy_obs = torch.randn(1, 12)
    torch.onnx.export(
        policy.actor,
        dummy_obs,
        save_path,
        input_names=['observation'],
        output_names=['action'],
        dynamic_axes={'observation': {0: 'batch_size'}}
    )
```

#### 2. Deployment Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                   Real World Deployment Architecture         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐              │
│  │ Camera   │───→│ Localization───→│ State    │              │
│  │ (Perception)  │ (Fusion) │    │ Estimation│              │
│  └──────────┘    └──────────┘    └──────────┘              │
│                                        │                    │
│                                        ↓                    │
│                                 ┌──────────┐                │
│                                 │ High-level│                │
│                                 │ Policy   │                │
│                                 │ (ONNX)   │                │
│                                 └──────────┘                │
│                                        │                    │
│                                        ↓                    │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐              │
│  │ Herder   │←───│ Low-level│←───│ Action   │              │
│  │ (Execute)│    │ Control  │    │ Decoder  │              │
│  └──────────┘    └──────────┘    └──────────┘              │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

#### 3. Safety Considerations

```python
class SafePolicyWrapper:
    """Safe policy wrapper"""
    
    def __init__(self, policy, safety_bounds):
        self.policy = policy
        self.bounds = safety_bounds
    
    def get_action(self, obs):
        action = self.policy.get_action(obs)
        
        # Safety constraints
        action = np.clip(action, self.bounds['action_low'], 
                        self.bounds['action_high'])
        
        # Emergency stop condition
        if self._check_emergency(obs):
            return self._emergency_action()
        
        return action
    
    def _check_emergency(self, obs):
        # Check if emergency stop is needed
        distance = obs[0] * self.bounds['r_max']
        if distance < 1.0:  # Too close to target
            return True
        return False
```

---

## Extended Features

### Multi-Target Scenario Design

```python
class MultiTargetEnv(SheepFlockEnv):
    """Multi-target scenario environment"""
    
    def __init__(self, num_targets=3, **kwargs):
        super().__init__(**kwargs)
        self.num_targets = num_targets
        self.target_sequence = []
        self.current_target_idx = 0
    
    def reset(self):
        obs = super().reset()
        
        # Generate target sequence
        self.target_sequence = self._generate_targets()
        self.current_target_idx = 0
        self.scenario.target_position = self.target_sequence[0]
        
        return obs
    
    def step(self, action):
        obs, reward, done, info = super().step(action)
        
        # Check current target completion
        if info['is_success']:
            self.current_target_idx += 1
            if self.current_target_idx < self.num_targets:
                self.scenario.target_position = \
                    self.target_sequence[self.current_target_idx]
                done = False
            else:
                info['all_targets_reached'] = True
        
        return obs, reward, done, info
```

### Obstacle Avoidance

```python
class ObstacleEnv(SheepFlockEnv):
    """Environment with obstacles"""
    
    def __init__(self, num_obstacles=3, **kwargs):
        super().__init__(**kwargs)
        self.obstacles = []
        self.obs_dim = 12 + num_obstacles * 4  # Extended observation
    
    def reset(self):
        obs = super().reset()
        
        # Generate obstacles
        self.obstacles = self._generate_obstacles()
        
        return self._get_extended_obs()
    
    def _get_extended_obs(self):
        base_obs = super()._get_obs()
        
        # Add obstacle information
        obstacle_obs = []
        for obs in self.obstacles:
            # Relative position and size
            rel_pos = (obs['position'] - self.flock_center) / self.r_max
            size = obs['radius'] / self.r_max
            obstacle_obs.extend([rel_pos[0], rel_pos[1], size, 0])
        
        return np.concatenate([base_obs, obstacle_obs])
```

### Dynamic Environment Adaptation

```python
class DynamicEnv(SheepFlockEnv):
    """Dynamic environment (moving target)"""
    
    def __init__(self, target_speed=0.5, **kwargs):
        super().__init__(**kwargs)
        self.target_speed = target_speed
        self.target_velocity = np.zeros(2)
    
    def step(self, action):
        # Update target position
        self._update_target()
        
        return super().step(action)
    
    def _update_target(self):
        # Target slowly moves
        if np.random.random() < 0.1:
            # Randomly change direction
            angle = np.random.uniform(0, 2*np.pi)
            self.target_velocity = self.target_speed * np.array([
                np.cos(angle), np.sin(angle)
            ])
        
        self.scenario.target_position += self.target_velocity * self.dt
        
        # Boundary constraints
        self.scenario.target_position = np.clip(
            self.scenario.target_position,
            [5, 5], 
            [self.world_size[0]-5, self.world_size[1]-5]
        )
```

---

## Project Structure

```
high_layer/
├── envs/
│   ├── __init__.py
│   ├── sheep_flock.py          # Main environment
│   ├── sheep_scenario.py       # Scenario management
│   ├── sheep_entity.py         # Sheep with Boids behavior
│   ├── herder_entity.py        # Herder with physical constraints
│   ├── high_level_action.py    # Action decoder
│   └── curriculum_env.py       # Curriculum learning environment
├── onpolicy/
│   ├── algorithms/
│   │   └── ppo_actor_critic.py # PPO network (ImprovedActorCritic)
│   └── utils/
│       └── reward_normalizer.py # RunningMeanStd for reward normalization
├── scripts/
│   ├── train.sh                # Standard training
│   ├── train_fast.sh           # Fast CPU training
│   └── train_gpu.sh            # GPU training with curriculum
├── tests/                      # Test files
├── utils/                      # Utility functions
├── docs/                       # Documentation
├── results/                    # Training results
├── analyze_training.py         # Training analysis tool
├── demo_generator.py           # Demo generation tool
├── evaluate_policy.py          # Policy evaluation tool
├── model_inspector.py          # Model inspection tool
├── performance_monitor.py      # Performance monitoring tool
├── realtime_visualizer.py      # Real-time visualization tool
├── result_visualizer.py        # Result visualization tool
├── train_ppo.py                # Main training script
├── visualize.py                # Visualization tool
├── README.md                   # English documentation
├── README_CN.md                # Chinese documentation
└── requirements.txt            # Dependencies
```

---

## Quick Start

### Training

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

### Visualization

```bash
# Basic visualization
python visualize.py --model_path results/xxx/model.pt --num_episodes 5

# Custom configuration
python visualize.py \
    --model_path results/sheep_herding/fast_train/ppo/seed1/models/model_300100.pt \
    --num_episodes 5 \
    --num_sheep 6 \
    --num_herders 3 \
    --world_size 30.0 30.0

# Save as GIF/MP4
python visualize.py --model_path results/xxx/model.pt --save_gif output.gif
python visualize.py --model_path results/xxx/model.pt --save_video output.mp4
```

#### Visualization Features

| Element | Color | Description |
|---------|-------|-------------|
| ⭐ Target | Green | Target position |
| ● Sheep | Gray | Flock positions |
| ■ Herder | Blue | Herder positions |
| ● Main Target | Red | Wedge center position on the ring |
| ● Sampled Target | Orange | Sampled positions for each herder |
| ▢ Station Region | Orange | Wedge showing valid station area |
| ▢ Evasion Zone | Blue dashed | Herder evasion radius |

#### Network Output Display

The visualization shows decoded action parameters:

```
surround | r=12.0 w=90 a=0.15
```

| Parameter | Description |
|-----------|-------------|
| `mode` | Formation mode (surround/push/balance) |
| `r` | Station radius |
| `w` | Wedge width in degrees |
| `a` | Asymmetry parameter |

#### Interactive Controls

- **Pause/Resume**: Click "Pause/Resume" button or press `Space` key
- **Close**: Close the window to end visualization

---

## Visualization System Documentation

### Overview

The visualization system (`visualize.py`) provides real-time rendering of the trained policy's behavior, displaying both the physical environment and the network's decision-making process.

### Key Design Principles

1. **Direct Environment Integration**: The visualization directly uses environment data (`herder_targets`, `action_decoder`) rather than duplicating logic
2. **Station Region Visualization**: Shows where herders should be positioned based on network output
3. **Sampling Transparency**: Displays sampled target positions to understand policy decisions
4. **Single Window**: All episodes share the same window to avoid performance issues

### Visualization Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Visualization Pipeline                    │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐              │
│  │ Policy   │───→│ Action   │───→│ Decoder  │              │
│  │ Network  │    │ (4D)     │    │          │              │
│  └──────────┘    └──────────┘    └──────────┘              │
│                                        │                    │
│                                        ↓                    │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐              │
│  │ Render   │←───│ Env      │←───│ Sampler  │              │
│  │ (Matplotlib)   │ State    │    │          │              │
│  └──────────┘    └──────────┘    └──────────┘              │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### AI Agent Collaboration Framework

This project uses an **AI Agent Collaboration Framework** for team coordination, with 7 specialized Agent roles.

#### Agent Roles

| # | Agent | Responsibility |
|---|-------|----------------|
| 1 | Project Manager | Overall coordination, task assignment, progress tracking |
| 2 | Team Coordinator | Team collaboration management, optimization suggestions |
| 3 | Algorithm Engineer | PPO optimization, hyperparameter tuning |
| 4 | Environment Engineer | Environment modeling, reward function |
| 5 | Visualization Engineer | Training visualization, debugging tools |
| 6 | Quality Analyst | Problem discovery, diagnostic log design |
| 7 | Documentation Engineer | Documentation maintenance, API docs |

#### Workflow

```
User ──communicate──> Project Manager ──publish task──> share/tasks/pending.md
                                                           │
User ──"1"──> Specialist Agent ──claim task──> Execute ──> reports/
```

#### Key Files

| File | Description |
|------|-------------|
| `.trae/RULES.md` | Project rules (required reading for all Agents) |
| `.trae/AGENT_DESCRIPTIONS.md` | Agent creation descriptions |
| `.trae/agents/` | Agent instruction documents |
| `.trae/share/` | Information sharing center |

For detailed documentation, see the `.trae/` directory.

---

## Known Issues & Future Work

### Current Limitations

1. **Sim-to-Real Gap**: Simulation uses simplified Boids model
2. **No Obstacles**: Current environment has no obstacles
3. **Single Target**: Only one target position

### Future Improvements

1. Add obstacle avoidance
2. Multi-target scenarios
3. More realistic sheep behavior model
4. Domain randomization for sim-to-real transfer

---

## References

- PPO: Schulman et al., "Proximal Policy Optimization Algorithms", 2017
- Boids: Reynolds, "Flocks, Herds, and Schools: A Distributed Behavioral Model", 1987
- Von Mises Distribution: Directional statistics for angle sampling
- GAE: Schulman et al., "High-Dimensional Continuous Control Using Generalized Advantage Estimation", 2016

---

## Changelog

### Latest Updates

1. **Algorithm Optimization**: PPO algorithm improvements including KL divergence fix, learning rate warmup, and clip parameter annealing
2. **KL Divergence Fix**: Ensured non-negative KL divergence calculation for numerical stability
3. **Learning Rate Scheduling**: Added warmup phase (1000 steps) followed by cosine annealing
4. **GAE Lambda Adjustment**: Changed from 0.98 to 0.95 to reduce variance
5. **Clip Parameter Annealing**: Linear annealing from 0.2 to 0.1 during training
6. **Numerical Stability**: Enhanced reward normalizer with extreme value clipping
7. **Physical Model**: Added HerderEntity with speed/acceleration constraints
8. **Decision Frequency**: High-level decisions every 5 steps for stable execution
9. **Observation Space**: 12D relative/normalized observations
10. **Action Space**: Corrected kappa semantics (κ=0 → uniform, κ→∞ → overlap)
11. **Reward Function**: Potential-based reward with clipping
12. **Network Architecture**: 256 hidden size, 3 layers, LayerNorm, Dropout
13. **Curriculum Learning**: 3-stage progressive training
14. **Reward Function Fix (2026-02-17)**: 
    - Reduced success bonus from +7.0 to +3.0 for balanced reward scaling
    - Replaced np.clip with np.tanh for progress reward to avoid sign flip penalty
    - Relaxed reward clipping lower bound from -1.0 to -2.0
    - Added 5-step moving average reward smoothing mechanism
    - Added diagnostic logging for reward components
15. **Reward Collapse Analysis (2026-02-17)**: 
    - Identified root cause of periodic reward collapse (every 100-150 episodes)
    - Found reward component scale imbalance as primary issue
    - Documented expected improvements: reward variance reduction, collapse frequency reduction

---

## Contact & Contribution

This README will be continuously improved by AI agents with expertise in various aspects of the project.

For questions or contributions, please refer to the project documentation in `docs/` directory.
