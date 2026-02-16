# Physics Model Improvement Checklist

## HerderEntity Implementation

- [x] HerderEntity class created in envs/herder_entity.py
- [x] position, velocity, target_position attributes implemented
- [x] set_target() method implemented
- [x] update() method with physical constraints implemented
- [x] max_speed = 3.0
- [x] max_accel = 1.0

## SheepScenario Integration

- [x] herder_positions array replaced with HerderEntity list
- [x] get_herder_positions() returns positions from entities
- [x] set_herder_targets() sets targets instead of positions
- [x] update_herders() updates herder positions with physical constraints

## SheepFlockEnv Integration

- [x] high_level_interval parameter added (default: 5)
- [x] Decision frequency control implemented
- [x] step() method updated
- [x] step_count tracks steps for decision frequency

## Speed Parameters

- [x] Sheep max_speed = 2.0
- [x] Sheep max_force = 0.2
- [x] Herder max_speed = 3.0
- [x] Herder max_accel = 1.0

## Visualization Verification

- [ ] Herder positions change smoothly (no flickering) - 待验证
- [ ] Sheep evasion force is continuous - 待验证
- [ ] Sheep move towards target in reasonable time - 待验证
- [ ] 30-unit world traversable in ~50 steps - 待验证
