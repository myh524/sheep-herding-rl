# Tasks

## Phase 1: HerderEntity Implementation

- [ ] Task 1: Create HerderEntity class
  - [ ] SubTask 1.1: Create envs/herder_entity.py
  - [ ] SubTask 1.2: Implement position, velocity, target_position attributes
  - [ ] SubTask 1.3: Implement set_target() method
  - [ ] SubTask 1.4: Implement update() method with physical constraints

## Phase 2: Scenario Integration

- [ ] Task 2: Update SheepScenario
  - [ ] SubTask 2.1: Replace herder_positions array with HerderEntity list
  - [ ] SubTask 2.2: Update get_herder_positions() to return positions from entities
  - [ ] SubTask 2.3: Update update_herders() to set targets instead of positions
  - [ ] SubTask 2.4: Add herder update in update_sheep()

## Phase 3: Environment Integration

- [ ] Task 3: Update SheepFlockEnv
  - [ ] SubTask 3.1: Add high_level_interval parameter
  - [ ] SubTask 3.2: Implement decision frequency control
  - [ ] SubTask 3.3: Update step() method
  - [ ] SubTask 3.4: Update observation to include herder velocities

## Phase 4: Parameter Adjustment

- [ ] Task 4: Adjust speed parameters
  - [ ] SubTask 4.1: Increase sheep max_speed to 2.0
  - [ ] SubTask 4.2: Increase sheep max_force to 0.2
  - [ ] SubTask 4.3: Set herder max_speed to 3.0
  - [ ] SubTask 4.4: Set herder max_accel to 1.0

## Phase 5: Verification

- [ ] Task 5: Test and verify
  - [ ] SubTask 5.1: Run visualization to check smooth movement
  - [ ] SubTask 5.2: Verify herder positions are continuous
  - [ ] SubTask 5.3: Verify sheep evasion force is continuous
  - [ ] SubTask 5.4: Test training stability

# Task Dependencies

- Task 2 depends on Task 1
- Task 3 depends on Task 2
- Task 4 depends on Task 3
- Task 5 depends on Task 4
