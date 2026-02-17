"""
羊群引导环境单元测试
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from envs import SheepEntity, SheepScenario, SheepFlockEnv, SheepFlockEnvWrapper


def test_sheep_entity():
    """测试羊实体类"""
    print("Testing SheepEntity...")
    
    sheep = SheepEntity(position=np.array([10.0, 10.0]))
    
    assert sheep.position.shape == (2,)
    assert sheep.velocity.shape == (2,)
    assert sheep.max_speed > 0
    
    sheep.update(dt=0.1)
    
    print("SheepEntity test passed!")


def test_sheep_scenario():
    """测试场景类"""
    print("Testing SheepScenario...")
    
    scenario = SheepScenario(
        world_size=(50.0, 50.0),
        num_sheep=10,
        num_herders=3,
    )
    
    assert len(scenario.sheep) == 10
    assert scenario.herder_positions.shape == (3, 2)
    assert scenario.target_position.shape == (2,)
    
    center = scenario.get_flock_center()
    assert center.shape == (2,)
    
    spread = scenario.get_flock_spread()
    assert spread >= 0
    
    direction = scenario.get_flock_direction()
    assert direction.shape == (2,)
    
    scenario.update_sheep(dt=0.1)
    
    print("SheepScenario test passed!")


def test_sheep_flock_env():
    """测试环境类"""
    print("Testing SheepFlockEnv...")
    
    env = SheepFlockEnv(
        world_size=(50.0, 50.0),
        num_sheep=10,
        num_herders=3,
        episode_length=50,
    )
    
    assert env.observation_space.shape == (10,)
    assert env.action_space.shape == (4,)
    
    obs = env.reset()
    assert obs.shape == (10,)
    
    actions = np.random.uniform(
        env.action_space.low,
        env.action_space.high,
        (env.num_herders, env.action_dim)
    )
    
    obs, reward, done, info = env.step(actions)
    
    assert obs.shape == (10,)
    assert isinstance(reward, float)
    assert isinstance(done, bool)
    assert isinstance(info, dict)
    
    print("SheepFlockEnv test passed!")


def test_env_wrapper():
    """测试环境包装器"""
    print("Testing SheepFlockEnvWrapper...")
    
    wrapper = SheepFlockEnvWrapper(num_envs=2)
    
    assert wrapper.num_envs == 2
    assert len(wrapper.envs) == 2
    
    obs = wrapper.reset()
    assert obs.shape == (2, 10)
    
    actions = np.random.uniform(
        wrapper.action_space.low,
        wrapper.action_space.high,
        (2, wrapper.num_agents, wrapper.action_space.shape[0])
    )
    
    obs, rewards, dones, infos = wrapper.step(actions)
    
    assert obs.shape == (2, 10)
    assert rewards.shape == (2,)
    assert dones.shape == (2,)
    
    print("SheepFlockEnvWrapper test passed!")


def test_full_episode():
    """测试完整episode"""
    print("Testing full episode...")
    
    env = SheepFlockEnv(episode_length=20)
    obs = env.reset()
    
    total_reward = 0
    done = False
    step = 0
    
    while not done:
        actions = np.random.uniform(
            env.action_space.low,
            env.action_space.high,
            (env.num_herders, env.action_dim)
        )
        obs, reward, done, info = env.step(actions)
        total_reward += reward
        step += 1
    
    print(f"Episode finished in {step} steps, total reward: {total_reward:.2f}")
    print("Full episode test passed!")


if __name__ == "__main__":
    test_sheep_entity()
    test_sheep_scenario()
    test_sheep_flock_env()
    test_env_wrapper()
    test_full_episode()
    
    print("\n" + "="*50)
    print("All tests passed!")
    print("="*50)