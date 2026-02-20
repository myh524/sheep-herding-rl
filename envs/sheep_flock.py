"""
SheepFlockEnv: 羊群引导强化学习环境
实现Gym风格的多智能体环境接口

简化版：无物理限制，策略输出立即生效
支持离散动作空间
"""

import numpy as np
from typing import Tuple, List, Dict, Any, Optional, Union
from gym import spaces
from envs.sheep_scenario import SheepScenario
from envs.high_level_action import HighLevelAction


WEDGE_CENTER_VALUES = {
    "front": 1.0,
    "left": -0.5,
    "right": 0.5,
}

WEDGE_WIDTH_VALUES = {
    "push": -0.8,
    "half_surround": 0.0,
    "full_surround": 1.0,
}

RADIUS_VALUES = {
    "near": -0.6,
    "mid": 0.0,
    "far": 0.6,
}

DISCRETE_ACTIONS = {}
ACTION_INDEX = 0

for center_name, center_val in WEDGE_CENTER_VALUES.items():
    for width_name, width_val in WEDGE_WIDTH_VALUES.items():
        for radius_name, radius_val in RADIUS_VALUES.items():
            if width_name == "full_surround" and center_name != "front":
                continue
            
            action_name = f"{center_name.upper()}_{width_name.upper()}_{radius_name.upper()}"
            desc = f"{center_name}方向-{width_name}-{radius_name}距离"
            DISCRETE_ACTIONS[ACTION_INDEX] = {
                "name": action_name,
                "action": np.array([center_val, width_val, radius_val, 0.0], dtype=np.float32),
                "desc": desc,
                "center": center_name,
                "width": width_name,
                "radius": radius_name,
            }
            ACTION_INDEX += 1

NUM_DISCRETE_ACTIONS = len(DISCRETE_ACTIONS)


class SheepFlockEnv:
    """
    羊群引导多智能体环境（简化版）
    
    特点：
    - 无物理限制：机械狗直接瞬移到期望位置
    - 每步更新站位：策略输出立即生效
    - 简化奖励函数：直接奖励距离和进度
    - 支持离散动作空间：21个动作（去重后）
    """
    
    def __init__(
        self,
        world_size: Tuple[float, float] = (50.0, 50.0),
        num_sheep: int = 10,
        num_herders: int = 3,
        episode_length: int = 60,
        dt: float = 0.1,
        action_repeat: int = 5,
        reward_config: Optional[Dict[str, float]] = None,
        random_seed: Optional[int] = None,
    ):
        self.world_size = world_size
        self.num_sheep = num_sheep
        self.num_herders = num_herders
        self.episode_length = episode_length
        self.dt = dt
        self.action_repeat = action_repeat
        self.random_seed = random_seed
        
        self.step_count = 0
        
        self.reward_config = reward_config or {
            'distance_weight': 1.0,
            'progress_weight': 1.5,
            'spread_penalty_weight': 0.3,
            'success_bonus': 10.0,
            'quick_success_bonus': 5.0,
            'time_penalty': 0.02,
        }
        
        self.scenario = SheepScenario(
            world_size=world_size,
            num_sheep=num_sheep,
            num_herders=num_herders,
            random_seed=random_seed,
        )
        
        self.action_decoder = HighLevelAction(
            R_ref=np.linalg.norm(world_size) / 6.0,
            R_min=1.0,
            R_max=np.linalg.norm(world_size) / 2.0,
        )
        
        self._setup_spaces()
        
        self.current_step = 0
        self.prev_distance = None
        
        self._reward_components = {}
        
        self._seed = random_seed
        if random_seed is not None:
            np.random.seed(random_seed)
    
    def _setup_spaces(self):
        """Set up observation and action spaces"""
        self.obs_dim = 5
        
        obs_low = np.array([0.0, -1.0, -1.0, 0.0, 0.0], dtype=np.float32)
        obs_high = np.array([10.0, 1.0, 1.0, 1.0, 10.0], dtype=np.float32)
        
        self.observation_space = spaces.Box(
            low=obs_low,
            high=obs_high,
            shape=(self.obs_dim,),
            dtype=np.float32,
        )
        
        self.action_space = spaces.Discrete(NUM_DISCRETE_ACTIONS)
        self.action_dim = NUM_DISCRETE_ACTIONS
        
        self.share_observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.obs_dim + self.num_herders * 2,),
            dtype=np.float32,
        )
    
    def reset(self) -> np.ndarray:
        self.current_step = 0
        self.step_count = 0
        
        if self._seed is not None:
            np.random.seed(self._seed)
            self._seed = None
        
        target_pos = np.array([
            self.world_size[0] * np.random.uniform(0.7, 0.9),
            self.world_size[1] * np.random.uniform(0.3, 0.7)
        ])
        
        self.scenario.reset(target_position=target_pos)
        self.prev_distance = self.scenario.get_distance_to_target()
        
        self._reward_components = {}
        
        return self._get_obs()
    
    def get_shared_obs(self) -> np.ndarray:
        return self.scenario.get_shared_observation()

    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        self.current_step += 1
        self.step_count += 1
        
        action_idx = int(actions) if np.isscalar(actions) else int(actions[0])
        action = DISCRETE_ACTIONS[action_idx]["action"]
        
        target_positions = self._sample_herder_positions(action)
        self.scenario.set_herder_targets(target_positions)
        
        total_reward = 0.0
        reward_components_sum = {}
        done = False
        
        for sub_step in range(self.action_repeat):
            self.scenario.update_herders(self.dt)
            self.scenario.update_sheep(self.dt)
            
            reward = self._compute_reward()
            total_reward += reward
            
            for key, value in self._reward_components.items():
                if key not in reward_components_sum:
                    reward_components_sum[key] = 0.0
                reward_components_sum[key] += value
            
            if self._check_done():
                done = True
                break
        
        avg_reward = total_reward / (sub_step + 1)
        
        for key in reward_components_sum:
            if key != 'total_reward':
                reward_components_sum[key] = reward_components_sum[key] / (sub_step + 1)
        reward_components_sum['total_reward'] = avg_reward
        self._reward_components = reward_components_sum
        
        info = self._get_info()
        info['action_repeat_steps'] = sub_step + 1
        obs = self._get_obs()
        
        return obs, avg_reward, done, info
    
    def _sample_herder_positions(self, action: np.ndarray) -> np.ndarray:
        flock_center = self.scenario.get_flock_center()
        target = self.scenario.get_target_position()
        
        target_direction = np.arctan2(
            target[1] - flock_center[1],
            target[0] - flock_center[0]
        )
        
        positions = np.zeros((self.num_herders, 2), dtype=np.float32)
        
        decoded = self.action_decoder.decode_action(action, target_direction)
        
        angles, radii = self.action_decoder.sample_herder_positions(
            self.num_herders,
            **decoded
        )
        
        for i in range(self.num_herders):
            pos = flock_center + np.array([
                radii[i] * np.cos(angles[i]),
                radii[i] * np.sin(angles[i])
            ])
            
            positions[i] = pos
        
        return positions
    
    def _compute_reward(self) -> float:
        reward = 0.0
        
        current_distance = self.scenario.get_distance_to_target()
        max_distance = np.linalg.norm(self.world_size)
        max_distance = max(max_distance, 1.0)
        
        cfg = self.reward_config
        distance_ratio = current_distance / max_distance
        
        distance_reward = -distance_ratio * cfg.get('distance_weight', 2.0)
        reward += distance_reward
        
        progress_reward = 0.0
        if self.prev_distance is not None:
            progress = (self.prev_distance - current_distance) / max_distance
            progress_reward = progress * cfg.get('progress_weight', 3.0)
            reward += progress_reward
        
        self.prev_distance = current_distance
        
        spread = self.scenario.get_flock_spread()
        ideal_spread = 5.0
        if spread > ideal_spread:
            spread_penalty = -cfg.get('spread_penalty_weight', 0.1) * (spread - ideal_spread) / 10.0
            reward += spread_penalty
        else:
            spread_penalty = 0.0
        
        success_bonus = 0.0
        if self.scenario.is_flock_at_target(threshold=5.0):
            success_bonus = cfg.get('success_bonus', 10.0)
            reward += success_bonus
            
            if self.current_step < self.episode_length * 0.5:
                quick_bonus = cfg.get('quick_success_bonus', 5.0)
                reward += quick_bonus
                success_bonus += quick_bonus
        
        time_penalty = -cfg.get('time_penalty', 0.01)
        reward += time_penalty
        
        reward = np.clip(reward, -5.0, 20.0)
        
        if np.isnan(reward) or np.isinf(reward):
            reward = 0.0
        
        self._reward_components = {
            'distance_reward': float(distance_reward),
            'progress_reward': float(progress_reward),
            'spread_penalty': float(spread_penalty),
            'success_bonus': float(success_bonus),
            'time_penalty': float(time_penalty),
            'total_reward': float(reward),
        }
        
        return float(reward)
    
    def _check_done(self) -> bool:
        if self.scenario.is_flock_at_target(threshold=5.0):
            return True
        
        if self.current_step >= self.episode_length:
            return True
        
        return False
    
    def _get_obs(self) -> np.ndarray:
        return self.scenario.get_observation()
    
    def _get_info(self) -> Dict[str, Any]:
        return {
            'step': self.current_step,
            'distance_to_target': self.scenario.get_distance_to_target(),
            'flock_spread': self.scenario.get_flock_spread(),
            'is_success': self.scenario.is_flock_at_target(threshold=5.0),
            'flock_state': self.scenario.get_flock_state(),
            'reward_components': getattr(self, '_reward_components', {}),
        }
    
    def render(self, mode: str = 'human') -> Optional[np.ndarray]:
        if mode == 'rgb_array':
            return self._render_rgb_array()
        else:
            self._render_human()
    
    def _render_human(self):
        print(f"\n--- Step {self.current_step} ---")
        print(f"Flock center: {self.scenario.get_flock_center()}")
        print(f"Flock spread: {self.scenario.get_flock_spread():.2f}")
        print(f"Distance to target: {self.scenario.get_distance_to_target():.2f}")
        print(f"Herder positions: {self.scenario.get_herder_positions()}")
    
    def _render_rgb_array(self) -> np.ndarray:
        img_size = 400
        scale = img_size / max(self.world_size)
        
        img = np.ones((img_size, img_size, 3), dtype=np.uint8) * 240
        
        target = self.scenario.get_target_position()
        tx, ty = int(target[0] * scale), int(target[1] * scale)
        img[max(0, ty-10):min(img_size, ty+10), max(0, tx-10):min(img_size, tx+10)] = [0, 200, 0]
        
        flock_state = self.scenario.get_flock_state()
        for pos in flock_state['positions']:
            px, py = int(pos[0] * scale), int(pos[1] * scale)
            if 0 <= px < img_size and 0 <= py < img_size:
                img[max(0, py-3):min(img_size, py+3), max(0, px-3):min(img_size, px+3)] = [200, 200, 200]
        
        for hpos in self.scenario.get_herder_positions():
            hx, hy = int(hpos[0] * scale), int(hpos[1] * scale)
            if 0 <= hx < img_size and 0 <= hy < img_size:
                img[max(0, hy-5):min(img_size, hy+5), max(0, hx-5):min(img_size, hx+5)] = [0, 0, 200]
        
        return img
    
    def close(self):
        pass
    
    def seed(self, seed: Optional[int] = None):
        self._seed = seed
        if seed is not None:
            np.random.seed(seed)
    
    def get_env_info(self) -> Dict[str, Any]:
        return {
            'num_agents': self.num_herders,
            'obs_dim': self.obs_dim,
            'action_dim': self.action_dim,
            'episode_length': self.episode_length,
            'world_size': self.world_size,
            'discrete_actions': self.discrete_actions,
        }
    
    @staticmethod
    def get_discrete_action_name(action_idx: int) -> str:
        if 0 <= action_idx < NUM_DISCRETE_ACTIONS:
            return DISCRETE_ACTIONS[action_idx]["name"]
        return "UNKNOWN"
    
    @staticmethod
    def get_discrete_action_desc(action_idx: int) -> str:
        if 0 <= action_idx < NUM_DISCRETE_ACTIONS:
            return DISCRETE_ACTIONS[action_idx]["desc"]
        return "未知动作"
    
    @staticmethod
    def get_action_index(center: str, width: str, radius: str) -> int:
        for idx, action_info in DISCRETE_ACTIONS.items():
            if (action_info["center"] == center and 
                action_info["width"] == width and 
                action_info["radius"] == radius):
                return idx
        return 0
    
    @staticmethod
    def print_action_table():
        print("\n" + "="*80)
        print("离散动作表 (共{}个动作)".format(NUM_DISCRETE_ACTIONS))
        print("="*80)
        
        print("\n【参数说明】")
        print("-"*60)
        print("方向 (wedge_center):")
        print("  front  = 1.0  (羊群后方，推羊朝目标走)")
        print("  left   = -0.5 (左侧)")
        print("  right  = 0.5  (右侧)")
        print("\n展开 (wedge_width):")
        print("  push          = -0.8 (推进队形，稍微分开)")
        print("  half_surround = 0.0  (半包围 180°)")
        print("  full_surround = 1.0  (全包围 360°均匀)")
        print("\n距离 (radius):")
        print("  near = -0.6 (近距离站位)")
        print("  mid  = 0.0  (中距离站位)")
        print("  far  = 0.6  (远距离站位)")
        
        print("\n【注意】全包围时方向无意义，只保留front方向")
        
        print("\n" + "="*80)
        print("【动作列表】")
        print("="*80)
        
        current_width = None
        for idx in range(NUM_DISCRETE_ACTIONS):
            action_info = DISCRETE_ACTIONS[idx]
            
            if action_info["width"] != current_width:
                current_width = action_info["width"]
                print(f"\n--- {current_width.upper()} 模式 ---")
            
            action_str = ", ".join([f"{v:.1f}" for v in action_info["action"]])
            print(f"  [{idx:2d}] {action_info['name']:30s} | [{action_str}]")
        
        print("\n" + "="*80)


class SheepFlockEnvWrapper:
    def __init__(
        self,
        num_envs: int = 1,
        **kwargs
    ):
        self.num_envs = num_envs
        self.envs = [SheepFlockEnv(**kwargs) for _ in range(num_envs)]
        
        self.num_agents = self.envs[0].num_herders
        self.observation_space = self.envs[0].observation_space
        self.action_space = self.envs[0].action_space
        self.share_observation_space = self.envs[0].share_observation_space
    
    def reset(self) -> np.ndarray:
        obs_list = [env.reset() for env in self.envs]
        return np.array(obs_list)
    
    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict]]:
        obs_list = []
        rewards_list = []
        dones_list = []
        infos_list = []
        
        for i, env in enumerate(self.envs):
            env_actions = actions[i] if actions.ndim > 1 else actions
            obs, reward, done, info = env.step(env_actions)
            obs_list.append(obs)
            rewards_list.append(reward)
            dones_list.append(done)
            infos_list.append(info)
        
        return (
            np.array(obs_list),
            np.array(rewards_list),
            np.array(dones_list),
            infos_list,
        )
    
    def render(self, mode: str = 'human'):
        return self.envs[0].render(mode)
    
    def close(self):
        for env in self.envs:
            env.close()
    
    def seed(self, seed: Optional[int] = None):
        for i, env in enumerate(self.envs):
            env.seed(seed + i if seed is not None else None)


if __name__ == "__main__":
    SheepFlockEnv.print_action_table()
