"""
SheepFlockEnv: 羊群引导强化学习环境
实现Gym风格的多智能体环境接口
"""

import numpy as np
from typing import Tuple, List, Dict, Any, Optional, Union
from gym import spaces
from envs.sheep_scenario import SheepScenario
from envs.high_level_action import HighLevelAction


class SheepFlockEnv:
    """
    羊群引导多智能体环境
    
    高层控制器通过观测羊群状态，输出站位参数，
    指导机械狗围堵和引导羊群到达目标位置。
    """
    
    def __init__(
        self,
        world_size: Tuple[float, float] = (50.0, 50.0),
        num_sheep: int = 10,
        num_herders: int = 3,
        episode_length: int = 100,
        dt: float = 0.1,
        reward_config: Optional[Dict[str, float]] = None,
        random_seed: Optional[int] = None,
    ):
        """
        初始化环境
        
        Args:
            world_size: 世界大小
            num_sheep: 羊的数量
            num_herders: 机械狗数量
            episode_length: 每个episode的最大步数
            dt: 时间步长
            reward_config: 奖励配置
            random_seed: 随机种子
        """
        self.world_size = world_size
        self.num_sheep = num_sheep
        self.num_herders = num_herders
        self.episode_length = episode_length
        self.dt = dt
        self.random_seed = random_seed
        
        self.high_level_interval = 5
        self.step_count = 0
        
        self.reward_config = reward_config or {
            'distance_reward_weight': 1.0,
            'spread_penalty_weight': 0.1,
            'success_bonus': 10.0,
            'timeout_penalty': -5.0,
            'collision_penalty': -0.5,
            'time_penalty_weight': 1.0,
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
        self.prev_potential = None
        
        # 奖励平滑机制
        self.reward_history = []
        self.reward_smoothing_window = 5
        
        # 奖励组件诊断日志
        self._reward_components = {}
        
        self._seed = random_seed
        if random_seed is not None:
            np.random.seed(random_seed)
    
    def _setup_spaces(self):
        """Set up observation and action spaces"""
        self.obs_dim = 10
        self.action_dim = 4
        
        obs_low = np.array([0.0, -1.0, -1.0, 0.0, -1.0, -1.0, 0.0, -1.0, -1.0, 0.0], dtype=np.float32)
        obs_high = np.array([10.0, 1.0, 1.0, 1.0, 1.0, 1.0, 10.0, 1.0, 1.0, 2.0], dtype=np.float32)
        
        self.observation_space = spaces.Box(
            low=obs_low,
            high=obs_high,
            shape=(self.obs_dim,),
            dtype=np.float32,
        )
        
        action_low = np.array([-1.0, -1.0, -1.0, -1.0], dtype=np.float32)
        action_high = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
        
        self.action_space = spaces.Box(
            low=action_low,
            high=action_high,
            shape=(self.action_dim,),
            dtype=np.float32,
        )
        
        self.share_observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.obs_dim + self.num_herders * 2,),
            dtype=np.float32,
        )
    
    def reset(self) -> np.ndarray:
        """
        Reset environment
        
        Returns:
            Initial observation
        """
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
        self.prev_potential = None
        
        # Reset reward smoothing history
        self.reward_history = []
        self._reward_components = {}
        
        return self._get_obs()
    
    def get_shared_obs(self) -> np.ndarray:
        """
        获取共享观测
        
        Returns:
            共享观测，形状为 (num_herders, obs_dim)
        """
        return self.scenario.get_shared_observation()

    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Execute action
        
        High-level decision frequency control:
        - Only update herder targets every N steps (high_level_interval)
        - Herders move towards targets with physical constraints
        - Sheep update with continuous evasion force
        
        Args:
            actions: Action array, shape (num_herders, action_dim)
        
        Returns:
            obs: Observation
            reward: Reward
            done: Done flag
            info: Additional info
        """
        self.current_step += 1
        self.step_count += 1
        
        if self.step_count % self.high_level_interval == 1:
            target_positions = self._sample_herder_positions(actions)
            self.scenario.set_herder_targets(target_positions)
        
        self.scenario.update_herders(self.dt)
        
        self.scenario.update_sheep(self.dt)
        
        reward = self._compute_reward()
        
        done = self._check_done()
        
        info = self._get_info()
        
        obs = self._get_obs()
        
        return obs, reward, done, info
    
    def _sample_herder_positions(self, actions: np.ndarray) -> np.ndarray:
        """
        Sample herder target positions based on actions
        
        New Action Space (4D):
        - wedge_center: Formation center angle (relative to target direction)
        - wedge_width: Formation spread (0=push, 1=surround)
        - radius: Distance from flock center
        - asymmetry: Flank offset for steering
        
        Physical constraints:
        - Minimum distance between herders
        - Safe distance from flock boundary
        - World boundary constraints
        """
        flock_center = self.scenario.get_flock_center()
        target = self.scenario.get_target_position()
        
        target_direction = np.arctan2(
            target[1] - flock_center[1],
            target[0] - flock_center[0]
        )
        
        positions = np.zeros((self.num_herders, 2), dtype=np.float32)
        
        action = actions[0] if actions.ndim > 1 else actions
        
        decoded = self.action_decoder.decode_action(action, target_direction)
        
        angles, radii = self.action_decoder.sample_herder_positions(
            self.num_herders,
            **decoded
        )
        
        min_herder_distance = 2.0
        flock_safe_distance = 3.0
        
        for i in range(self.num_herders):
            pos = flock_center + np.array([
                radii[i] * np.cos(angles[i]),
                radii[i] * np.sin(angles[i])
            ])
            
            for j in range(i):
                diff = pos - positions[j]
                dist = np.linalg.norm(diff)
                if dist < min_herder_distance and dist > 0:
                    correction = diff / dist * (min_herder_distance - dist) / 2
                    pos = pos + correction
                    positions[j] = positions[j] - correction
            
            pos = np.clip(pos, [flock_safe_distance, flock_safe_distance], 
                         [self.world_size[0] - flock_safe_distance, 
                          self.world_size[1] - flock_safe_distance])
            
            positions[i] = pos
        
        return positions
    
    def _compute_reward(self) -> float:
        """
        Redesigned reward function for stable training convergence
        
        Design Principles:
        ==================
        1. SMOOTH REWARD SIGNALS: Use continuous, differentiable reward functions
        2. PROGRESSIVE REWARDS: Encourage incremental progress toward goals
        3. STABLE SCALING: Normalize all reward components to similar ranges
        4. BALANCED OBJECTIVES: Balance distance, formation, and efficiency
        5. PREDICTABLE REWARDS: Avoid discrete jumps and sparse signals
        
        Improvements (2026-02-17):
        ==========================
        - Reduced success bonus from +7.0 to +3.0 to balance reward scales
        - Use np.tanh for progress reward to avoid sign flip penalty
        - Relaxed reward clipping lower bound from -1.0 to -2.0
        - Added reward smoothing mechanism
        - Added diagnostic logging for reward components
        
        Reward Components:
        ==================
        1. Distance Progress Reward (dense, smooth)
        2. Formation Quality Reward (stable, normalized)
        3. Direction Alignment Reward (continuous)
        4. Efficiency Bonus (gentle time penalty)
        5. Success Bonus (moderate, not overwhelming)
        """
        reward = 0.0
        
        # Get current state
        current_distance = self.scenario.get_distance_to_target()
        max_distance = np.linalg.norm(self.world_size)
        max_distance = max(max_distance, 1.0)
        
        # ========================================
        # 1. SMOOTH DISTANCE REWARD (Core Signal)
        # ========================================
        # Use smooth exponential decay instead of discrete milestones
        distance_ratio = current_distance / max_distance
        
        # Exponential reward: r = exp(-k * d) gives smooth, bounded reward
        # k=3 gives reasonable decay: d=0 -> r=1.0, d=0.5 -> r=0.22, d=1.0 -> r=0.05
        distance_reward = np.exp(-3.0 * distance_ratio)
        reward += distance_reward * 1.0  # Scale to [0.05, 1.0]
        
        # ========================================
        # 2. DISTANCE PROGRESS REWARD (Shaping)
        # ========================================
        # Reward for making progress toward target
        progress_reward = 0.0
        if self.prev_distance is not None:
            distance_delta = self.prev_distance - current_distance
            # Normalize by max possible progress per step
            max_progress = 5.0 * self.dt  # Max herder speed * dt
            progress_ratio = distance_delta / max(max_progress, 0.1)
            # IMPROVEMENT: Use tanh instead of clip to avoid sign flip penalty
            # tanh provides smooth transition and avoids harsh clipping
            progress_reward = np.tanh(progress_ratio) * 0.3
            reward += progress_reward
        
        self.prev_distance = current_distance
        
        # ========================================
        # 3. FORMATION QUALITY REWARD (Stable)
        # ========================================
        # Flock spread: reward for maintaining reasonable spread
        spread = self.scenario.get_flock_spread()
        target_spread = 4.0  # Ideal spread
        spread_error = abs(spread - target_spread) / max(target_spread, 1.0)
        # Smooth penalty using exponential
        spread_penalty = -0.1 * (1.0 - np.exp(-spread_error))
        reward += spread_penalty
        
        # Flock cohesion: reward for keeping sheep together
        # Use spread variance instead of eigenvalue ratio (more stable)
        cohesion_bonus = 0.0
        if spread < 8.0:  # Reasonable spread
            cohesion_bonus = 0.1 * np.exp(-spread / 8.0)
            reward += cohesion_bonus
        
        # ========================================
        # 4. DIRECTION ALIGNMENT REWARD (Continuous)
        # ========================================
        flock_center = self.scenario.get_flock_center()
        target = self.scenario.get_target_position()
        flock_velocity = self.scenario.get_flock_direction()
        
        # Calculate flock speed (average velocity magnitude)
        velocities = np.array([s.velocity for s in self.scenario.sheep])
        mean_velocity = np.mean(velocities, axis=0)
        flock_speed = np.linalg.norm(mean_velocity)
        
        # Direction alignment (continuous, not thresholded)
        target_direction = target - flock_center
        target_direction_norm = np.linalg.norm(target_direction)
        
        alignment_reward = 0.0
        proximity_bonus = 0.0
        if target_direction_norm > 1e-6:
            target_unit = target_direction / target_direction_norm
            
            # Alignment reward: dot product of velocity and target direction
            # This is naturally in [-1, 1] range
            if flock_speed > 0.05:  # Lower threshold for more consistent signal
                alignment = np.dot(flock_velocity, target_unit)
                # Smooth alignment reward: only reward positive alignment
                alignment_reward = max(0.0, alignment) * 0.2
                reward += alignment_reward
            else:
                # Small bonus for stationary flock near target
                if distance_ratio < 0.2:
                    reward += 0.05
        
        # ========================================
        # 5. PROXIMITY BONUS (Progressive)
        # ========================================
        # Smooth proximity bonus using sigmoid-like function
        # This provides continuous reward as distance decreases
        proximity_bonus = 0.2 / (1.0 + np.exp(10.0 * (distance_ratio - 0.3)))
        reward += proximity_bonus
        
        # ========================================
        # 6. SUCCESS BONUS (Moderate - REDUCED)
        # ========================================
        # IMPROVEMENT: Reduced success bonus to prevent overwhelming other signals
        # Original: +5.0 base + +2.0 quick bonus = +7.0 total
        # New: +2.0 base + +1.0 quick bonus = +3.0 total
        success_bonus = 0.0
        if self.scenario.is_flock_at_target(threshold=5.0):
            # Moderate bonus, not too large to avoid policy collapse
            success_bonus = 2.0  # Reduced from 5.0
            reward += success_bonus
            # Additional bonus for quick success
            if self.current_step < self.episode_length * 0.5:
                quick_bonus = 1.0  # Reduced from 2.0
                reward += quick_bonus
                success_bonus += quick_bonus
        
        # ========================================
        # 7. EFFICIENCY PENALTY (Gentle)
        # ========================================
        # Very gentle time penalty to encourage efficiency
        # Not too strong to avoid confusing the agent
        time_penalty = -0.002
        reward += time_penalty
        
        # ========================================
        # 8. REWARD NORMALIZATION & SAFETY
        # ========================================
        # IMPROVEMENT: Relaxed lower bound from -1.0 to -2.0
        # This preserves more gradient information for learning
        # Expected range: [-1.5, 4.0] typically
        reward = np.clip(reward, -2.0, 10.0)
        
        # Safety check for numerical stability
        if np.isnan(reward) or np.isinf(reward):
            reward = 0.0
        
        # ========================================
        # 9. REWARD SMOOTHING (NEW)
        # ========================================
        # Apply exponential moving average smoothing
        self.reward_history.append(reward)
        if len(self.reward_history) > self.reward_smoothing_window:
            self.reward_history.pop(0)
        
        smoothed_reward = float(np.mean(self.reward_history))
        
        # ========================================
        # 10. DIAGNOSTIC LOGGING (NEW)
        # ========================================
        # Record reward components for analysis
        self._reward_components = {
            'distance_reward': float(distance_reward),
            'progress_reward': float(progress_reward),
            'spread_penalty': float(spread_penalty),
            'cohesion_bonus': float(cohesion_bonus),
            'alignment_reward': float(alignment_reward),
            'proximity_bonus': float(proximity_bonus),
            'success_bonus': float(success_bonus),
            'time_penalty': float(time_penalty),
            'raw_reward': float(reward),
            'smoothed_reward': smoothed_reward,
        }
        
        return smoothed_reward
    
    def _check_done(self) -> bool:
        """检查episode是否结束"""
        if self.scenario.is_flock_at_target(threshold=5.0):
            return True
        
        if self.current_step >= self.episode_length:
            return True
        
        return False
    
    def _get_obs(self) -> np.ndarray:
        """获取观测"""
        return self.scenario.get_observation()
    
    def _get_info(self) -> Dict[str, Any]:
        """获取额外信息"""
        return {
            'step': self.current_step,
            'distance_to_target': self.scenario.get_distance_to_target(),
            'flock_spread': self.scenario.get_flock_spread(),
            'is_success': self.scenario.is_flock_at_target(threshold=5.0),
            'flock_state': self.scenario.get_flock_state(),
            'reward_components': getattr(self, '_reward_components', {}),
        }
    
    def render(self, mode: str = 'human') -> Optional[np.ndarray]:
        """
        渲染环境
        
        Args:
            mode: 渲染模式，'human' 或 'rgb_array'
        """
        if mode == 'rgb_array':
            return self._render_rgb_array()
        else:
            self._render_human()
    
    def _render_human(self):
        """控制台渲染"""
        print(f"\n--- Step {self.current_step} ---")
        print(f"Flock center: {self.scenario.get_flock_center()}")
        print(f"Flock spread: {self.scenario.get_flock_spread():.2f}")
        print(f"Distance to target: {self.scenario.get_distance_to_target():.2f}")
        print(f"Herder positions: {self.scenario.get_herder_positions()}")
    
    def _render_rgb_array(self) -> np.ndarray:
        """生成RGB图像"""
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
        """关闭环境"""
        pass
    
    def seed(self, seed: Optional[int] = None):
        """设置随机种子"""
        self._seed = seed
        if seed is not None:
            np.random.seed(seed)
    
    def get_env_info(self) -> Dict[str, Any]:
        """获取环境信息"""
        return {
            'num_agents': self.num_herders,
            'obs_dim': self.obs_dim,
            'action_dim': self.action_dim,
            'episode_length': self.episode_length,
            'world_size': self.world_size,
        }


class SheepFlockEnvWrapper:
    """
    多进程环境包装器
    
    用于创建多个并行环境，符合MAPPO训练框架的接口
    """
    
    def __init__(
        self,
        num_envs: int = 1,
        **kwargs
    ):
        """
        初始化包装器
        
        Args:
            num_envs: 并行环境数量
            **kwargs: 传递给SheepFlockEnv的参数
        """
        self.num_envs = num_envs
        self.envs = [SheepFlockEnv(**kwargs) for _ in range(num_envs)]
        
        self.num_agents = self.envs[0].num_herders
        self.observation_space = self.envs[0].observation_space
        self.action_space = self.envs[0].action_space
        self.share_observation_space = self.envs[0].share_observation_space
    
    def reset(self) -> np.ndarray:
        """重置所有环境"""
        obs_list = [env.reset() for env in self.envs]
        return np.array(obs_list)
    
    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict]]:
        """
        在所有环境中执行动作
        
        Args:
            actions: 形状为 (num_envs, num_agents, action_dim) 的动作数组
        
        Returns:
            obs: 观测数组
            rewards: 奖励数组
            dones: 结束标志数组
            infos: 信息列表
        """
        obs_list = []
        rewards_list = []
        dones_list = []
        infos_list = []
        
        for i, env in enumerate(self.envs):
            env_actions = actions[i] if actions.ndim > 2 else actions
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
        """渲染第一个环境"""
        return self.envs[0].render(mode)
    
    def close(self):
        """关闭所有环境"""
        for env in self.envs:
            env.close()
    
    def seed(self, seed: Optional[int] = None):
        """设置随机种子"""
        for i, env in enumerate(self.envs):
            env.seed(seed + i if seed is not None else None)