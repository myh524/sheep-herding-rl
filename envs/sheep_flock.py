"""
SheepFlockEnv: 羊群引导强化学习环境
实现Gym风格的多智能体环境接口
"""

import numpy as np
from typing import Tuple, List, Dict, Any, Optional, Union
from gym import spaces
from envs.defaults import (
    DEFAULT_DT,
    DEFAULT_FLOCK_EPISODE_LENGTH,
    DEFAULT_NUM_HERDERS,
    DEFAULT_NUM_SHEEP,
    DEFAULT_WORLD_SIZE,
    FORMATION_DELTA_COVERAGE_MAX,
    FORMATION_DELTA_RADIUS_MAX,
    FORMATION_DELTA_THETA_MAX_RAD,
    default_reward_config,
    merged_herder_motion_config,
)
from envs.sheep_scenario import SheepScenario, clip_position_to_disk
from envs.high_level_action import (
    HighLevelAction,
    STANCE_RADIUS_MAX,
    STANCE_RADIUS_MIN,
)


class SheepFlockEnv:
    """
    羊群引导多智能体环境
    
    高层控制器通过观测羊群状态，输出站位参数，
    指导机械狗围堵和引导羊群到达目标位置。
    """
    
    def __init__(
        self,
        world_size: Tuple[float, float] = DEFAULT_WORLD_SIZE,
        num_sheep: int = DEFAULT_NUM_SHEEP,
        num_herders: int = DEFAULT_NUM_HERDERS,
        episode_length: int = DEFAULT_FLOCK_EPISODE_LENGTH,
        dt: float = DEFAULT_DT,
        reward_config: Optional[Dict[str, float]] = None,
        random_seed: Optional[int] = None,
        use_herder_kinematics: bool = True,
        end_episode_when_at_target: bool = False,
        info_include_flock_state: bool = False,
        formation_delta_mode: bool = False,
        formation_delta_theta_max_rad: float = FORMATION_DELTA_THETA_MAX_RAD,
        formation_delta_radius_max: float = FORMATION_DELTA_RADIUS_MAX,
        formation_delta_coverage_max: float = FORMATION_DELTA_COVERAGE_MAX,
        high_level_interval: Optional[int] = None,
        herder_motion: Optional[Dict[str, Any]] = None,
        herder_physics_legacy: bool = False,
    ):
        """
        初始化环境
        
        Args:
            world_size: (W,H)，圆形场地半径 R=min(W,H)/2，目标在圆心 (0,0)
            num_sheep: 羊的数量
            num_herders: 机械狗数量
            episode_length: 每个episode的最大步数
            dt: 时间步长
            reward_config: 稠密奖励系数（见 _compute_reward 内注释）
            random_seed: 随机种子
            use_herder_kinematics: False 时狗直接出现在编队目标（关闭势场运动学）
            end_episode_when_at_target: True 时羊群进入目标阈值后立刻结束（旧行为）；
                False 时始终跑满 episode_length，便于学习抵达后仍把羊群控制在目标附近
            info_include_flock_state: True 时 step 的 info 含完整 flock_state（耗 CPU）；调试/可视化可开
            formation_delta_mode: True 时 a[0:3] 为每档高层决策上的增量（θ_in、R、coverage），并增广观测 3 维
            formation_delta_theta_max_rad: |a[0]|≤1 时单档最大转角增量
            formation_delta_radius_max: |a[1]|≤1 时单档半径增量（米）
            formation_delta_coverage_max: |a[2]|≤1 时单档 coverage 增量
            high_level_interval: 每 N 个 env step 刷新编队目标；None 时增量模式默认 1，绝对动作默认 5
            herder_motion: 覆盖 envs.defaults 中机械狗运动/分配参数
            herder_physics_legacy: True 时等价于旧版固定初值、恒等槽位分配、无牧者间斥力
        """
        self.world_size = world_size
        self.num_sheep = num_sheep
        self.num_herders = num_herders
        self.episode_length = episode_length
        self.dt = dt
        self.random_seed = random_seed
        self.end_episode_when_at_target = end_episode_when_at_target
        self.info_include_flock_state = info_include_flock_state
        self.formation_delta_mode = bool(formation_delta_mode)
        self.formation_delta_theta_max_rad = float(formation_delta_theta_max_rad)
        self.formation_delta_radius_max = float(formation_delta_radius_max)
        self.formation_delta_coverage_max = float(formation_delta_coverage_max)

        self._formation_theta_in = 0.0
        self._formation_radius = float(
            (STANCE_RADIUS_MIN + STANCE_RADIUS_MAX) / 2.0
        )
        self._formation_coverage = 0.5

        if high_level_interval is None:
            self.high_level_interval = 1 if self.formation_delta_mode else 5
        else:
            self.high_level_interval = max(1, int(high_level_interval))
        self.step_count = 0
        
        self.reward_config = reward_config or default_reward_config()
        self.herder_motion = merged_herder_motion_config(
            overrides=herder_motion,
            legacy=bool(herder_physics_legacy),
        )

        self.scenario = SheepScenario(
            world_size=world_size,
            num_sheep=num_sheep,
            num_herders=num_herders,
            random_seed=random_seed,
            use_herder_kinematics=use_herder_kinematics,
            herder_motion=self.herder_motion,
        )
        
        self.action_decoder = HighLevelAction()
        
        self._setup_spaces()
        
        self.current_step = 0
        self.prev_d_hat = 1.0
        # 最近一次真正用于 set_herder_targets 的 decode（与 high_level_interval 对齐；供可视化）
        self._last_formation_decoded: Optional[Dict[str, Any]] = None
        
        # 奖励组件诊断日志
        self._reward_components = {}
        
        self._seed = random_seed
        if random_seed is not None:
            np.random.seed(random_seed)
    
    def _setup_spaces(self):
        """Set up observation and action spaces"""
        # 8 维羊群相关 + 2 维机械狗**质心**相对**羊质心**的极坐标（与 N 无关）
        # 增量编队模式再 +3：θ_in/π、R 归一化、coverage 归一化
        base_obs = 10
        self.obs_dim = base_obs + (3 if self.formation_delta_mode else 0)
        self.action_dim = 5

        obs_low = np.full(self.obs_dim, -10.0, dtype=np.float32)
        obs_low[2:4] = -1.0
        obs_low[9] = -1.0
        obs_high = np.full(self.obs_dim, 10.0, dtype=np.float32)
        obs_high[2:4] = 1.0
        obs_high[9] = 1.0
        if self.formation_delta_mode:
            obs_low[10:13] = -1.0
            obs_high[10:13] = 1.0
        
        self.observation_space = spaces.Box(
            low=obs_low,
            high=obs_high,
            shape=(self.obs_dim,),
            dtype=np.float32,
        )
        
        action_low = np.full(5, -1.0, dtype=np.float32)
        action_high = np.full(5, 1.0, dtype=np.float32)
        
        self.action_space = spaces.Box(
            low=action_low,
            high=action_high,
            shape=(self.action_dim,),
            dtype=np.float32,
        )
        
        # 主观测 + 各狗相对羊群质心的极坐标（每只 2 维，供中心化 Critic）
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
        
        self.scenario.reset(target_position=None)
        self._sync_prev_d_hat_after_reset()
        self._last_formation_decoded = None
        self._reset_formation_integrator()

        self._reward_components = {}

        return self._get_obs()
    
    def get_shared_obs(self) -> np.ndarray:
        """
        获取共享观测
        
        Returns:
            一维向量，长度 obs_dim + num_herders * 2（与 share_observation_space 一致）
        """
        base = self.scenario.get_shared_observation()
        if not self.formation_delta_mode:
            return base
        return np.concatenate([base, self._formation_obs_tail()], axis=0).astype(
            np.float32
        )

    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Execute action
        
        High-level decision frequency control:
        - 每 high_level_interval 个 env step 刷新编队目标（(step_count-1) % N == 0，故 N=1 时每步刷新）
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
        
        if (self.step_count - 1) % self.high_level_interval == 0:
            target_positions = self._sample_herder_positions(actions)
            self.scenario.set_herder_targets(target_positions)
        
        self.scenario.update_herders(self.dt)
        
        self.scenario.update_sheep(self.dt)
        
        reward = self._compute_reward()
        
        done = self._check_done()
        
        info = self._get_info()
        
        obs = self._get_obs()
        
        return obs, reward, done, info
    
    def _reset_formation_integrator(self) -> None:
        if not self.formation_delta_mode:
            return
        d = self.action_decoder
        self._formation_theta_in = 0.0
        self._formation_radius = float((d.R_min + d.R_max) / 2.0)
        self._formation_coverage = 0.5

    def _formation_obs_tail(self) -> np.ndarray:
        d = self.action_decoder
        span = max(float(d.R_max - d.R_min), 1e-6)
        r_n = 2.0 * (self._formation_radius - d.R_min) / span - 1.0
        r_n = float(np.clip(r_n, -1.0, 1.0))
        ti_n = float(np.clip(self._formation_theta_in / np.pi, -1.0, 1.0))
        cov_n = float(np.clip(self._formation_coverage * 2.0 - 1.0, -1.0, 1.0))
        return np.array([ti_n, r_n, cov_n], dtype=np.float32)

    def _sample_herder_positions(self, actions: np.ndarray) -> np.ndarray:
        """
        羊质心圆弧：5D 动作中仅 a[0]–a[2] 有效。
        默认：`a[0]·π`=θ_in（弧中点→羊质心），R、coverage 由 a[1],a[2] 直接映射。
        formation_delta_mode：a[0:3] 为每档高层决策上的增量，再 clip/wrap 后解码。

        Physical constraints:
        - Minimum distance between herders
        - Safe distance from flock boundary
        - World boundary constraints
        """
        action = actions[0] if actions.ndim > 1 else actions
        flock_center = self.scenario.get_flock_center()

        if self.formation_delta_mode:
            da = np.asarray(action[:3], dtype=np.float64)
            self._formation_theta_in += float(da[0]) * self.formation_delta_theta_max_rad
            self._formation_theta_in = float(
                np.arctan2(
                    np.sin(self._formation_theta_in),
                    np.cos(self._formation_theta_in),
                )
            )
            self._formation_radius += float(da[1]) * self.formation_delta_radius_max
            self._formation_radius = float(
                np.clip(
                    self._formation_radius,
                    self.action_decoder.R_min,
                    self.action_decoder.R_max,
                )
            )
            self._formation_coverage += float(da[2]) * self.formation_delta_coverage_max
            self._formation_coverage = float(np.clip(self._formation_coverage, 0.0, 1.0))
            decoded = self.action_decoder.decoded_from_formation_params(
                flock_center,
                self.world_size,
                self._formation_theta_in,
                self._formation_radius,
                self._formation_coverage,
            )
        else:
            decoded = self.action_decoder.decode_action(
                action,
                flock_center,
                self.world_size,
            )
        self._last_formation_decoded = {
            "flock_center": np.asarray(decoded["flock_center"], dtype=np.float32).copy(),
            "radius": float(decoded["radius"]),
            "coverage": float(decoded["coverage"]),
            "theta_in_rad": float(decoded["theta_in_rad"]),
            "theta_mid_rad": float(decoded["theta_mid_rad"]),
        }
        positions = self.action_decoder.sample_herder_positions(
            self.num_herders,
            decoded["flock_center"],
            decoded["radius"],
            decoded["coverage"],
            decoded["theta_mid_rad"],
        ).copy()
        
        min_herder_distance = 2.0
        flock_safe_distance = 3.0
        
        for i in range(self.num_herders):
            pos = positions[i].copy()
            
            for j in range(i):
                diff = pos - positions[j]
                dist = np.linalg.norm(diff)
                if dist < min_herder_distance and dist > 0:
                    correction = diff / dist * (min_herder_distance - dist) / 2
                    pos = pos + correction
                    positions[j] = positions[j] - correction
            
            R = float(self.scenario.world_radius)
            lim = max(R - flock_safe_distance, 0.5)
            pos = clip_position_to_disk(pos, lim)
            
            positions[i] = pos
        
        return positions

    @staticmethod
    def _flock_axis_envelope_area(
        sheep_positions: np.ndarray,
        target: np.ndarray,
    ) -> float:
        """
        与观测四向包络一致：e0..e3 为轴对齐支持值，AABB 面积 (e0+e2)(e1+e3)。
        sheep_positions: (N, 2)，target: (2,)
        """
        if sheep_positions.size == 0:
            return 0.0
        rel = sheep_positions.astype(np.float64) - target.astype(np.float64).reshape(1, 2)
        e0 = float(np.max(rel[:, 0]))
        e1 = float(np.max(rel[:, 1]))
        e2 = float(np.max(-rel[:, 0]))
        e3 = float(np.max(-rel[:, 1]))
        return float((e0 + e2) * (e1 + e3))

    def _sync_prev_d_hat_after_reset(self) -> None:
        """重置后根据当前质心距目标初始化归一化距离，供势函数差分用。"""
        r_scale = float(max(2.0 * self.scenario.world_radius, 1e-6))
        d = float(self.scenario.get_distance_to_target())
        self.prev_d_hat = float(np.clip(d / r_scale, 0.0, 1.0))

    def _compute_reward(self) -> float:
        """
        稠密奖励（无机械狗几何项）：
        - 势函数进度：w_pot * (prev_d_hat^2 - d_hat^2)，d_hat = d / (2*world_radius)
        - 近目标：w_near * exp(-d / d0)，d0 = near_d0_alpha * (2*world_radius)
        - 速度正则（近处更强）：-w_speed * tanh(|v_mean|/v_ref) * exp(-d/d0)
        - 形状：四向轴对齐包络的 AABB 面积 A=(e0+e2)(e1+e3)，归一化 A/r_scale^2 后 tanh 惩罚
        - 时间：质心距目标 < time_penalty_off_threshold_m 时为 0，否则 -time_penalty
        系数见 self.reward_config；返回 clip 后的标量（无滑动平均）。
        """
        cfg = self.reward_config
        r_scale = float(max(2.0 * self.scenario.world_radius, 1e-6))
        d = float(self.scenario.get_distance_to_target())
        d_hat = float(np.clip(d / r_scale, 0.0, 1.0))

        w_pot = float(cfg.get("w_potential", 3.0))
        r_pot = w_pot * (self.prev_d_hat ** 2 - d_hat ** 2)

        alpha = float(cfg.get("near_d0_alpha", 0.15))
        d0 = max(alpha * r_scale, 1e-3)
        w_near = float(cfg.get("w_near", 0.35))
        r_near = w_near * float(np.exp(-d / d0))

        max_spd = float(self.scenario.sheep_config.get("max_speed", 3.0))
        if max_spd <= 0:
            max_spd = 1.0
        v_ref = float(cfg.get("v_ref_scale", 1.0)) * max_spd
        if self.scenario.sheep:
            v_mean = np.mean(
                np.array([s.velocity for s in self.scenario.sheep], dtype=np.float64),
                axis=0,
            )
            v_mean_norm = float(np.linalg.norm(v_mean))
        else:
            v_mean_norm = 0.0
        w_spd = float(cfg.get("w_speed", 0.25))
        sigma_near = float(np.exp(-d / d0))
        r_spd = -w_spd * float(np.tanh(v_mean_norm / max(v_ref, 1e-6))) * sigma_near

        tgt = self.scenario.target_position.astype(np.float64)
        if self.scenario.sheep:
            pos_arr = np.array([s.position for s in self.scenario.sheep], dtype=np.float64)
            area_bb = self._flock_axis_envelope_area(pos_arr, tgt)
        else:
            area_bb = 0.0
        area_norm = float(area_bb / max(r_scale ** 2, 1e-6))
        w_spr = float(cfg.get("w_spread", 0.12))
        env_ref = float(cfg.get("envelope_area_ref", 0.12))
        r_spr = -w_spr * float(np.tanh(area_norm / max(env_ref, 1e-6)))

        c = float(cfg.get("time_penalty", 0.005))
        off_m = float(cfg.get("time_penalty_off_threshold_m", 5.0))
        if d < off_m:
            r_time = 0.0
        else:
            r_time = -c

        raw = r_pot + r_near + r_spd + r_spr + r_time
        lo = float(cfg.get("reward_clip_low", -3.0))
        hi = float(cfg.get("reward_clip_high", 5.0))
        reward = float(np.clip(raw, lo, hi))
        if np.isnan(reward) or np.isinf(reward):
            reward = 0.0

        self.prev_d_hat = d_hat

        self._reward_components = {
            "r_potential": float(r_pot),
            "r_near": float(r_near),
            "r_speed": float(r_spd),
            "r_envelope": float(r_spr),
            "r_time": float(r_time),
            "envelope_area": float(area_bb),
            "envelope_area_norm": float(area_norm),
            "d_hat": d_hat,
            "distance": d,
            "raw_reward": reward,
        }
        return reward
    
    def _check_done(self) -> bool:
        """检查 episode 是否结束：默认仅按步数截断，不因到达目标提前结束。"""
        if self.end_episode_when_at_target and self.scenario.is_flock_at_target(threshold=5.0):
            return True
        return self.current_step >= self.episode_length
    
    def _get_obs(self) -> np.ndarray:
        """获取观测"""
        base = self.scenario.get_observation()
        if not self.formation_delta_mode:
            return base
        return np.concatenate([base, self._formation_obs_tail()], axis=0).astype(
            np.float32
        )
    
    def _get_info(self) -> Dict[str, Any]:
        """获取额外信息"""
        info: Dict[str, Any] = {
            'step': self.current_step,
            'distance_to_target': self.scenario.get_distance_to_target(),
            'flock_spread': self.scenario.get_flock_spread(),
            'is_success': self.scenario.is_flock_at_target(threshold=5.0),
            'reward_components': getattr(self, '_reward_components', {}),
        }
        if self.info_include_flock_state:
            info['flock_state'] = self.scenario.get_flock_state()
        return info
    
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
        """生成RGB图像（圆心在原点，图像中心为 (0,0)）"""
        img_size = 400
        R = float(max(self.scenario.world_radius, 1e-6))
        pad = 20
        span = max(img_size - 2 * pad, 1)
        scale = span / (2.0 * R)
        cx = img_size / 2.0
        cy = img_size / 2.0

        def to_px(p: np.ndarray) -> Tuple[int, int]:
            return int(cx + float(p[0]) * scale), int(cy - float(p[1]) * scale)

        img = np.ones((img_size, img_size, 3), dtype=np.uint8) * 240
        
        target = self.scenario.get_target_position()
        tx, ty = to_px(target)
        img[max(0, ty - 10):min(img_size, ty + 10), max(0, tx - 10):min(img_size, tx + 10)] = [
            0,
            200,
            0,
        ]
        
        flock_state = self.scenario.get_flock_state()
        for pos in flock_state['positions']:
            px, py = to_px(pos)
            if 0 <= px < img_size and 0 <= py < img_size:
                img[max(0, py - 3):min(img_size, py + 3), max(0, px - 3):min(img_size, px + 3)] = [
                    200,
                    200,
                    200,
                ]
        
        for hpos in self.scenario.get_herder_positions():
            hx, hy = to_px(hpos)
            if 0 <= hx < img_size and 0 <= hy < img_size:
                img[max(0, hy - 5):min(img_size, hy + 5), max(0, hx - 5):min(img_size, hx + 5)] = [
                    0,
                    0,
                    200,
                ]
        
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
            'world_radius': float(self.scenario.world_radius),
            'formation_delta_mode': self.formation_delta_mode,
            'high_level_interval': self.high_level_interval,
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