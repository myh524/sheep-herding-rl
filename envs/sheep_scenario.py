"""
SheepScenario: 羊群引导场景管理类
管理羊群、机械狗、目标位置等场景元素
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from envs.sheep_entity import SheepEntity


class SheepScenario:
    """
    场景管理类
    
    管理整个羊群引导场景，包括:
    - 羊群实体
    - 机械狗位置
    - 目标位置
    - 场景边界
    """
    
    def __init__(
        self,
        world_size: Tuple[float, float] = (50.0, 50.0),
        num_sheep: int = 10,
        num_herders: int = 3,
        target_position: Optional[np.ndarray] = None,
        sheep_config: Optional[Dict[str, Any]] = None,
        random_seed: Optional[int] = None,
    ):
        """
        初始化场景
        
        Args:
            world_size: 世界大小 (width, height)
            num_sheep: 羊的数量
            num_herders: 机械狗数量
            target_position: 目标位置，如果为None则随机生成
            sheep_config: 羊的配置参数
            random_seed: 随机种子
        """
        if random_seed is not None:
            np.random.seed(random_seed)
        
        self.world_size = world_size
        self.num_sheep = num_sheep
        self.num_herders = num_herders
        
        self.sheep_config = sheep_config or {
            'max_speed': 3.0,
            'max_force': 0.3,
            'perception_radius': 5.0,
            'separation_radius': 2.0,
        }
        
        self.sheep: List[SheepEntity] = []
        self.herder_positions: np.ndarray = np.zeros((num_herders, 2), dtype=np.float32)
        self.target_position = target_position
        
        self.boids_weights = {
            'separation': 1.5,
            'alignment': 1.0,
            'cohesion': 1.0,
            'evasion': 2.0,
            'boundary': 1.0,
        }
        
        self._init_scenario()
    
    def _init_scenario(self):
        """初始化场景元素"""
        self._init_sheep()
        self._init_herders()
        self._init_target()
    
    def _init_sheep(self):
        """初始化羊群"""
        self.sheep = []
        center = np.array([self.world_size[0] * 0.3, self.world_size[1] * 0.5])
        spread = 5.0
        
        for _ in range(self.num_sheep):
            pos = center + np.random.uniform(-spread, spread, 2)
            pos = np.clip(pos, [1, 1], [self.world_size[0]-1, self.world_size[1]-1])
            
            sheep = SheepEntity(
                position=pos,
                max_speed=self.sheep_config['max_speed'],
                max_force=self.sheep_config['max_force'],
                perception_radius=self.sheep_config['perception_radius'],
                separation_radius=self.sheep_config['separation_radius'],
            )
            self.sheep.append(sheep)
    
    def _init_herders(self):
        """初始化机械狗位置"""
        self.herder_positions = np.zeros((self.num_herders, 2), dtype=np.float32)
        start_x = self.world_size[0] * 0.1
        spacing = self.world_size[1] / (self.num_herders + 1)
        
        for i in range(self.num_herders):
            self.herder_positions[i] = [
                start_x,
                spacing * (i + 1)
            ]
    
    def _init_target(self):
        """初始化目标位置"""
        if self.target_position is None:
            self.target_position = np.array([
                self.world_size[0] * 0.85,
                self.world_size[1] * 0.5
            ], dtype=np.float32)
        else:
            self.target_position = np.array(self.target_position, dtype=np.float32)
    
    def reset(
        self,
        target_position: Optional[np.ndarray] = None,
        random_seed: Optional[int] = None,
    ):
        """
        重置场景
        
        Args:
            target_position: 新的目标位置
            random_seed: 新的随机种子
        """
        if random_seed is not None:
            np.random.seed(random_seed)
        
        if target_position is not None:
            self.target_position = np.array(target_position, dtype=np.float32)
        
        self._init_sheep()
        self._init_herders()
    
    def set_herder_targets(self, targets: np.ndarray):
        """
        设置机械狗的目标位置
        
        Args:
            targets: 目标位置数组，形状为 (num_herders, 2)
        """
        targets = np.array(targets, dtype=np.float32)
        self.herder_targets = targets.copy()
    
    def update_herders(self, dt: float = 0.1):
        """
        更新机械狗位置（向目标移动，自动避开羊群）
        
        使用势场法：
        - 目标点产生吸引力
        - 羊群产生排斥力（当距离过近时）
        
        Args:
            dt: 时间步长
        """
        if not (hasattr(self, 'herder_targets') and self.herder_targets is not None):
            return
        
        flock_center = self.get_flock_center()
        flock_spread = self.get_flock_spread()
        avoid_radius = flock_spread * 2.5 + 3.0
        
        for i in range(self.num_herders):
            current_pos = self.herder_positions[i]
            target_pos = self.herder_targets[i]
            
            to_target = target_pos - current_pos
            target_dist = np.linalg.norm(to_target)
            
            if target_dist < 0.1:
                continue
            
            attract_force = to_target / target_dist
            
            to_flock = current_pos - flock_center
            flock_dist = np.linalg.norm(to_flock)
            
            repel_force = np.zeros(2, dtype=np.float32)
            if flock_dist < avoid_radius and flock_dist > 0.1:
                repel_strength = (avoid_radius - flock_dist) / avoid_radius
                repel_force = (to_flock / flock_dist) * repel_strength * 1.5
            
            move_direction = attract_force + repel_force
            move_norm = np.linalg.norm(move_direction)
            
            if move_norm > 0.1:
                move_direction = move_direction / move_norm
                self.herder_positions[i] += move_direction * min(5.0 * dt, target_dist)
            
            self.herder_positions[i] = np.clip(
                self.herder_positions[i],
                [0, 0],
                [self.world_size[0], self.world_size[1]]
            )
    
    def set_herder_positions(self, positions: np.ndarray):
        """
        直接设置机械狗位置
        
        Args:
            positions: 新的机械狗位置数组，形状为 (num_herders, 2)
        """
        positions = np.array(positions, dtype=np.float32)
        
        for i in range(min(len(positions), self.num_herders)):
            self.herder_positions[i] = np.clip(
                positions[i],
                [0, 0],
                [self.world_size[0], self.world_size[1]]
            )
    
    def update_sheep(self, dt: float = 0.1):
        """
        更新羊群状态
        
        应用Boids规则并更新每只羊的位置
        """
        herder_list = [self.herder_positions[i] for i in range(self.num_herders)]
        
        for sheep in self.sheep:
            sheep.apply_boids_rules(
                all_sheep=self.sheep,
                herders=herder_list,
                world_size=self.world_size,
                weights=self.boids_weights,
            )
            sheep.update(dt)
            
            sheep.position = np.clip(
                sheep.position,
                [0, 0],
                [self.world_size[0], self.world_size[1]]
            )
    
    def get_flock_center(self) -> np.ndarray:
        """获取羊群质心位置"""
        if not self.sheep:
            return np.zeros(2, dtype=np.float32)
        
        positions = np.array([s.position for s in self.sheep])
        return np.mean(positions, axis=0)
    
    def get_flock_spread(self) -> float:
        """获取羊群扩散度"""
        if len(self.sheep) < 2:
            return 0.0
        
        center = self.get_flock_center()
        positions = np.array([s.position for s in self.sheep])
        distances = np.linalg.norm(positions - center, axis=1)
        return float(np.std(distances))
    
    def get_flock_direction(self) -> np.ndarray:
        """获取羊群主方向"""
        if not self.sheep:
            return np.zeros(2, dtype=np.float32)
        
        velocities = np.array([s.velocity for s in self.sheep])
        avg_velocity = np.mean(velocities, axis=0)
        
        norm = np.linalg.norm(avg_velocity)
        if norm > 1e-6:
            return avg_velocity / norm
        return np.zeros(2, dtype=np.float32)
    
    def get_flock_state(self) -> Dict[str, Any]:
        """
        获取羊群状态信息
        
        Returns:
            包含质心、方向、扩散度等信息的字典
        """
        return {
            'center': self.get_flock_center(),
            'direction': self.get_flock_direction(),
            'spread': self.get_flock_spread(),
            'num_sheep': len(self.sheep),
            'positions': np.array([s.position for s in self.sheep]),
            'velocities': np.array([s.velocity for s in self.sheep]),
        }
    
    def get_flock_shape_eigenvalues(self) -> Tuple[float, float]:
        """
        计算羊群形状的特征值
        
        使用协方差矩阵的特征值描述羊群的椭圆形状:
        - λ1: 主轴长度（较大特征值）
        - λ2: 次轴长度（较小特征值）
        
        Returns:
            (lambda1, lambda2): 特征值元组，lambda1 >= lambda2
        """
        if len(self.sheep) < 2:
            return 1.0, 1.0
        
        positions = np.array([s.position for s in self.sheep])
        center = np.mean(positions, axis=0)
        centered = positions - center
        
        if np.isnan(centered).any():
            return 1.0, 1.0
        
        cov_matrix = np.cov(centered.T)
        
        if np.isnan(cov_matrix).any() or np.isinf(cov_matrix).any():
            return 1.0, 1.0
        
        eigenvalues = np.linalg.eigvalsh(cov_matrix)
        eigenvalues = np.sort(eigenvalues)[::-1]
        
        eigenvalues = np.maximum(eigenvalues, 0.1)
        
        return float(eigenvalues[0]), float(eigenvalues[1])
    
    def get_flock_main_direction(self) -> float:
        """
        获取羊群主方向角度（相对于世界坐标系）
        
        Returns:
            主方向角度（弧度），范围 [-π, π]
        """
        if len(self.sheep) < 2:
            return 0.0
        
        positions = np.array([s.position for s in self.sheep])
        center = np.mean(positions, axis=0)
        centered = positions - center
        
        cov_matrix = np.cov(centered.T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        
        max_idx = np.argmax(eigenvalues)
        main_axis = eigenvectors[:, max_idx]
        
        angle = np.arctan2(main_axis[1], main_axis[0])
        return float(angle)
    
    def get_high_state(self) -> np.ndarray:
        """
        获取相对化/归一化的高层观测向量
        
        观测向量结构 (8维):
        - [0] 到目标距离归一化 (d_goal / r_max)
        - [1] 目标方向角相对化 (θ_target - θ_main)，范围 [-π, π]
        - [2] 羊群形状特征值 λ1 归一化
        - [3] 羊群形状特征值 λ2 归一化
        - [4] 羊群主方向相对化 (θ_main - θ_target)
        - [5] 当前站位半径归一化
        - [6] 羊群速度大小归一化
        - [7] 羊群速度方向相对化
        
        Returns:
            8维相对化观测向量
        """
        high_state = np.zeros(8, dtype=np.float32)
        
        flock_center = self.get_flock_center()
        target = self.target_position
        
        r_max = np.linalg.norm(self.world_size) / 2.0
        d_goal = np.linalg.norm(flock_center - target)
        high_state[0] = d_goal / r_max
        
        theta_target = np.arctan2(
            target[1] - flock_center[1],
            target[0] - flock_center[0]
        )
        
        lambda1, lambda2 = self.get_flock_shape_eigenvalues()
        high_state[2] = np.sqrt(lambda1) / r_max
        high_state[3] = np.sqrt(lambda2) / r_max
        
        theta_main = self.get_flock_main_direction()
        theta_main_rel = theta_main - theta_target
        theta_main_rel = np.arctan2(np.sin(theta_main_rel), np.cos(theta_main_rel))
        high_state[4] = theta_main_rel / np.pi
        
        herder_positions = self.get_herder_positions()
        herder_center = np.mean(herder_positions, axis=0)
        current_radius = np.linalg.norm(herder_center - flock_center)
        high_state[5] = current_radius / r_max
        
        flock_velocity = self.get_flock_direction()
        speed = np.linalg.norm(
            np.mean(np.array([s.velocity for s in self.sheep]), axis=0)
        )
        high_state[6] = speed / 2.0
        
        if np.linalg.norm(flock_velocity) > 1e-6:
            theta_vel = np.arctan2(flock_velocity[1], flock_velocity[0])
            theta_vel_rel = theta_vel - theta_target
            theta_vel_rel = np.arctan2(np.sin(theta_vel_rel), np.cos(theta_vel_rel))
            high_state[7] = theta_vel_rel / np.pi
        else:
            high_state[7] = 0.0
        
        return high_state
    
    def get_distance_to_target(self) -> float:
        """获取羊群质心到目标的距离"""
        center = self.get_flock_center()
        return float(np.linalg.norm(center - self.target_position))
    
    def is_flock_at_target(self, threshold: float = 5.0) -> bool:
        """检查羊群是否到达目标"""
        return self.get_distance_to_target() < threshold
    
    def get_herder_positions(self) -> np.ndarray:
        """获取所有机械狗位置"""
        return self.herder_positions.copy()
    
    def get_target_position(self) -> np.ndarray:
        """获取目标位置"""
        return self.target_position.copy()
    
    def get_observation(self) -> np.ndarray:
        """
        Get normalized observation vector (10-dimensional)
        
        Observation structure:
        - [0]: d_goal / r_max - distance from flock center to target (normalized)
        - [1:3]: cos φ, sin φ - direction from flock to target (unit vector)
        - [3]: flock_speed / max_speed - flock speed magnitude (normalized)
        - [4:6]: cos θ_vel, sin θ_vel - flock velocity direction (relative to target)
        - [6]: spread / r_max - flock spread (normalized)
        - [7:9]: cos θ_main, sin θ_main - flock main direction (relative to target)
        - [9]: num_sheep / 30.0 - flock size (normalized)
        
        Returns:
            10-dimensional normalized observation vector
        """
        obs = np.zeros(10, dtype=np.float32)
        
        flock_center = self.get_flock_center()
        target = self.target_position
        r_max = np.linalg.norm(self.world_size) / 2.0
        r_max = max(r_max, 1.0)
        
        to_target = target - flock_center
        d_goal = np.linalg.norm(to_target)
        obs[0] = np.clip(d_goal / r_max, 0.0, 10.0)
        
        if d_goal > 1e-6:
            direction = to_target / d_goal
            obs[1] = direction[0]
            obs[2] = direction[1]
        else:
            obs[1] = 1.0
            obs[2] = 0.0
        
        theta_target = np.arctan2(to_target[1], to_target[0])
        
        velocities = np.array([s.velocity for s in self.sheep])
        mean_velocity = np.mean(velocities, axis=0)
        flock_speed = np.linalg.norm(mean_velocity)
        max_sheep_speed = 3.0
        obs[3] = np.clip(flock_speed / max_sheep_speed, 0.0, 1.0)
        
        if flock_speed > 1e-6:
            vel_direction = mean_velocity / flock_speed
            theta_vel = np.arctan2(vel_direction[1], vel_direction[0])
            theta_vel_rel = theta_vel - theta_target
            theta_vel_rel = np.arctan2(np.sin(theta_vel_rel), np.cos(theta_vel_rel))
            obs[4] = np.cos(theta_vel_rel)
            obs[5] = np.sin(theta_vel_rel)
        else:
            obs[4] = 0.0
            obs[5] = 0.0
        
        spread = self.get_flock_spread()
        obs[6] = np.clip(spread / r_max, 0.0, 10.0)
        
        theta_main = self.get_flock_main_direction()
        theta_rel = theta_main - theta_target
        theta_rel = np.arctan2(np.sin(theta_rel), np.cos(theta_rel))
        obs[7] = np.cos(theta_rel)
        obs[8] = np.sin(theta_rel)
        
        obs[9] = np.clip(len(self.sheep) / 30.0, 0.0, 2.0)
        
        obs = np.nan_to_num(obs, nan=0.0, posinf=10.0, neginf=-10.0)
        
        return obs
    
    def get_shared_observation(self) -> np.ndarray:
        """
        Get shared observation vector (for centralized Critic)
        
        Uses 10-dim observation, extended with individual herder positions
        Shape: (obs_dim + num_herders * 2,) = (10 + 3 * 2,) = (16,)
        """
        obs = self.get_observation()
        
        shared_obs = list(obs)
        
        flock_center = self.get_flock_center()
        r_max = np.linalg.norm(self.world_size) / 2.0
        for i in range(self.num_herders):
            rel_pos = self.herder_positions[i] - flock_center
            shared_obs.extend(rel_pos / r_max)
        
        return np.array(shared_obs, dtype=np.float32)
    
    def set_boids_weights(self, weights: Dict[str, float]):
        """设置Boids规则权重"""
        self.boids_weights.update(weights)
    
    def __repr__(self):
        return (
            f"SheepScenario("
            f"world_size={self.world_size}, "
            f"num_sheep={self.num_sheep}, "
            f"num_herders={self.num_herders}, "
            f"target={self.target_position})"
        )