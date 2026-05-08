"""
SheepEntity: 羊群中单个羊的实体类
实现Boids模型的基本行为规则
"""

import numpy as np
from typing import List, Optional


class SheepEntity:
    """
    单个羊实体类
    
    属性:
        position: 羊的位置 [x, y]
        velocity: 羊的速度 [vx, vy]
        max_speed: 最大速度
        max_force: 最大转向力
        perception_radius: 感知半径
        velocity_drag: 全局速度阻尼 (1/s)，无持续受力时渐停
        evasion_radius: 感知到机械狗并产生逃避力的距离（世界坐标单位）
    """
    
    def __init__(
        self,
        position: np.ndarray,
        velocity: Optional[np.ndarray] = None,
        max_speed: float = 1.0,
        max_force: float = 0.1,
        perception_radius: float = 5.0,
        separation_radius: float = 2.0,
        velocity_drag: float = 0.55,
        evasion_radius: float = 8.0,
    ):
        self.position = np.array(position, dtype=np.float32)
        
        if velocity is None:
            angle = np.random.uniform(0, 2 * np.pi)
            speed = np.random.uniform(0, max_speed * 0.5)
            self.velocity = np.array([
                np.cos(angle) * speed,
                np.sin(angle) * speed
            ], dtype=np.float32)
        else:
            self.velocity = np.array(velocity, dtype=np.float32)
        
        self.max_speed = max_speed
        self.max_force = max_force
        self.perception_radius = perception_radius
        self.separation_radius = separation_radius
        self.velocity_drag = velocity_drag
        self.evasion_radius = evasion_radius
        
        self.acceleration = np.zeros(2, dtype=np.float32)
    
    def update(self, dt: float = 0.2):
        """更新羊的位置和速度"""
        self.velocity += self.acceleration * dt
        
        if self.velocity_drag > 0.0:
            self.velocity *= np.float32(np.exp(-self.velocity_drag * dt))
        
        speed = np.linalg.norm(self.velocity)
        if speed > self.max_speed:
            self.velocity = self.velocity / speed * self.max_speed
        
        self.position += self.velocity * dt
        self.acceleration = np.zeros(2, dtype=np.float32)
    
    def apply_force(self, force: np.ndarray):
        """应用转向力"""
        self.acceleration += force
    
    def separation(self, neighbors: List['SheepEntity']) -> np.ndarray:
        """
        分离规则: 避免与邻近羊碰撞
        """
        if not neighbors:
            return np.zeros(2, dtype=np.float32)
        
        steer = np.zeros(2, dtype=np.float32)
        count = 0
        
        sep_sq = self.separation_radius * self.separation_radius
        for other in neighbors:
            diff = self.position - other.position
            dist_sq = float(diff[0] * diff[0] + diff[1] * diff[1])
            if dist_sq <= 0.0 or dist_sq >= sep_sq:
                continue
            distance = np.sqrt(dist_sq)
            diff = diff / (distance * distance + 1e-6)
            steer = steer + diff
            count += 1
        
        if count > 0:
            steer = steer / count
            if np.linalg.norm(steer) > 0:
                steer = steer / np.linalg.norm(steer) * self.max_speed
                steer = steer - self.velocity
                if np.linalg.norm(steer) > self.max_force:
                    steer = steer / np.linalg.norm(steer) * self.max_force
        
        return steer
    
    def alignment(self, neighbors: List['SheepEntity']) -> np.ndarray:
        """
        对齐规则: 与邻近羊保持相同方向
        """
        if not neighbors:
            return np.zeros(2, dtype=np.float32)
        
        avg_velocity = np.zeros(2, dtype=np.float32)
        count = 0
        per_sq = self.perception_radius * self.perception_radius

        for other in neighbors:
            diff = self.position - other.position
            dist_sq = float(diff[0] * diff[0] + diff[1] * diff[1])
            if dist_sq <= 0.0 or dist_sq >= per_sq:
                continue
            avg_velocity += other.velocity
            count += 1
        
        if count > 0:
            avg_velocity = avg_velocity / count
            if np.linalg.norm(avg_velocity) > 0:
                avg_velocity = avg_velocity / np.linalg.norm(avg_velocity) * self.max_speed
                steer = avg_velocity - self.velocity
                if np.linalg.norm(steer) > self.max_force:
                    steer = steer / np.linalg.norm(steer) * self.max_force
                return steer
        
        return np.zeros(2, dtype=np.float32)
    
    def cohesion(self, neighbors: List['SheepEntity']) -> np.ndarray:
        """
        聚合规则: 向邻近羊群的中心移动
        """
        if not neighbors:
            return np.zeros(2, dtype=np.float32)
        
        center = np.zeros(2, dtype=np.float32)
        count = 0
        per_sq = self.perception_radius * self.perception_radius

        for other in neighbors:
            diff = self.position - other.position
            dist_sq = float(diff[0] * diff[0] + diff[1] * diff[1])
            if dist_sq <= 0.0 or dist_sq >= per_sq:
                continue
            center += other.position
            count += 1
        
        if count > 0:
            center = center / count
            return self.seek(center)
        
        return np.zeros(2, dtype=np.float32)
    
    def evasion(self, herders: List[np.ndarray]) -> np.ndarray:
        """
        逃避规则: 远离机械狗（作用距离为 self.evasion_radius）
        """
        if not herders:
            return np.zeros(2, dtype=np.float32)
        
        steer = np.zeros(2, dtype=np.float32)
        count = 0
        r = float(self.evasion_radius)
        ev_sq = r * r

        for herder_pos in herders:
            diff = self.position - herder_pos
            dist_sq = float(diff[0] * diff[0] + diff[1] * diff[1])
            if dist_sq <= 0.0 or dist_sq >= ev_sq:
                continue
            distance = np.sqrt(dist_sq)
            diff = diff / (distance * distance + 1e-6)
            steer += diff
            count += 1
        
        if count > 0:
            steer = steer / count
            if np.linalg.norm(steer) > 0:
                steer = steer / np.linalg.norm(steer) * self.max_speed * 1.5
                steer = steer - self.velocity
                if np.linalg.norm(steer) > self.max_force * 2:
                    steer = steer / np.linalg.norm(steer) * self.max_force * 2
        
        return steer
    
    def seek(self, target: np.ndarray) -> np.ndarray:
        """向目标位置移动"""
        desired = target - self.position
        distance = np.linalg.norm(desired)
        
        if distance > 0:
            desired = desired / distance * self.max_speed
            steer = desired - self.velocity
            if np.linalg.norm(steer) > self.max_force:
                steer = steer / np.linalg.norm(steer) * self.max_force
            return steer
        
        return np.zeros(2, dtype=np.float32)
    
    def boundary_force(self, world_radius: float, margin: float = 2.0) -> np.ndarray:
        """圆形场地：r > R−margin 时受指向圆心的边界力。"""
        steer = np.zeros(2, dtype=np.float32)
        p = self.position.astype(np.float32)
        r = float(np.linalg.norm(p))
        lim = float(world_radius) - float(margin)
        if r > lim and r > 1e-6:
            inward = (-p / np.float32(r)).astype(np.float32)
            steer = inward * np.float32(self.max_force)
        return steer
    
    def get_neighbors(self, all_sheep: List['SheepEntity']) -> List['SheepEntity']:
        """获取感知范围内的邻居"""
        neighbors = []
        per_sq = self.perception_radius * self.perception_radius
        for other in all_sheep:
            if other is not self:
                diff = self.position - other.position
                dist_sq = float(diff[0] * diff[0] + diff[1] * diff[1])
                if dist_sq < per_sq:
                    neighbors.append(other)
        return neighbors
    
    def apply_boids_rules(
        self,
        all_sheep: List['SheepEntity'],
        herders: List[np.ndarray],
        world_radius: float,
        weights: Optional[dict] = None,
        apply_boundary: bool = True,
    ):
        """
        应用所有Boids规则
        
        weights: 各规则的权重
            - separation: 分离权重
            - alignment: 对齐权重
            - cohesion: 聚合权重
            - evasion: 逃避权重
            - boundary: 边界权重
        """
        w = weights or {}
        neighbors = self.get_neighbors(all_sheep)

        sep_force = self.separation(neighbors) * w.get('separation', 1.0)
        ali_force = self.alignment(neighbors) * w.get('alignment', 1.0)
        coh_force = self.cohesion(neighbors) * w.get('cohesion', 1.0)
        eva_force = self.evasion(herders) * w.get('evasion', 1.0)
        if apply_boundary:
            bnd_force = self.boundary_force(world_radius) * w.get('boundary', 1.0)
        else:
            bnd_force = np.zeros(2, dtype=np.float32)

        self.apply_force(sep_force + ali_force + coh_force + eva_force + bnd_force)
    
    def __repr__(self):
        return f"SheepEntity(pos={self.position}, vel={self.velocity})"