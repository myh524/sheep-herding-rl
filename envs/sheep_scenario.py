"""
SheepScenario: 羊群引导场景管理类
管理羊群、机械狗、目标位置等场景元素
"""

import copy
from typing import List, Tuple, Optional, Dict, Any

import numpy as np
from scipy.optimize import linear_sum_assignment

from envs.defaults import (
    DEFAULT_NUM_HERDERS,
    DEFAULT_NUM_SHEEP,
    DEFAULT_WORLD_SIZE,
    default_boids_weights,
    default_herder_motion_config,
    default_sheep_config,
)
from envs.sheep_entity import SheepEntity


def _np_seek_to_position(
    pos: np.ndarray,
    target: np.ndarray,
    max_speed: float,
    max_force: float,
    vel: np.ndarray,
) -> np.ndarray:
    """与 SheepEntity.seek 一致的转向力（float32）。"""
    desired = target.astype(np.float32) - pos.astype(np.float32)
    distance = float(np.linalg.norm(desired))
    if distance <= 0:
        return np.zeros(2, dtype=np.float32)
    desired = desired / distance * np.float32(max_speed)
    steer = desired - vel
    sn = float(np.linalg.norm(steer))
    if sn > max_force:
        steer = (steer / np.float32(sn) * np.float32(max_force)).astype(np.float32)
    return steer.astype(np.float32)


def _boids_separation(
    diff: np.ndarray,
    dist_sq: np.ndarray,
    per_mask: np.ndarray,
    sep_r_sq: float,
    max_speed: float,
    max_force: float,
    vel: np.ndarray,
) -> np.ndarray:
    sep_m = per_mask & (dist_sq > np.float32(1e-12)) & (dist_sq < np.float32(sep_r_sq))
    if not np.any(sep_m):
        return np.zeros(2, dtype=np.float32)
    ds = dist_sq[sep_m][:, np.newaxis].astype(np.float64) + 1e-6
    contrib = diff[sep_m].astype(np.float64) / ds
    steer = np.mean(contrib, axis=0).astype(np.float32)
    sn = float(np.linalg.norm(steer))
    if sn <= 0:
        return np.zeros(2, dtype=np.float32)
    steer = (steer / np.float32(sn) * np.float32(max_speed) - vel).astype(np.float32)
    sn2 = float(np.linalg.norm(steer))
    if sn2 > max_force:
        steer = (steer / np.float32(sn2) * np.float32(max_force)).astype(np.float32)
    return steer


def _boids_alignment(
    velocities: np.ndarray,
    per_mask: np.ndarray,
    max_speed: float,
    max_force: float,
    vel_i: np.ndarray,
) -> np.ndarray:
    if not np.any(per_mask):
        return np.zeros(2, dtype=np.float32)
    avg_v = np.mean(velocities[per_mask].astype(np.float64), axis=0).astype(np.float32)
    an = float(np.linalg.norm(avg_v))
    if an <= 0:
        return np.zeros(2, dtype=np.float32)
    desired = (avg_v / np.float32(an) * np.float32(max_speed)).astype(np.float32)
    steer = desired - vel_i
    sn = float(np.linalg.norm(steer))
    if sn > max_force:
        steer = (steer / np.float32(sn) * np.float32(max_force)).astype(np.float32)
    return steer


def _boids_cohesion(
    positions: np.ndarray,
    per_mask: np.ndarray,
    pos_i: np.ndarray,
    max_speed: float,
    max_force: float,
    vel_i: np.ndarray,
) -> np.ndarray:
    if not np.any(per_mask):
        return np.zeros(2, dtype=np.float32)
    center = np.mean(positions[per_mask].astype(np.float64), axis=0).astype(np.float32)
    return _np_seek_to_position(pos_i, center, max_speed, max_force, vel_i)


def _boids_evasion(
    pos: np.ndarray,
    herders: np.ndarray,
    evasion_radius: float,
    max_speed: float,
    max_force: float,
    vel_i: np.ndarray,
) -> np.ndarray:
    ev_sq = float(evasion_radius * evasion_radius)
    acc = np.zeros(2, dtype=np.float64)
    count = 0
    for hi in range(herders.shape[0]):
        diff = pos.astype(np.float64) - herders[hi].astype(np.float64)
        dsq = float(diff[0] * diff[0] + diff[1] * diff[1])
        if dsq < ev_sq and dsq > 0:
            acc = acc + diff / (dsq + 1e-6)
            count += 1
    if count == 0:
        return np.zeros(2, dtype=np.float32)
    steer = (acc / float(count)).astype(np.float32)
    sn = float(np.linalg.norm(steer))
    if sn <= 0:
        return np.zeros(2, dtype=np.float32)
    steer = (steer / np.float32(sn) * np.float32(max_speed * 1.5) - vel_i).astype(np.float32)
    sn2 = float(np.linalg.norm(steer))
    mf2 = max_force * 2.0
    if sn2 > mf2:
        steer = (steer / np.float32(sn2) * np.float32(mf2)).astype(np.float32)
    return steer


def world_radius_from_size(world_size: Tuple[float, float]) -> float:
    """圆形场地：半径 = min(W,H)/2，目标在圆心 (0,0)。"""
    return float(min(float(world_size[0]), float(world_size[1])) / 2.0)


def clip_position_to_disk(
    position: np.ndarray, world_radius: float, epsilon: float = 1e-6
) -> np.ndarray:
    p = np.asarray(position, dtype=np.float32).reshape(2)
    r = float(np.linalg.norm(p))
    lim = float(world_radius)
    if r > lim and r > epsilon:
        p = (p * (np.float32(lim) / np.float32(r))).astype(np.float32)
    return p


def assign_herders_to_slots(
    herder_positions: np.ndarray,
    slots: np.ndarray,
    assignment_mode: str,
) -> np.ndarray:
    """
    将弧上槽位分配给各牧者。

    Args:
        herder_positions: (N, 2)
        slots: (N, 2) 编队槽位，顺序为弧上索引
        assignment_mode: "ordered" 或 "min_cost"

    Returns:
        (N, 2)，第 i 行为牧者 i 应前往的目标（已分配）
    """
    slots = np.asarray(slots, dtype=np.float32)
    n = int(slots.shape[0])
    if n <= 0:
        return slots.copy()
    if str(assignment_mode) != "min_cost":
        return slots.copy()
    H = np.asarray(herder_positions, dtype=np.float64).reshape(n, 2)
    S = np.asarray(slots, dtype=np.float64).reshape(n, 2)
    diff = H[:, np.newaxis, :] - S[np.newaxis, :, :]
    cost = np.linalg.norm(diff, axis=2)
    row_ind, col_ind = linear_sum_assignment(cost)
    out = np.zeros_like(S)
    for k in range(n):
        out[int(row_ind[k])] = S[int(col_ind[k])]
    return out.astype(np.float32)


def _boids_boundary_disk_batch(
    positions: np.ndarray,
    world_radius: float,
    margin: float,
    max_force: float,
) -> np.ndarray:
    """越出圆盘 r > R−margin 时施加指向圆心的边界力。"""
    R = float(world_radius)
    m = float(margin)
    mf = np.float32(max_force)
    lim = max(R - m, 1e-3)
    xy = positions.astype(np.float32)
    r = np.sqrt(xy[:, 0] * xy[:, 0] + xy[:, 1] * xy[:, 1]).astype(np.float32)
    out = np.zeros_like(positions, dtype=np.float32)
    mask = r > np.float32(lim)
    if not np.any(mask):
        return out
    invr = np.where(r > np.float32(1e-6), np.float32(1.0) / r, np.float32(0.0))
    ux = (-xy[:, 0] * invr).astype(np.float32)
    uy = (-xy[:, 1] * invr).astype(np.float32)
    out[mask, 0] = ux[mask] * mf
    out[mask, 1] = uy[mask] * mf
    return out


def sample_random_target_position(
    world_size: Tuple[float, float],
    margin: float = 1.0,
    rng: Optional[Any] = None,
) -> np.ndarray:
    """
    圆形场地、目标固定在圆心 (0,0)；保留签名以兼容调用方，忽略 world_size/margin。
    """
    return np.zeros(2, dtype=np.float32)


class SheepScenario:
    """
    场景管理类（圆形场地，目标在圆心 (0,0)）

    - world_size 的较小边为直径，场地半径 R = min(W,H)/2，合法位置满足 |p| ≤ R
    - 目标固定为原点；观测中位置量以极坐标 (ρ/r_max, θ/π) 等形式给出
    """

    def __init__(
        self,
        world_size: Tuple[float, float] = DEFAULT_WORLD_SIZE,
        num_sheep: int = DEFAULT_NUM_SHEEP,
        num_herders: int = DEFAULT_NUM_HERDERS,
        target_position: Optional[np.ndarray] = None,
        sheep_config: Optional[Dict[str, Any]] = None,
        random_seed: Optional[int] = None,
        use_herder_kinematics: bool = True,
        herder_motion: Optional[Dict[str, Any]] = None,
    ):
        """
        Args:
            world_size: (W, H)，取 min 为直径定义圆形场地
            num_sheep: 羊的数量
            num_herders: 机械狗数量
            target_position: 忽略，目标恒为 (0,0)
            sheep_config: 羊的配置参数
            random_seed: 随机种子
            use_herder_kinematics: False 时机械狗每步直接置于编队目标点（无运动学），便于调试高层队形
            herder_motion: 覆盖默认机械狗运动/分配参数（见 envs.defaults.default_herder_motion_config）
        """
        if random_seed is not None:
            np.random.seed(random_seed)
        
        self.use_herder_kinematics = use_herder_kinematics
        self.world_size = world_size
        self.world_radius = world_radius_from_size(world_size)
        self.num_sheep = num_sheep
        self.num_herders = num_herders

        self._hm: Dict[str, Any] = default_herder_motion_config()
        if herder_motion:
            self._hm.update(copy.deepcopy(herder_motion))
        
        self.sheep_config = sheep_config or default_sheep_config()
        
        self.sheep: List[SheepEntity] = []
        self.herder_positions: np.ndarray = np.zeros((num_herders, 2), dtype=np.float32)
        self.target_position = np.zeros(2, dtype=np.float32)
        
        self.boids_weights = default_boids_weights()
        # 向量化 Boids 与 SheepEntity 行为对齐；False 时回退逐羊 Python 循环（调试用）
        self.vectorized_sheep_updates = True
        
        self._init_scenario()
    
    def _init_scenario(self):
        """初始化场景元素"""
        self._init_sheep()
        self._init_herders()
        self._init_target()
    
    def _init_sheep(self):
        """羊群以「一团」初始化：随机群体中心 + 中心附近小圆盘内均匀撒点。"""
        self.sheep = []
        R = float(self.world_radius)
        margin = 1.0
        lim = R - margin
        n = int(self.num_sheep)

        cluster_r = float(
            min(2.5 + 0.55 * np.sqrt(max(n, 1)), 0.22 * R)
        )
        cluster_r = max(cluster_r, min(1.0, 0.06 * R))
        max_center_r = max(lim - cluster_r - 0.5, 0.12 * R)
        min_center_r = min(0.26 * R, max_center_r * 0.55)
        if max_center_r <= min_center_r + 1e-3:
            min_center_r = max(0.1 * R, max_center_r * 0.3)
            max_center_r = max(min_center_r + 0.5, lim - cluster_r)

        ca = float(np.random.uniform(0.0, 2.0 * np.pi))
        cr = float(np.random.uniform(min_center_r, max(max_center_r, min_center_r + 1e-3)))
        flock_center = np.array(
            [np.cos(ca) * cr, np.sin(ca) * cr], dtype=np.float32
        )
        flock_center = clip_position_to_disk(flock_center, max(lim - cluster_r, 0.1 * R))

        for _ in range(n):
            ang = float(np.random.uniform(0.0, 2.0 * np.pi))
            rad = float(np.sqrt(np.random.uniform(0.0, 1.0)) * cluster_r)
            pos = flock_center + np.array(
                [np.cos(ang) * rad, np.sin(ang) * rad], dtype=np.float32
            )
            pos = clip_position_to_disk(pos, lim)

            sheep = SheepEntity(
                position=pos,
                max_speed=self.sheep_config['max_speed'],
                max_force=self.sheep_config['max_force'],
                perception_radius=self.sheep_config['perception_radius'],
                separation_radius=self.sheep_config['separation_radius'],
                velocity_drag=self.sheep_config.get('velocity_drag', 0.55),
                evasion_radius=self.sheep_config.get('evasion_radius', 8.0),
            )
            self.sheep.append(sheep)
    
    def _init_herders(self):
        """机械狗初始位置：random_disk 或 fixed_arc（π 侧弧，旧默认）。"""
        n = self.num_herders
        self.herder_positions = np.zeros((n, 2), dtype=np.float32)
        R = float(self.world_radius)
        mode = str(self._hm.get("herder_init_mode", "random_disk"))
        if mode == "fixed_arc":
            base_r = 0.62 * R
            for i in range(n):
                ang = float(np.pi + (i - (n - 1) / 2.0) * 0.4)
                p = np.array(
                    [np.cos(ang) * base_r, np.sin(ang) * base_r], dtype=np.float32
                )
                self.herder_positions[i] = clip_position_to_disk(p, R - 0.5)
            return
        margin = float(self._hm.get("herder_init_margin", 0.5))
        r_min_frac = float(self._hm.get("herder_init_r_min_frac", 0.12))
        r_min = float(np.clip(r_min_frac * R, 0.05, R * 0.9))
        r_max = float(max(R - margin, r_min + 1e-2))
        for i in range(n):
            ang = float(np.random.uniform(0.0, 2.0 * np.pi))
            u = float(np.random.uniform(0.0, 1.0))
            r_sq = u * (r_max * r_max - r_min * r_min) + r_min * r_min
            r = float(np.sqrt(max(r_sq, 0.0)))
            p = np.array([np.cos(ang) * r, np.sin(ang) * r], dtype=np.float32)
            self.herder_positions[i] = clip_position_to_disk(p, R)
    
    def _init_target(self):
        """目标固定在圆心。"""
        self.target_position = np.zeros(2, dtype=np.float32)
    
    def reset(
        self,
        target_position: Optional[np.ndarray] = None,
        random_seed: Optional[int] = None,
    ):
        """
        重置场景
        
        Args:
            target_position: 忽略（目标恒为圆心）
            random_seed: 新的随机种子
        """
        if random_seed is not None:
            np.random.seed(random_seed)
        
        self.target_position = np.zeros(2, dtype=np.float32)
        
        self._init_sheep()
        self._init_herders()
    
    def set_herder_targets(self, targets: np.ndarray):
        """
        设置机械狗的目标位置（先对槽位做最优分配，再写入 herder_targets[i]）。

        Args:
            targets: 编队槽位 (num_herders, 2)，顺序为弧上槽位索引
        """
        slots = np.asarray(targets, dtype=np.float32)
        n = self.num_herders
        if slots.shape[0] != n:
            raise ValueError(
                f"targets 行数 {slots.shape[0]} 与 num_herders={n} 不一致"
            )
        mode = str(self._hm.get("herder_slot_assignment", "min_cost"))
        self.herder_targets = assign_herders_to_slots(
            self.herder_positions, slots, mode
        ).copy()
    
    def update_herders(self, dt: float = 2.0):
        """
        更新机械狗位置：目标吸引 + 可选牧者间斥力 + 羊群质心斥力，再按最大速度积分。
        
        Args:
            dt: 时间步长
        """
        if not (hasattr(self, 'herder_targets') and self.herder_targets is not None):
            return
        
        if not self.use_herder_kinematics:
            for i in range(self.num_herders):
                self.herder_positions[i] = clip_position_to_disk(
                    self.herder_targets[i], self.world_radius
                ).astype(np.float32)
            return
        
        hm = self._hm
        k_attract = float(hm.get("k_attract", 1.0))
        f_spread = float(hm.get("flock_repel_spread_scale", 2.5))
        f_off = float(hm.get("flock_repel_offset", 3.0))
        f_gain = float(hm.get("flock_repel_gain", 1.5))
        peer_r = float(hm.get("herder_peer_repel_radius", 4.0))
        peer_g = float(hm.get("herder_peer_repel_gain", 1.0))
        v_max = float(hm.get("herder_max_speed", 5.0))
        reach_eps = float(hm.get("herder_target_reach_eps", 0.1))
        move_min = float(hm.get("herder_move_min_norm", 0.1))

        if self.sheep:
            pos_arr = np.asarray([s.position for s in self.sheep], dtype=np.float32)
            flock_center = np.mean(pos_arr, axis=0).astype(np.float32)
            if pos_arr.shape[0] < 2:
                flock_spread = 0.0
            else:
                flock_spread = float(
                    np.std(np.linalg.norm(pos_arr - flock_center, axis=1))
                )
        else:
            flock_center = np.zeros(2, dtype=np.float32)
            flock_spread = 0.0
        avoid_radius = float(flock_spread * f_spread + f_off)

        H = self.herder_positions

        for i in range(self.num_herders):
            current_pos = self.herder_positions[i]
            target_pos = self.herder_targets[i]

            to_target = target_pos - current_pos
            target_dist = float(np.linalg.norm(to_target))

            if target_dist < reach_eps:
                self.herder_positions[i] = clip_position_to_disk(
                    self.herder_positions[i], self.world_radius
                ).astype(np.float32)
                continue

            attract_force = (to_target / np.float32(target_dist)) * np.float32(k_attract)

            to_flock = current_pos - flock_center
            flock_dist = float(np.linalg.norm(to_flock))

            repel_force = np.zeros(2, dtype=np.float32)
            if flock_dist < avoid_radius and flock_dist > 0.1:
                repel_strength = (avoid_radius - flock_dist) / max(avoid_radius, 1e-6)
                repel_force = (
                    (to_flock / np.float32(flock_dist))
                    * np.float32(repel_strength * f_gain)
                ).astype(np.float32)

            peer_force = np.zeros(2, dtype=np.float32)
            if peer_g != 0.0 and peer_r > 0.0:
                for j in range(self.num_herders):
                    if j == i:
                        continue
                    diff_ij = current_pos - H[j]
                    dij = float(np.linalg.norm(diff_ij))
                    if dij < peer_r and dij > 1e-6:
                        w = (1.0 - dij / peer_r) ** 2
                        peer_force += (
                            (diff_ij / np.float32(dij)) * np.float32(peer_g * w)
                        ).astype(np.float32)

            move_direction = attract_force + repel_force + peer_force
            move_norm = float(np.linalg.norm(move_direction))

            if move_norm > move_min:
                move_direction = (move_direction / np.float32(move_norm)).astype(
                    np.float32
                )
                step = min(v_max * float(dt), target_dist)
                self.herder_positions[i] = (
                    self.herder_positions[i] + move_direction * np.float32(step)
                ).astype(np.float32)

            self.herder_positions[i] = clip_position_to_disk(
                self.herder_positions[i], self.world_radius
            ).astype(np.float32)
    
    def set_herder_positions(self, positions: np.ndarray):
        """
        直接设置机械狗位置
        
        Args:
            positions: 新的机械狗位置数组，形状为 (num_herders, 2)
        """
        positions = np.array(positions, dtype=np.float32)
        
        for i in range(min(len(positions), self.num_herders)):
            self.herder_positions[i] = clip_position_to_disk(
                positions[i], self.world_radius
            ).astype(np.float32)
    
    def update_sheep(self, dt: float = 2.0):
        """
        更新羊群状态
        
        应用Boids规则并更新每只羊的位置
        """
        if self.vectorized_sheep_updates:
            self._update_sheep_vectorized(dt)
        else:
            self._update_sheep_legacy(dt)

    def _update_sheep_legacy(self, dt: float = 2.0):
        herder_list = [self.herder_positions[i] for i in range(self.num_herders)]
        for sheep in self.sheep:
            sheep.apply_boids_rules(
                all_sheep=self.sheep,
                herders=herder_list,
                world_radius=self.world_radius,
                weights=self.boids_weights,
            )
            sheep.update(dt)
            sheep.position[:] = clip_position_to_disk(
                sheep.position, self.world_radius
            )

    def _update_sheep_vectorized(self, dt: float = 2.0):
        """
        与 _update_sheep_legacy 相同的更新顺序：按列表顺序每只羊先算力再积分，
        后续羊看到的是前面羊已更新后的位置/速度（与逐实体循环语义一致）。
        """
        n = len(self.sheep)
        if n == 0:
            return
        sh0 = self.sheep[0]
        per_r2 = np.float32(sh0.perception_radius * sh0.perception_radius)
        sep_r2 = float(sh0.separation_radius * sh0.separation_radius)
        max_speed = float(sh0.max_speed)
        max_force = float(sh0.max_force)
        evasion_r = float(getattr(sh0, 'evasion_radius', 8.0))
        H = self.herder_positions.astype(np.float32)
        w = self.boids_weights

        for i in range(n):
            P = np.stack([s.position for s in self.sheep]).astype(np.float32)
            V = np.stack([s.velocity for s in self.sheep]).astype(np.float32)
            bdf = _boids_boundary_disk_batch(P, self.world_radius, 2.0, max_force)[i]

            diff = (P[i] - P).astype(np.float32)
            dist_sq = np.sum(diff * diff, axis=1).astype(np.float32)
            per = (dist_sq < per_r2) & (dist_sq > np.float32(1e-12))
            per = per.copy()
            per[i] = False

            sep_f = _boids_separation(diff, dist_sq, per, sep_r2, max_speed, max_force, V[i])
            ali_f = _boids_alignment(V, per, max_speed, max_force, V[i])
            coh_f = _boids_cohesion(P, per, P[i], max_speed, max_force, V[i])
            eva_f = _boids_evasion(P[i], H, evasion_r, max_speed, max_force, V[i])

            total = (
                sep_f * np.float32(w.get('separation', 1.0))
                + ali_f * np.float32(w.get('alignment', 1.0))
                + coh_f * np.float32(w.get('cohesion', 1.0))
                + eva_f * np.float32(w.get('evasion', 1.0))
                + bdf * np.float32(w.get('boundary', 1.0))
            )

            sheep = self.sheep[i]
            sheep.acceleration[:] = 0.0
            sheep.apply_force(total)
            sheep.update(dt)
            sheep.position[:] = clip_position_to_disk(
                sheep.position, self.world_radius
            ).astype(np.float32)
    
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
        
        r_max = float(self.world_radius)
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

    def _append_herder_centroid_polar_flock(
        self,
        obs: np.ndarray,
        r_max: float,
    ) -> np.ndarray:
        """拼接机械狗质心相对羊群质心的极坐标：ρ/r_max、θ/π（与 SheepFlockEnv 主观测一致）。"""
        base = 8
        rm = np.float32(max(r_max, 1e-6))
        if self.num_herders <= 0:
            obs[base] = np.float32(0.0)
            obs[base + 1] = np.float32(0.0)
            return obs
        fc = self.get_flock_center().astype(np.float32)
        hc = np.mean(self.herder_positions.astype(np.float32), axis=0)
        rel = hc - fc
        rn = float(np.linalg.norm(rel))
        obs[base] = np.clip(
            np.float32(rn / rm), np.float32(-10.0), np.float32(10.0)
        )
        if rn > 1e-6:
            ang = float(np.arctan2(float(rel[1]), float(rel[0])) / np.pi)
        else:
            ang = 0.0
        obs[base + 1] = np.clip(np.float32(ang), np.float32(-1.0), np.float32(1.0))
        return obs

    def _observation_snapshot(self) -> Tuple[np.ndarray, np.ndarray, float]:
        """10 维 obs：8 维羊群 + 2 维狗质心相对羊质心极坐标；返回 (obs, flock_center, r_max_shared)。"""
        dim = 10
        obs = np.zeros(dim, dtype=np.float32)
        r_shared = float(max(self.world_radius, 1e-6))
        target = self.target_position.astype(np.float32)
        r_max = r_shared

        if not self.sheep:
            obs = self._append_herder_centroid_polar_flock(obs, r_max)
            obs = np.nan_to_num(obs, nan=0.0, posinf=10.0, neginf=-10.0).astype(np.float32)
            return obs, np.zeros(2, dtype=np.float32), r_shared

        positions = np.asarray([s.position for s in self.sheep], dtype=np.float32)
        velocities = np.asarray([s.velocity for s in self.sheep], dtype=np.float32)
        flock_center = np.mean(positions, axis=0).astype(np.float32)

        max_sheep_speed = float(self.sheep_config.get("max_speed", 3.0))
        if max_sheep_speed <= 0:
            max_sheep_speed = 1.0

        to_target = target - flock_center
        d_fc = float(np.linalg.norm(to_target))
        obs[0] = np.clip(
            np.float32(d_fc / r_max), np.float32(-10.0), np.float32(10.0)
        )
        if d_fc > 1e-6:
            obs[1] = np.clip(
                np.float32(
                    np.arctan2(float(to_target[1]), float(to_target[0])) / np.pi
                ),
                np.float32(-1.0),
                np.float32(1.0),
            )
        else:
            obs[1] = np.float32(0.0)

        mean_velocity = np.mean(velocities, axis=0).astype(np.float32)
        v_sp = float(np.linalg.norm(mean_velocity))
        obs[2] = np.clip(
            np.float32(v_sp / max_sheep_speed), np.float32(-1.0), np.float32(1.0)
        )
        if v_sp > 1e-6:
            obs[3] = np.clip(
                np.float32(
                    np.arctan2(float(mean_velocity[1]), float(mean_velocity[0]))
                    / np.pi
                ),
                np.float32(-1.0),
                np.float32(1.0),
            )
        else:
            obs[3] = np.float32(0.0)

        # [4:8] 与奖励包络一致：轴对齐四向极值 e0..e3（相对目标），各除以 r_max
        rel = positions.astype(np.float64) - target.astype(np.float64).reshape(1, 2)
        e0 = float(np.max(rel[:, 0]))
        e1 = float(np.max(rel[:, 1]))
        e2 = float(np.max(-rel[:, 0]))
        e3 = float(np.max(-rel[:, 1]))
        obs[4] = np.clip(np.float32(e0 / r_max), np.float32(-10.0), np.float32(10.0))
        obs[5] = np.clip(np.float32(e1 / r_max), np.float32(-10.0), np.float32(10.0))
        obs[6] = np.clip(np.float32(e2 / r_max), np.float32(-10.0), np.float32(10.0))
        obs[7] = np.clip(np.float32(e3 / r_max), np.float32(-10.0), np.float32(10.0))

        obs = self._append_herder_centroid_polar_flock(obs, r_max)
        obs = np.nan_to_num(obs, nan=0.0, posinf=10.0, neginf=-10.0).astype(np.float32)
        return obs, flock_center, r_shared
    
    def get_observation(self) -> np.ndarray:
        """
        10 维（与 N 无关）：
        - [0]: 质心到目标距离 / r_max；[1]: 质心相对目标的方位角 / π
        - [2]: 平均速度模 / max_speed；[3]: 平均速度方位角 / π
        - [4:8]: 羊群相对目标轴对齐四向极值 e0..e3 / r_max（与 r_envelope 包络一致）
        - [8]: 狗质心相对羊质心的极径 / r_max；[9]: 该相对位移方位角 / π
        """
        return self._observation_snapshot()[0]
    
    def get_shared_observation(self) -> np.ndarray:
        """
        Get shared observation vector (for centralized Critic)
        
        在完整 get_observation() 后再拼接各狗相对羊群质心的极坐标 (ρ/r_shared, θ/π)。
        Shape: (10 + 2 * num_herders)，默认 N=3 时为 16
        """
        obs, flock_center, r_shared = self._observation_snapshot()
        parts = [obs]
        rs = np.float32(max(r_shared, 1e-6))
        for i in range(self.num_herders):
            rel = (self.herder_positions[i] - flock_center).astype(np.float32)
            rn = float(np.linalg.norm(rel))
            pr = np.clip(np.float32(rn / rs), np.float32(-10.0), np.float32(10.0))
            if rn > 1e-6:
                pa = np.clip(
                    np.float32(np.arctan2(float(rel[1]), float(rel[0])) / np.pi),
                    np.float32(-1.0),
                    np.float32(1.0),
                )
            else:
                pa = np.float32(0.0)
            parts.append(np.array([pr, pa], dtype=np.float32))
        return np.concatenate(parts).astype(np.float32)
    
    def set_boids_weights(self, weights: Dict[str, float]):
        """设置Boids规则权重"""
        self.boids_weights.update(weights)
    
    def __repr__(self):
        return (
            f"SheepScenario("
            f"world_size={self.world_size}, "
            f"world_radius={self.world_radius:.3g}, "
            f"num_sheep={self.num_sheep}, "
            f"num_herders={self.num_herders}, "
            f"use_herder_kinematics={self.use_herder_kinematics}, "
            f"herder_init_mode={self._hm.get('herder_init_mode')}, "
            f"herder_slot_assignment={self._hm.get('herder_slot_assignment')}, "
            f"target={self.target_position})"
        )