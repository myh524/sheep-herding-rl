"""
环境与训练/评估脚本共享的默认超参（单源维护）。

不同入口若历史上默认不同（如 episode_length），在此用不同常量显式区分，避免隐性分叉。
修改默认行为时只改本文件即可；具体用法见各引用处（SheepFlockEnv、train_ppo 等）。
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# 场地 / 规模（与 SheepScenario、SheepFlockEnv、CLI 默认一致）
# ---------------------------------------------------------------------------

# 矩形 (W, H)：圆形场地半径 R = min(W,H)/2，目标在圆心 (0,0)
DEFAULT_WORLD_SIZE: Tuple[float, float] = (100.0, 100.0)

# argparse `nargs=2` 用的浮点列表，与 DEFAULT_WORLD_SIZE 同步
DEFAULT_WORLD_SIZE_ARGV: List[float] = [
    float(DEFAULT_WORLD_SIZE[0]),
    float(DEFAULT_WORLD_SIZE[1]),
]

DEFAULT_NUM_SHEEP = 10  # 羊只数
DEFAULT_NUM_HERDERS = 3  # 机械狗数量
DEFAULT_DT = 2.0  # 环境单步时间（秒），与 SheepFlockEnv.dt 一致

# ---------------------------------------------------------------------------
# Episode 最大步数：按入口区分（训练/评估/可视化历史默认不同，勿混用）
# ---------------------------------------------------------------------------

DEFAULT_FLOCK_EPISODE_LENGTH = 100  # SheepFlockEnv / RandomizedSheepFlockEnv 基类默认
DEFAULT_TRAIN_EPISODE_LENGTH = 150  # train_ppo 默认
DEFAULT_EVAL_EPISODE_LENGTH = 100  # evaluate_policy 默认
DEFAULT_VIS_EPISODE_LENGTH = 150  # visualize 默认

# ---------------------------------------------------------------------------
# 增量编队模式（--formation_delta）：|a[i]|≤1 时单档最大变化量
# 与 train_ppo / visualize / evaluate_policy 及环境构造默认一致
# ---------------------------------------------------------------------------

FORMATION_DELTA_THETA_MAX_DEG = 22.5  # θ_in 增量上限（度）
FORMATION_DELTA_THETA_MAX_RAD = float(np.deg2rad(FORMATION_DELTA_THETA_MAX_DEG))  # 同上，弧度
FORMATION_DELTA_RADIUS_MAX = 3.0  # 编队弧半径 R 的增量上限（米）
FORMATION_DELTA_COVERAGE_MAX = 0.15  # coverage 的增量上限（无量纲）

# ---------------------------------------------------------------------------
# RandomizedSheepFlockEnv：每 episode 随机抽样的范围与固定羊参
# ---------------------------------------------------------------------------

DEFAULT_RANDOMIZED_NUM_SHEEP_RANGE: Tuple[int, int] = (5, 15)  # 闭区间 [low, high]
DEFAULT_RANDOMIZED_NUM_HERDERS_RANGE: Tuple[int, int] = (2, 4)
# 世界尺寸上下界 (W,H)，reset 时在矩形内均匀采样
DEFAULT_RANDOMIZED_WORLD_SIZE_RANGE: Tuple[Tuple[float, float], Tuple[float, float]] = (
    (100.0, 100.0),
    (100.0, 100.0),
)
DEFAULT_RANDOMIZED_SHEEP_SPEED_RANGE: Tuple[float, float] = (0.5, 1.5)  # max_speed 采样范围

# reset 里稀疏 sheep_config 的固定项（max_speed 由上面范围采样后传入）
RANDOMIZED_SHEEP_MAX_FORCE = 0.1
RANDOMIZED_SHEEP_PERCEPTION_RADIUS = 5.0
RANDOMIZED_SHEEP_SEPARATION_RADIUS = 2.0


def randomized_sheep_config(max_speed: float) -> Dict[str, float]:
    """
    RandomizedSheepFlockEnv.reset 中构造的 sheep_config（仅四键，与旧行为一致）。

    Args:
        max_speed: 本 episode 随机得到的羊最大速度。
    """
    return {
        "max_speed": float(max_speed),
        "max_force": RANDOMIZED_SHEEP_MAX_FORCE,
        "perception_radius": RANDOMIZED_SHEEP_PERCEPTION_RADIUS,
        "separation_radius": RANDOMIZED_SHEEP_SEPARATION_RADIUS,
    }


# ---------------------------------------------------------------------------
# 课程学习：供 CurriculumStage(**spec) 使用的纯 dict 列表
# 字段含义与 CurriculumStage 构造函数一致。
# 可选 num_sheep_range: (low, high) 闭区间，每 episode reset 均匀随机羊数（需 num_sheep≥high，作上界占位）。
# ---------------------------------------------------------------------------

DEFAULT_CURRICULUM_STAGE_SPECS: List[Dict[str, Any]] = [
    {
        "name": "Stage 0: Simple",  # 阶段名称（日志 / info）
        "num_sheep": 3,
        "num_herders": 3,
        "world_size": (70.0, 70.0),
        "episode_length": 200,
        "target_success_rate": 0.8,  # 成功率 ≥ 此值且满 min_episodes 可升阶
        "min_episodes": 100,  # 至少经历的本阶段 episode 数再检查升阶
    },
    {
        "name": "Stage 1: Medium",
        "num_sheep": 6,
        "num_herders": 3,
        "world_size": (100.0, 100.0),
        "episode_length": 200,
        "target_success_rate": 0.8,
        "min_episodes": 100,
    },
    {
        "name": "Stage 2: Target",
        "num_sheep": 10,
        "num_herders": 3,
        "world_size": (120.0, 120.0),
        "episode_length": 300,
        "target_success_rate": 0.8,
        "min_episodes": 200,
    },
    {
        "name": "Stage 3: Target1",
        "num_sheep": 15,
        "num_herders": 4,
        "world_size": (120.0, 120.0),
        "episode_length": 300,
        "target_success_rate": 0.8,
        "min_episodes": 200,
    },
    {
        "name": "Stage 4: Target2",
        "num_sheep": 20,
        "num_herders": 4,
        "world_size": (120.0, 120.0),
        "episode_length": 300,
        "target_success_rate": 0.8,
        "min_episodes": 200,
    },
    {
        "name": "Stage 5: Target3",
        "num_sheep": 25,
        "num_herders": 5,
        "world_size": (140.0, 140.0),
        "episode_length": 300,
        "target_success_rate": 0.8,
        "min_episodes": 200,
    },
    {
        "name": "Stage 6: Target4",
        "num_sheep": 30,
        "num_herders": 5,
        "world_size": (140.0, 140.0),
        "episode_length": 300,
        "target_success_rate": 0.8,
        "min_episodes": 200,
    },
    {
        "name": "Stage 7: Target5",
        "num_sheep": 30,
        "num_sheep_range": (5, 30),
        "num_herders": 5,
        "world_size": (140.0, 140.0),
        "episode_length": 300,
        "target_success_rate": 0.8,
        "min_episodes": 200,
    },
]

# SheepFlockEnv._compute_reward 稠密奖励系数（详见该函数 docstring）
_DEFAULT_REWARD_CONFIG: Dict[str, float] = {
    "w_potential": 3.0,  # 归一化距离势函数差分项权重
    "w_near": 0.35,  # 近目标指数奖励权重
    "near_d0_alpha": 0.15,  # 近场尺度 d0 = alpha * (2*world_radius)
    "w_speed": 0.40,  # 质心速度正则（近目标更强）
    "v_ref_scale": 1.0,  # 速度参考 v_ref = v_ref_scale * sheep max_speed
    "w_spread": 0.12,  # 羊群轴对齐包络面积（扩散）惩罚权重
    "envelope_area_ref": 0.12,  # 包络面积归一化参考，用于 tanh 尺度
    "time_penalty": 0.005,  # 每步时间惩罚系数（远离目标时）
    "time_penalty_off_threshold_m": 5.0,  # 质心距目标 < 此米数则时间惩罚为 0
    "reward_clip_low": -3.0,  # 总奖励下界
    "reward_clip_high": 5.0,  # 总奖励上界
}

# SheepScenario 默认羊动力学 / Boids 相关（未传 sheep_config 时使用）
_DEFAULT_SHEEP_CONFIG: Dict[str, Any] = {
    "max_speed": 2.0,  # 单羊速度上限
    "max_force": 0.3,  # 单步 steering 力上限
    "perception_radius": 30.0,  # 邻居感知半径
    "separation_radius": 3.0,  # 分离行为作用半径
    "velocity_drag": 0.55,  # 速度阻尼系数
    "evasion_radius": 5.0,  # 躲避机械狗半径
}

# 向量化 Boids 各力项相对权重（与 SheepEntity 行为对齐）
_DEFAULT_BOIDS_WEIGHTS: Dict[str, float] = {
    "separation": 1.0,
    "alignment": 0.3,
    "cohesion": 0.5,
    "evasion": 1.0,
    "boundary": 1.0,
}


def default_reward_config() -> Dict[str, float]:
    """返回奖励系数字典的深拷贝，避免运行中修改污染模板。"""
    return copy.deepcopy(_DEFAULT_REWARD_CONFIG)


def default_sheep_config() -> Dict[str, Any]:
    """返回默认 sheep_config 的深拷贝。"""
    return copy.deepcopy(_DEFAULT_SHEEP_CONFIG)


def default_boids_weights() -> Dict[str, float]:
    """返回默认 Boids 权重的深拷贝。"""
    return copy.deepcopy(_DEFAULT_BOIDS_WEIGHTS)


# ---------------------------------------------------------------------------
# 机械狗：初始化、槽位分配（匈牙利）、势场运动学（与 SheepScenario 一致）
# ---------------------------------------------------------------------------

_DEFAULT_HERDER_MOTION: Dict[str, Any] = {
    # random_disk：圆环内均匀面积采样；fixed_arc：π 侧弧排布（旧行为）
    "herder_init_mode": "random_disk",
    "herder_init_margin": 0.5,
    "herder_init_r_min_frac": 0.12,
    # min_cost：当前位置到弧槽总路程最小；ordered：下标 i 对应槽位 i
    "herder_slot_assignment": "min_cost",
    "k_attract": 1.0,
    "flock_repel_spread_scale": 6.0,
    "flock_repel_offset": 3.0,
    "flock_repel_gain": 4.0,
    "herder_peer_repel_radius": 4.0,
    "herder_peer_repel_gain": 1.0,
    # 每步位移上限 v_max*dt；原为 5.0，现为五分之一
    "herder_max_speed": 0.7,
    "herder_target_reach_eps": 0.1,
    "herder_move_min_norm": 0.1,
}

# 与旧版 SheepScenario 一致：固定初值、恒等分配、无牧者间斥力
HERDER_PHYSICS_LEGACY_OVERRIDES: Dict[str, Any] = {
    "herder_init_mode": "fixed_arc",
    "herder_slot_assignment": "ordered",
    "herder_peer_repel_gain": 0.0,
    "herder_max_speed": 5.0,
}


def default_herder_motion_config() -> Dict[str, Any]:
    """机械狗运动/分配参数字典的深拷贝。"""
    return copy.deepcopy(_DEFAULT_HERDER_MOTION)


def merged_herder_motion_config(
    overrides: Optional[Dict[str, Any]] = None,
    legacy: bool = False,
) -> Dict[str, Any]:
    """
    合并默认、可选 legacy 覆盖与用户覆盖（后者优先）。
    """
    cfg = default_herder_motion_config()
    if legacy:
        cfg.update(copy.deepcopy(HERDER_PHYSICS_LEGACY_OVERRIDES))
    if overrides:
        cfg.update(copy.deepcopy(overrides))
    return cfg
