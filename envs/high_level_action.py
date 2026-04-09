"""
High-Level Action Decoder：以羊群质心为圆心的圆弧站位

动作 5 维 ∈ [-1,1]；**仅前 3 维参与解码**，`a[3]、a[4]` 固定按常数 0 处理（与网络输出无关）。

- [0]：`θ_in = a[0]·π`（wrap）为 **占位圆弧中点 → 羊质心** 的方位角（世界系，指向圆心）。弧在圆上的角向中心为 **`θ_mid = θ_in − π`**（即 **羊质心 → 弧中点**）。
- [1]: 站位半径 `R ∈ [R_min, R_max]`（默认 5–20）
- [2]: 占位分散度 → coverage ∈ [0,1]，再映射为弧张角 Θ（与 N 相关）

几何：圆心严格为羊质心 `c_flock`，半径 `R`，在 `[θ_mid − Θ/2, θ_mid + Θ/2]` 上对称均匀取 N 点。不依赖狗质心。
"""

import numpy as np
from typing import Dict, Any, Tuple

from envs.sheep_scenario import world_radius_from_size

# 站位半径全局统一（课程 / 随机化 / 基类环境均使用同一区间）
STANCE_RADIUS_MIN = 5.0
STANCE_RADIUS_MAX = 20.0


class HighLevelAction:
    """将 [-1,1]^5 解码为羊质心圆弧站位（仅用 a[0]–a[2]）。"""

    def __init__(
        self,
        R_ref: float = 8.0,
        R_min: float = STANCE_RADIUS_MIN,
        R_max: float = STANCE_RADIUS_MAX,
        world_margin: float = 1.0,
        min_arc_step_rad: float = 0.28,
        full_circle_eps: float = 1e-3,
        anchor_rho_min: float = 5.0,
    ):
        self.R_ref = R_ref
        self.R_min = R_min
        self.R_max = R_max
        self.world_margin = world_margin
        self.min_arc_step_rad = float(min_arc_step_rad)
        self.full_circle_eps = float(full_circle_eps)
        self.anchor_rho_min = float(anchor_rho_min)

    def _theta_max_for_n(self, n: int) -> float:
        two_pi = 2.0 * np.pi
        if n < 2:
            return 0.0
        return float(two_pi * (n - 1) / n)

    def _theta_min_for_n(self, n: int) -> float:
        eps = self.full_circle_eps
        if n < 2:
            return 0.0
        theta_max = self._theta_max_for_n(n)
        theta_min = (n - 1) * self.min_arc_step_rad
        theta_min = float(min(theta_min, theta_max - eps))
        return float(max(0.0, theta_min))

    def _theta_span_from_coverage(self, coverage: float, n: int) -> float:
        cov = float(np.clip(coverage, 0.0, 1.0))
        if n < 2:
            return 0.0
        theta_min = self._theta_min_for_n(n)
        theta_max = self._theta_max_for_n(n)
        if theta_max <= theta_min + 1e-6:
            return float(theta_max)
        return float(theta_min + cov * (theta_max - theta_min))

    def decode_action(
        self,
        raw_action: np.ndarray,
        flock_center: np.ndarray,
        world_size: Tuple[float, float],
    ) -> Dict[str, Any]:
        """
        Args:
            raw_action: shape (5,) 推荐；`a[3]、a[4]` 忽略，按 0 解码
            flock_center: 羊质心 (2,)
            world_size: (W, H)
        """
        ra = np.asarray(raw_action, dtype=np.float32).reshape(-1)
        padded = np.zeros(5, dtype=np.float32)
        n_in = min(5, ra.size)
        if n_in > 0:
            padded[:n_in] = ra[:n_in]
        padded = np.clip(padded, -1.0, 1.0)
        padded[3] = 0.0
        padded[4] = 0.0
        raw_action = padded

        fc = np.asarray(flock_center, dtype=np.float32).reshape(2)

        world_R = world_radius_from_size(world_size)
        r_lim = max(world_R - float(self.world_margin), 1e-3)
        r_max = float(max(r_lim, 1.0))

        theta_in = float(raw_action[0]) * np.pi
        theta_in = float(np.arctan2(np.sin(theta_in), np.cos(theta_in)))
        theta_mid = float(np.arctan2(np.sin(theta_in - np.pi), np.cos(theta_in - np.pi)))

        radius = self.R_min + (raw_action[1] + 1.0) / 2.0 * (self.R_max - self.R_min)
        radius = float(np.clip(radius, self.R_min, self.R_max))

        coverage = float((raw_action[2] + 1.0) / 2.0)
        coverage = float(np.clip(coverage, 0.0, 1.0))

        return {
            "flock_center": fc.copy(),
            "radius": radius,
            "coverage": coverage,
            "theta_in_rad": theta_in,
            "theta_mid_rad": theta_mid,
            "r_max": r_max,
        }

    def decoded_from_formation_params(
        self,
        flock_center: np.ndarray,
        world_size: Tuple[float, float],
        theta_in_rad: float,
        radius: float,
        coverage: float,
    ) -> Dict[str, Any]:
        """
        由已积分的编队参数生成与 decode_action 同结构的字典（用于增量动作模式）。
        """
        fc = np.asarray(flock_center, dtype=np.float32).reshape(2)
        world_R = world_radius_from_size(world_size)
        r_lim = max(world_R - float(self.world_margin), 1e-3)
        r_max = float(max(r_lim, 1.0))

        theta_in = float(theta_in_rad)
        theta_in = float(np.arctan2(np.sin(theta_in), np.cos(theta_in)))
        theta_mid = float(np.arctan2(np.sin(theta_in - np.pi), np.cos(theta_in - np.pi)))

        radius = float(np.clip(radius, self.R_min, self.R_max))
        coverage = float(np.clip(coverage, 0.0, 1.0))

        return {
            "flock_center": fc.copy(),
            "radius": radius,
            "coverage": coverage,
            "theta_in_rad": theta_in,
            "theta_mid_rad": theta_mid,
            "r_max": r_max,
        }

    def effective_span_radians(
        self, coverage: float, num_herders: int
    ) -> Tuple[float, bool]:
        n = int(num_herders)
        eps = self.full_circle_eps
        if n < 2:
            return 0.0, False
        theta_max = self._theta_max_for_n(n)
        span = self._theta_span_from_coverage(coverage, n)
        at_max = span >= theta_max - eps
        return float(span), at_max

    def sample_herder_positions(
        self,
        num_herders: int,
        flock_center: np.ndarray,
        radius: float,
        coverage: float,
        theta_mid_rad: float,
    ) -> np.ndarray:
        """
        圆心为羊质心，在半径 R 的圆上于 θ_mid 附近对称取弧上 N 点。
        """
        n = int(num_herders)
        O = np.asarray(flock_center, dtype=np.float32).reshape(2)
        R = float(radius)
        cov = float(np.clip(coverage, 0.0, 1.0))
        theta_mid = float(theta_mid_rad)

        positions = np.zeros((n, 2), dtype=np.float32)
        if n <= 0:
            return positions
        if n == 1:
            positions[0, 0] = O[0] + np.float32(R * np.cos(theta_mid))
            positions[0, 1] = O[1] + np.float32(R * np.sin(theta_mid))
            return positions

        theta_span = self._theta_span_from_coverage(cov, n)
        for i in range(n):
            t = i / (n - 1)
            ang = theta_mid + (t - 0.5) * theta_span
            positions[i, 0] = O[0] + np.float32(R * np.cos(ang))
            positions[i, 1] = O[1] + np.float32(R * np.sin(ang))

        return positions

    def get_formation_mode(self, coverage: float) -> str:
        if coverage < 0.2:
            return "PUSH"
        elif coverage < 0.5:
            return "NARROW"
        elif coverage < 0.8:
            return "WIDE"
        else:
            return "SURROUND"


class FormationAnalyzer:
    """调试用人可读描述"""

    @staticmethod
    def describe_formation(decoded_action: Dict[str, Any], num_herders: int) -> str:
        coverage = decoded_action["coverage"]
        radius = decoded_action["radius"]
        fc = decoded_action["flock_center"]
        dec = HighLevelAction()
        mode = dec.get_formation_mode(coverage)
        span_rad, at_max = dec.effective_span_radians(coverage, num_herders)
        spread_deg = np.degrees(span_rad)
        theta_max_deg = np.degrees(dec._theta_max_for_n(num_herders))
        mx = f" Θ_max={theta_max_deg:.0f}°" if num_herders >= 2 else ""
        am = " [MAX]" if at_max else ""
        ti = float(np.degrees(decoded_action.get("theta_in_rad", 0.0)))
        tm = float(np.degrees(decoded_action.get("theta_mid_rad", 0.0)))
        return (
            f"Formation: {mode} (coverage={coverage:.2f}){am}{mx}\n"
            f"  - θ_in(弧中点→羊, a[0]·π): {ti:.0f}°  θ_mid(羊→弧中点): {tm:.0f}°\n"
            f"  - Θ: {spread_deg:.0f}°\n"
            f"  - Radius: {radius:.1f}\n"
            f"  - Flock center: ({fc[0]:.1f}, {fc[1]:.1f})\n"
            f"  - Herders: {num_herders}"
        )


class KappaScheduler:
    """Legacy：原 wedge_width 预热，现视为 coverage 预热"""

    def __init__(
        self,
        warmup_epochs: int = 100,
        kappa_init: float = 0.5,
        **kwargs,
    ):
        self.warmup_epochs = warmup_epochs
        self.coverage_init = kappa_init

    def get_wedge_width(self, raw_width: float, epoch: int) -> float:
        if epoch < self.warmup_epochs:
            return self.coverage_init
        return (raw_width + 1) / 2
