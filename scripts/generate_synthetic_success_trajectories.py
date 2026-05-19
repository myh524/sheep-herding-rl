#!/usr/bin/env python3
"""
生成「成功引导羊群到达目标」效果的合成轨迹 PNG（非环境仿真，仅作示意或配图）。
风格对齐 visualize.py 中保存轨迹图：矩形参考框、刻度读数缩放、虚线单羊、红色平滑质心。
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import FuncFormatter

_SCRIPTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)))
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)
from matplotlib_zh import setup_matplotlib_chinese, zh_font  # noqa: E402

setup_matplotlib_chinese()

VIZ_AXIS_LABEL_DISPLAY_HALF = 250.0


def smooth_xy_traj(xy: np.ndarray, half_window: int | None = None) -> np.ndarray:
    xy = np.asarray(xy, dtype=np.float64)
    t = int(xy.shape[0])
    if t < 3:
        return xy.copy()
    if half_window is None:
        hw = max(1, min(8, max(1, t // 8)))
    else:
        hw = max(1, int(half_window))
    hw = min(hw, t // 2)
    out = np.zeros_like(xy)
    for i in range(t):
        lo, hi = max(0, i - hw), min(t, i + hw + 1)
        out[i] = xy[lo:hi].mean(axis=0)
    return out


def axis_scales(world_wx: float, world_wy: float) -> tuple[float, float]:
    half = float(VIZ_AXIS_LABEL_DISPLAY_HALF)
    hx = max(world_wx / 2.0, 1e-9)
    hy = max(world_wy / 2.0, 1e-9)
    return (half / hx, half / hy)


def apply_axis_display_tick_labels(ax, sx: float, sy: float) -> None:
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _p, sx=sx: f"{v * sx:g}"))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _p, sy=sy: f"{v * sy:g}"))


def allocate_path(out_dir: str, base: str) -> str:
    path = os.path.join(out_dir, f"{base}.png")
    if not os.path.isfile(path):
        return path
    k = 1
    while True:
        path = os.path.join(out_dir, f"{base}_{k}.png")
        if not os.path.isfile(path):
            return path
        k += 1


def synthetic_success_trajectory(
    rng: np.random.Generator,
    num_steps: int,
    num_sheep: int,
    target: np.ndarray,
    *,
    variant: int,
) -> np.ndarray:
    """
    合成轨迹：羊群从一侧被引导到目标位置，但路径比较坎坷，有来回波动。
    模仿单层MAPPO算法偶然成功的情况。
    variant 换相位，得到多张不同图。
    """
    pos = np.zeros((num_sheep, 2), dtype=np.float64)
    
    # 初始位置：在右侧形成一个紧凑的群体
    start_x = 180.0 + variant * 10.0
    start_y = rng.uniform(-30.0, 30.0) + variant * 5.0
    
    for i in range(num_sheep):
        angle = rng.uniform(0.0, 2.0 * np.pi)
        radius = rng.uniform(15.0, 35.0)
        pos[i] = [
            start_x + np.cos(angle) * radius,
            start_y + np.sin(angle) * radius
        ]
    
    buf = [pos.copy()]
    tgt = np.asarray(target, dtype=np.float64).reshape(2)
    
    # 生成一些关键点，让路径有波折
    phase_shift = variant * 0.5
    
    for t in range(1, num_steps):
        com = pos.mean(axis=0)
        
        # 向目标移动的主要推动力，但有波动
        progress = t / num_steps
        
        # 添加周期性波动，让路径更坎坷
        wobble_x = np.sin(t * 0.08 + phase_shift) * 25.0 * (1.0 - progress * 0.7)
        wobble_y = np.cos(t * 0.12 + phase_shift * 1.3) * 20.0 * (1.0 - progress * 0.6)
        
        # 主方向向目标，但带有波动
        toward_target = (tgt - com) * (0.025 + 0.015 * (1.0 - progress))
        movement = toward_target + np.array([wobble_x, wobble_y]) * 0.008
        
        # 有时会稍微远离目标，增加波折感
        if (t % 40) < 15 and t > 20:
            # 短暂的远离
            away = (com - tgt) * 0.012
            movement += away
        
        # 凝聚力：保持羊群相对紧凑
        for i in range(num_sheep):
            pos[i] += (com - pos[i]) * 0.018
        
        # 群体整体移动
        pos += movement
        
        # 添加一些随机性，但保持队形
        pos += rng.normal(0, 1.8, pos.shape)
        
        # 控制扩散范围
        for i in range(num_sheep):
            v = pos[i] - com
            nv = float(np.hypot(v[0], v[1])) + 1e-3
            if nv > 32.0:
                pos[i] -= (v / nv) * 0.5
        
        buf.append(pos.copy())
    
    return np.stack(buf, axis=0)


def _square_limits_around_data(
    traj: np.ndarray,
    target: np.ndarray,
    world_wx: float,
    world_wy: float,
) -> tuple[tuple[float, float], tuple[float, float]]:
    hx, hy = world_wx / 2.0, world_wy / 2.0
    tx, ty = float(target[0]), float(target[1])
    tr = 5.0
    xs = np.concatenate(
        [
            traj[..., 0].ravel(),
            np.asarray([tx, tx - tr, tx + tr, -hx, hx, -hx, hx], dtype=np.float64),
        ]
    )
    ys = np.concatenate(
        [
            traj[..., 1].ravel(),
            np.asarray([ty, ty - tr, ty + tr, -hy, hy, hy, -hy], dtype=np.float64),
        ]
    )
    span = max(float(xs.max() - xs.min()), float(ys.max() - ys.min()), 1e-3)
    pad = span * 0.08 + 4.0
    cx = 0.5 * (float(xs.min() + xs.max()))
    cy = 0.5 * (float(ys.min() + ys.max()))
    half = 0.5 * span + pad
    return (cx - half, cx + half), (cy - half, cy + half)


def render_trajectory_png(
    path: str,
    traj: np.ndarray,
    target: np.ndarray,
    world_wx: float,
    world_wy: float,
) -> None:
    (x0, x1), (y0, y1) = _square_limits_around_data(traj, target, world_wx, world_wy)
    sx, sy = axis_scales(world_wx, world_wy)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect("equal")
    ax.set_facecolor("#f5f5f5")

    hx, hy = world_wx / 2.0, world_wy / 2.0
    ax.add_patch(
        patches.Rectangle(
            (-hx, -hy),
            world_wx,
            world_wy,
            fill=False,
            edgecolor="0.65",
            linewidth=1.0,
            linestyle="--",
        )
    )

    ax.add_patch(
        patches.Circle(
            (float(target[0]), float(target[1])),
            25.0,
            color="green",
            alpha=0.3,
            zorder=2,
        )
    )
    ax.scatter(
        float(target[0]),
        float(target[1]),
        c="green",
        s=200,
        marker="*",
        zorder=8,
        edgecolors="darkgreen",
        linewidths=0.8,
        label="目标",
    )

    n_sheep = traj.shape[1]
    _tab = plt.get_cmap("tab10")
    sheep_colors = [_tab(j) for j in range(10)]
    
    for i in range(n_sheep):
        color = sheep_colors[i % len(sheep_colors)]
        ax.plot(
            traj[:, i, 0],
            traj[:, i, 1],
            "--",
            color=color,
            lw=1.6,
            alpha=0.88,
            label=f"羊 {i}",
            zorder=3,
        )
        ax.scatter(
            traj[0, i, 0],
            traj[0, i, 1],
            color=color,
            s=42,
            marker="o",
            edgecolors="black",
            linewidths=0.8,
            zorder=6,
        )
        ax.scatter(
            traj[-1, i, 0],
            traj[-1, i, 1],
            color=color,
            s=48,
            marker="s",
            edgecolors="black",
            linewidths=0.8,
            zorder=7,
        )

    centroid = np.mean(traj, axis=1)
    centroid_s = smooth_xy_traj(centroid)
    ax.plot(
        centroid_s[:, 0],
        centroid_s[:, 1],
        "-",
        color="crimson",
        lw=2.4,
        alpha=0.92,
        solid_capstyle="round",
        solid_joinstyle="round",
        label="羊群质心（平滑）",
        zorder=5,
    )
    ax.scatter(
        centroid_s[0, 0],
        centroid_s[0, 1],
        c="crimson",
        s=58,
        marker="D",
        zorder=8,
        edgecolors="white",
        linewidths=0.9,
    )
    ax.scatter(
        centroid_s[-1, 0],
        centroid_s[-1, 1],
        c="crimson",
        s=62,
        marker="P",
        zorder=8,
        edgecolors="white",
        linewidths=0.9,
    )

    ax.grid(True, alpha=0.3)
    apply_axis_display_tick_labels(ax, sx, sy)
    ax.legend(loc="upper right", prop=zh_font(size=22), ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> int:
    p = argparse.ArgumentParser(description="合成成功引导、羊群到达目标的轨迹图")
    p.add_argument(
        "--out_dir",
        type=str,
        default="figures/sheep_trajectories",
        help="输出目录（默认 figures/sheep_trajectories）",
    )
    p.add_argument("--num_plots", type=int, default=4, help="生成 PNG 数量")
    p.add_argument("--num_sheep", type=int, default=5, help="羊只数")
    p.add_argument("--num_steps", type=int, default=150, help="轨迹长度（时间步）")
    p.add_argument("--world_wx", type=float, default=500.0)
    p.add_argument("--world_wy", type=float, default=500.0)
    p.add_argument("--seed", type=int, default=20260507, help="随机种子基值")
    args = p.parse_args()

    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    target = np.array([0.0, 0.0], dtype=np.float64)
    stem = "synthetic_success"

    for k in range(int(args.num_plots)):
        rng = np.random.default_rng(int(args.seed) + k * 9973)
        traj = synthetic_success_trajectory(
            rng,
            int(args.num_steps),
            int(args.num_sheep),
            target,
            variant=k,
        )
        base = f"sheep_trajectory_{stem}_ep{k + 1:03d}"
        path = allocate_path(out_dir, base)
        render_trajectory_png(
            path,
            traj,
            target,
            float(args.world_wx),
            float(args.world_wy),
        )
        print(path, flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())