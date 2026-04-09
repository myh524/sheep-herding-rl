#!/usr/bin/env python3
"""
随机生成羊群位置，可视化「四向轴对齐径向包络」的四个极值点与对应 AABB。

与观测中 e0..e3 定义一致（相对目标 target）：
  e0 = max(rel_x), e1 = max(rel_y), e2 = max(-rel_x), e3 = max(-rel_y)
  AABB 面积 = (e0+e2)(e1+e3)

用法（仓库根目录）:
  python scripts/visualize_flock_envelope.py
  python scripts/visualize_flock_envelope.py --seed 7 --num_sheep 15 --radius 50
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def sample_sheep_in_disk(
    n: int, radius: float, rng: np.random.Generator
) -> np.ndarray:
    """圆盘内近似均匀分布 (N, 2)。"""
    ang = rng.uniform(0.0, 2.0 * np.pi, size=n)
    rad = np.sqrt(rng.uniform(0.0, 1.0, size=n)) * float(radius) * 0.92
    x = rad * np.cos(ang)
    y = rad * np.sin(ang)
    return np.stack([x, y], axis=1).astype(np.float64)


def axis_envelope_indices(
    positions: np.ndarray, target: np.ndarray
) -> tuple[np.ndarray, tuple[int, int, int, int], np.ndarray]:
    """
    Returns:
        rel: (N, 2)
        (i0, i1, i2, i3): 分别达到 e0,e1,e2,e3 的羊索引（可重复）
        env: (e0, e1, e2, e3)
    """
    rel = positions.astype(np.float64) - target.astype(np.float64).reshape(1, 2)
    i0 = int(np.argmax(rel[:, 0]))
    i1 = int(np.argmax(rel[:, 1]))
    i2 = int(np.argmin(rel[:, 0]))
    i3 = int(np.argmin(rel[:, 1]))
    e0 = float(rel[i0, 0])
    e1 = float(rel[i1, 1])
    e2 = float(-rel[i2, 0])
    e3 = float(-rel[i3, 1])
    env = np.array([e0, e1, e2, e3], dtype=np.float64)
    return rel, (i0, i1, i2, i3), env


def main() -> None:
    p = argparse.ArgumentParser(description="可视化羊群四向包络极值点")
    p.add_argument("--seed", type=int, default=None, help="随机种子，默认每次不同")
    p.add_argument("--num_sheep", type=int, default=14)
    p.add_argument("--radius", type=float, default=50.0, help="采样圆盘半径（100×100 场地时 R=50）")
    p.add_argument("--target_x", type=float, default=0.0)
    p.add_argument("--target_y", type=float, default=0.0)
    p.add_argument("-o", "--output", type=str, default=None, help="保存 png 路径，不设则弹窗")
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    target = np.array([args.target_x, args.target_y], dtype=np.float64)
    positions = sample_sheep_in_disk(args.num_sheep, args.radius, rng)
    rel, (i0, i1, i2, i3), env = axis_envelope_indices(positions, target)
    e0, e1, e2, e3 = env
    area = (e0 + e2) * (e1 + e3)

    tx, ty = float(target[0]), float(target[1])
    xmin = tx - e2
    xmax = tx + e0
    ymin = ty - e3
    ymax = ty + e1

    fig, ax = plt.subplots(figsize=(9, 9))
    pad = max(args.radius * 0.12, 2.0)
    ax.set_xlim(tx - args.radius - pad, tx + args.radius + pad)
    ax.set_ylim(ty - args.radius - pad, ty + args.radius + pad)
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(
        f"四向包络极值点  |  A=(e0+e2)(e1+e3)={area:.1f}  |  "
        f"e0={e0:.2f}, e1={e1:.2f}, e2={e2:.2f}, e3={e3:.2f}"
    )
    ax.grid(True, alpha=0.3)

    arena = Circle((tx, ty), args.radius, fill=False, edgecolor="0.5", linestyle="--", linewidth=1.2)
    ax.add_patch(arena)
    ax.plot(tx, ty, "g*", markersize=22, label="目标", zorder=6)

    ax.scatter(positions[:, 0], positions[:, 1], c="0.45", s=55, alpha=0.75, label="羊", zorder=3)

    labels = ("e0: max +x", "e1: max +y", "e2: max -x (左)", "e3: max -y (下)")
    indices = (i0, i1, i2, i3)
    colors = ("#d62728", "#1f77b4", "#ff7f0e", "#9467bd")
    for lab, idx, c in zip(labels, indices, colors):
        pt = positions[idx]
        ax.scatter(pt[0], pt[1], s=180, facecolors="none", edgecolors=c, linewidths=2.5, zorder=5)
        ax.annotate(
            lab,
            (pt[0], pt[1]),
            textcoords="offset points",
            xytext=(8, 8),
            fontsize=10,
            color=c,
            fontweight="bold",
        )

    rect = Rectangle(
        (xmin, ymin),
        xmax - xmin,
        ymax - ymin,
        fill=False,
        edgecolor="crimson",
        linewidth=2.0,
        linestyle="-",
        label="AABB (e0+e2)×(e1+e3)",
        zorder=4,
    )
    ax.add_patch(rect)
    ax.legend(loc="upper right")
    fig.tight_layout()

    if args.output:
        fig.savefig(args.output, dpi=150, bbox_inches="tight")
        print(f"已保存: {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
