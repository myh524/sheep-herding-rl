#!/usr/bin/env python3
"""
交互式队形可视化：坐标系固定在**羊质心**（原点），与训练解码一致。

- `a[0]·π` = `θ_in`（弧中点 → 羊质心）；几何用 `θ_mid = θ_in − π`（羊质心 → 弧中点）。
- `a[1]`、`a[2]`：R、coverage；`a[3]、a[4]` 解码固定为 0。
- 不依赖狗质心。橙圈上可看点：弧中点（菱形）与狗目标方块。

用法（仓库根目录）:
    python scripts/formation_sliders.py
    python scripts/formation_sliders.py --world_size 80 80
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.widgets import Slider

from typing import Tuple

from envs.defaults import DEFAULT_WORLD_SIZE_ARGV
from envs.high_level_action import HighLevelAction
from envs.sheep_scenario import world_radius_from_size

FLOCK_CENTER = np.zeros(2, dtype=np.float32)

decoder = HighLevelAction()


def draw_formation(
    ax,
    action: np.ndarray,
    num_herders: int,
    world_size: Tuple[float, float],
):
    ax.clear()
    world_r = world_radius_from_size(world_size)
    pad = max(world_r * 0.08, 1.0)
    ax.set_xlim(-world_r - pad, world_r + pad)
    ax.set_ylim(-world_r - pad, world_r + pad)
    ax.set_aspect("equal")
    ax.set_xlabel("x（相对羊质心）")
    ax.set_ylabel("y（相对羊质心）")
    ax.set_title(
        f"羊质心系 · θ_in=a[0]·π（弧中点→羊）· 场地 R={world_r:.1f}"
    )

    ax.add_patch(
        patches.Circle(
            (0.0, 0.0),
            world_r,
            fill=False,
            edgecolor="0.45",
            linewidth=1.2,
            linestyle="--",
            label="场地边界",
        )
    )

    action = np.clip(action.astype(np.float32), -1.0, 1.0)
    num_herders = int(np.clip(round(num_herders), 1, 20))

    decoded = decoder.decode_action(action, FLOCK_CENTER, world_size)
    radius = decoded["radius"]
    coverage = decoded["coverage"]
    theta_mid = float(decoded["theta_mid_rad"])
    theta_mid_deg = float(np.degrees(theta_mid))

    pos = decoder.sample_herder_positions(
        num_herders,
        FLOCK_CENTER,
        radius,
        coverage,
        theta_mid,
    )

    span_rad, at_max_span = decoder.effective_span_radians(coverage, num_herders)
    span_deg = float(np.degrees(span_rad))

    O = np.zeros(2, dtype=float)
    circ = patches.Circle(
        O, radius, fill=False, color="orange", linewidth=2, alpha=0.85, zorder=2
    )
    ax.add_patch(circ)

    if span_deg > 0.5:
        wedge = patches.Wedge(
            O,
            radius,
            theta_mid_deg - span_deg / 2.0,
            theta_mid_deg + span_deg / 2.0,
            width=radius * 0.12,
            facecolor="orange",
            alpha=0.15,
            edgecolor="darkorange",
            linewidth=1.0,
            zorder=1,
        )
        ax.add_patch(wedge)

    ax.plot(0.0, 0.0, "o", color="red", markersize=12, label="羊质心/弧心(0,0)", zorder=6)
    ax.plot(0.0, 0.0, "g*", markersize=22, label="目标(与原点重合时)", zorder=7, alpha=0.9)

    # 弧中点（在站位圆上）：羊质心 → 该点的方向即 θ_mid；该点 → 羊质心为反向
    mid_x = float(radius * np.cos(theta_mid))
    mid_y = float(radius * np.sin(theta_mid))
    ax.plot(mid_x, mid_y, "D", color="darkorange", markersize=8, label="弧中点", zorder=8)
    if radius > 0.4:
        ax.annotate(
            "",
            xy=(0.0, 0.0),
            xytext=(mid_x, mid_y),
            arrowprops=dict(
                arrowstyle="->",
                color="darkorange",
                lw=1.0,
                shrinkA=8,
                shrinkB=4,
            ),
            zorder=4,
        )
        ax.text(
            mid_x * 0.55,
            mid_y * 0.55,
            "θ_in",
            fontsize=8,
            color="darkorange",
            ha="center",
            va="center",
            zorder=5,
        )

    for i in range(num_herders):
        ax.plot(
            pos[i, 0],
            pos[i, 1],
            "s",
            color="blue",
            markersize=10,
            markeredgecolor="navy",
            zorder=4,
        )
        ax.annotate(
            str(i),
            (pos[i, 0], pos[i, 1]),
            textcoords="offset points",
            xytext=(4, 4),
            fontsize=9,
            color="navy",
        )

    mode = decoder.get_formation_mode(coverage)
    mx = " [Θ_max]" if at_max_span else ""
    theta_max_deg = float(
        np.degrees(decoder._theta_max_for_n(num_herders))
        if num_herders >= 2
        else 0.0
    )
    ti_deg = float(np.degrees(decoded["theta_in_rad"]))
    info = (
        f"{mode}{mx}  |  R={radius:.2f}  |  cov={coverage:.3f}  Θ={span_deg:.0f}°"
        f" (Θ_max={theta_max_deg:.0f}°)\n"
        f"θ_in=a[0]·π={ti_deg:.0f}°（弧中点→羊） θ_mid={theta_mid_deg:.0f}° |  N={num_herders}"
    )
    ax.text(
        0.02,
        0.98,
        info,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.85),
    )
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)


def main():
    parser = argparse.ArgumentParser(
        description="队形滑块：羊质心系，θ_mid 由 a[0] 直接给定"
    )
    parser.add_argument(
        "--world_size",
        type=float,
        nargs=2,
        default=list(DEFAULT_WORLD_SIZE_ARGV),
        metavar=("W", "H"),
        help="与 train_ppo 相同语义；圆半径 R=min(W,H)/2",
    )
    args = parser.parse_args()
    world_size = (float(args.world_size[0]), float(args.world_size[1]))

    fig = plt.figure(figsize=(10, 9))

    ax = fig.add_axes([0.12, 0.32, 0.86, 0.63])

    slider_h = 0.030
    gap = 0.004
    y0 = 0.26
    axcolor = "lightgoldenrodyellow"
    sliders: list[Slider] = []

    labels = (
        "a[0] θ_in=a[0]·π（弧中点→羊质心，世界系）",
        "a[1] → 站位半径 R∈[5,20]",
        "a[2] → coverage→弧张角 Θ",
        "机械狗数量 N",
    )
    inits = [0.0, 0.0, 0.0, 3.0]
    ranges = [(-1, 1), (-1, 1), (-1, 1), (1, 20)]

    for i, (lab, v0, (vmin, vmax)) in enumerate(zip(labels, inits, ranges)):
        y = y0 - i * (slider_h + gap)
        ax_s = fig.add_axes([0.12, y, 0.72, slider_h], facecolor=axcolor)
        st = 0.02 if i < 3 else 1
        s = Slider(ax_s, lab, vmin, vmax, valinit=v0, valstep=st)
        sliders.append(s)

    def update(_=None):
        a = np.array(
            [sliders[0].val, sliders[1].val, sliders[2].val, 0.0, 0.0],
            dtype=np.float32,
        )
        n = sliders[3].val
        draw_formation(ax, a, n, world_size)
        fig.canvas.draw_idle()

    for s in sliders:
        s.on_changed(update)

    update()
    plt.show()


if __name__ == "__main__":
    main()
