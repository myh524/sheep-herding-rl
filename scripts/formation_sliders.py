#!/usr/bin/env python3
"""
交互式队形可视化：坐标系固定在**羊质心**（原点），与训练解码一致。

- `a[0]·π` = `θ_in`（弧中点 → 羊质心）；几何用 `θ_mid = θ_in − π`（羊质心 → 弧中点）。
- `a[1]`、`a[2]`：R、coverage；`a[3]、a[4]` 解码固定为 0。
- 不依赖狗质心。橙色圆为站位圆周 R；蓝方块为机械狗编队目标点。

用法（仓库根目录）:
    python scripts/formation_sliders.py
    python scripts/formation_sliders.py --world_size 60 60

窗口右下角「保存场景」仅导出主绘图区（不含标题、坐标轴文字与底部滑块）。
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Tuple

ROOT = Path(__file__).resolve().parents[1]
_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.transforms import Bbox
from matplotlib.widgets import Button, Slider

from matplotlib_zh import setup_matplotlib_chinese, zh_font  # noqa: E402

setup_matplotlib_chinese()

from envs.high_level_action import HighLevelAction
from envs.sheep_scenario import world_radius_from_size

# 默认场地 60×60 → 圆半径 30，坐标轴约 ±30（原 100×100 为 ±50）
_FORM_DEFAULT_WORLD_SIZE: Tuple[float, float] = (50.0, 50.0)

FLOCK_CENTER = np.zeros(2, dtype=np.float32)

decoder = HighLevelAction()

# 与 visualize.py 轴/说明文字字号层级一致
_FORM_AXIS_FONTSIZE = 12
_FORM_TITLE_FONTSIZE = 13
_FORM_ANNOT_FONTSIZE = 9

# 保存场景时在显示坐标（pt）上扩边，避免裁掉上/右边框线
_SCENE_SAVE_PAD_LEFT_PT = 30.0
_SCENE_SAVE_PAD_BOTTOM_PT = 30.0
_SCENE_SAVE_PAD_RIGHT_PT = 8.0
_SCENE_SAVE_PAD_TOP_PT = 8.0


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
    ax.set_xlabel("x（相对羊质心）", fontproperties=zh_font(size=_FORM_AXIS_FONTSIZE))
    ax.set_ylabel("y（相对羊质心）", fontproperties=zh_font(size=_FORM_AXIS_FONTSIZE))
    ax.set_title(
        f"羊质心系 · θ_in=a[0]·π（弧中点→羊）· 场地 R={world_r:.1f}",
        fontproperties=zh_font(size=_FORM_TITLE_FONTSIZE),
    )

    action = np.clip(action.astype(np.float32), -1.0, 1.0)
    num_herders = int(np.clip(round(num_herders), 1, 20))

    decoded = decoder.decode_action(action, FLOCK_CENTER, world_size)
    radius = decoded["radius"]
    coverage = decoded["coverage"]
    theta_mid = float(decoded["theta_mid_rad"])

    pos = decoder.sample_herder_positions(
        num_herders,
        FLOCK_CENTER,
        radius,
        coverage,
        theta_mid,
    )

    ax.add_patch(
        patches.Circle(
            (0.0, 0.0),
            radius,
            fill=False,
            color="orange",
            linewidth=2,
            alpha=0.85,
            zorder=2,
        )
    )

    ax.plot(0.0, 0.0, "o", color="red", markersize=12, zorder=6)
    ax.plot(0.0, 0.0, "g*", markersize=11, zorder=7, alpha=0.9)

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
            fontproperties=zh_font(size=_FORM_ANNOT_FONTSIZE),
            color="navy",
        )

    ax.grid(True, alpha=0.3)


def save_formation_scene(fig, ax, out_dir: Path) -> Path:
    """仅保存主仿真 axes 区域（无标题/轴标签/图外文字）。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    hidden: list = []
    title_text = ax.get_title()
    if ax.title.get_visible():
        ax.title.set_visible(False)
        hidden.append(ax.title)
    xlab = ax.xaxis.label
    if xlab.get_visible():
        xlab.set_visible(False)
        hidden.append(xlab)
    ylab = ax.yaxis.label
    if ylab.get_visible():
        ylab.set_visible(False)
        hidden.append(ylab)

    fig.canvas.draw()
    ext = ax.get_window_extent(renderer)
    bbox_disp = Bbox.from_extents(
        ext.x0 - _SCENE_SAVE_PAD_LEFT_PT,
        ext.y0 - _SCENE_SAVE_PAD_BOTTOM_PT,
        ext.x1 + _SCENE_SAVE_PAD_RIGHT_PT,
        ext.y1 + _SCENE_SAVE_PAD_TOP_PT,
    )
    bbox = bbox_disp.transformed(fig.dpi_scale_trans.inverted())
    path = out_dir / f"formation_scene_{time.strftime('%Y%m%d_%H%M%S')}.png"
    fig.savefig(
        path,
        bbox_inches=bbox,
        pad_inches=0,
        dpi=150,
        facecolor=ax.get_facecolor(),
    )

    for artist in hidden:
        artist.set_visible(True)
    if title_text:
        ax.set_title(title_text, fontproperties=zh_font(size=_FORM_TITLE_FONTSIZE))
    ax.set_xlabel("x（相对羊质心）", fontproperties=zh_font(size=_FORM_AXIS_FONTSIZE))
    ax.set_ylabel("y（相对羊质心）", fontproperties=zh_font(size=_FORM_AXIS_FONTSIZE))
    fig.canvas.draw_idle()
    return path


def main():
    parser = argparse.ArgumentParser(
        description="队形滑块：羊质心系，θ_mid 由 a[0] 直接给定"
    )
    parser.add_argument(
        "--world_size",
        type=float,
        nargs=2,
        default=list(_FORM_DEFAULT_WORLD_SIZE),
        metavar=("W", "H"),
        help="场地 (W,H)；圆半径 R=min(W,H)/2，默认 60×60 即坐标约 ±30",
    )
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=ROOT / "figures" / "formation_sliders",
        help="「保存场景」按钮输出目录（默认 figures/formation_sliders）",
    )
    args = parser.parse_args()
    world_size = (float(args.world_size[0]), float(args.world_size[1]))
    save_dir = Path(args.save_dir)

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

    ax_btn = fig.add_axes([0.84, 0.02, 0.14, 0.04])
    btn_save = Button(ax_btn, "保存场景")

    def on_save(_event):
        path = save_formation_scene(fig, ax, save_dir)
        print(f"已保存仿真场景: {path}", flush=True)

    btn_save.on_clicked(on_save)

    update()
    plt.show()


if __name__ == "__main__":
    main()
