#!/usr/bin/env python3
"""
生成 12 张典型编队站位示意图，合成 3 列 × 4 行网格 PNG。

与 formation_sliders.py 使用同一解码与绘图风格（无图头/子图标题）。

用法（仓库根目录）:
    python scripts/generate_formation_gallery.py
    python scripts/generate_formation_gallery.py -o figures/formation_gallery.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from matplotlib_zh import setup_matplotlib_chinese  # noqa: E402

setup_matplotlib_chinese()

from formation_sliders import (  # noqa: E402
    _FORM_DEFAULT_WORLD_SIZE,
    draw_formation,
)

# (a0, a1, a2, N) — 覆盖单/多狗、方位、半径与弧张角典型组合
GALLERY_CASES: List[Tuple[float, float, float, int]] = [
    (0.0, 0.0, 0.0, 1),
    (0.0, -1.0, 1.0, 2),
    (0.5, 0.0, -0.7, 2),
    (0.0, 0.0, -0.9, 3),
    (0.0, 0.0, 0.55, 3),
    (0.0, 1.0, 0.95, 3),
    (-0.5, 0.0, 0.35, 4),
    (0.25, -1.0, 1.0, 4),
    (0.0, 1.0, 0.9, 5),
    (-0.75, -0.3, -0.5, 5),
    (0.0, 0.0, 1.0, 6),
    (0.6, 0.4, 0.75, 3),
]

NROWS, NCOLS = 4, 3


def _action_vector(a0: float, a1: float, a2: float) -> np.ndarray:
    return np.array([a0, a1, a2, 0.0, 0.0], dtype=np.float32)


def draw_gallery_panel(
    ax,
    action: np.ndarray,
    num_herders: int,
    world_size: Tuple[float, float],
) -> None:
    draw_formation(ax, action, num_herders, world_size)
    ax.set_title("")
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(labelsize=7)


def build_gallery_figure(
    world_size: Tuple[float, float],
    *,
    dpi: int = 150,
) -> plt.Figure:
    fig, axes = plt.subplots(
        NROWS,
        NCOLS,
        figsize=(NCOLS * 4.0, NROWS * 3.6),
        dpi=dpi,
        constrained_layout=True,
    )
    for ax, (a0, a1, a2, n) in zip(axes.flat, GALLERY_CASES):
        draw_gallery_panel(ax, _action_vector(a0, a1, a2), n, world_size)
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description="合成 12 张典型站位图（3×4 网格）")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=ROOT / "figures" / "formation_gallery.png",
        help="输出 PNG 路径",
    )
    parser.add_argument(
        "--world-size",
        type=float,
        nargs=2,
        default=list(_FORM_DEFAULT_WORLD_SIZE),
        metavar=("W", "H"),
        help="与 formation_sliders 一致",
    )
    parser.add_argument("--dpi", type=int, default=150)
    args = parser.parse_args()

    world_size = (float(args.world_size[0]), float(args.world_size[1]))
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    fig = build_gallery_figure(world_size, dpi=args.dpi)
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"已保存: {out.resolve()}", flush=True)


if __name__ == "__main__":
    main()
