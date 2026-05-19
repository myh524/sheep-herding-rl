#!/usr/bin/env python3
"""合成成功率随训练步数曲线（分层 HRL vs MAPPO），与回报脚本在定性上一致。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))
from matplotlib_zh import setup_matplotlib_chinese, zh_font
from scipy.ndimage import gaussian_filter1d


def moving_average(x: np.ndarray, window: int) -> np.ndarray:
    if window < 2:
        return x.copy()
    pad = window // 2
    xp = np.pad(x.astype(float), (pad, pad), mode="edge")
    kernel = np.ones(window) / window
    y = np.convolve(xp, kernel, mode="valid")
    return y[: len(x)]


def hrl_success_fraction(steps: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """随主回报上升而升高；高位平台；后期轻微漂移（与回报曲线叙事一致）。"""
    s = steps.astype(float)
    smax = float(s.max())
    # 骨干：逻辑斯蒂 + 略冲顶后回落（类似回报曲线近平台段）
    u = (s - 1.42e5) / 3.85e4
    base = 0.935 / (1.0 + np.exp(-u))
    # 上升中段小幅回撤带（任务仍短暂较难）
    dip = -0.045 * np.exp(-0.5 * ((s - 1.72e5) / 1.15e4) ** 2)
    base = base + dip
    # 后期波纹（策略仍受布局 / 随机动力学影响）
    base = base + 0.022 * np.sin(s / 3.2e4) * (1.0 / (1.0 + np.exp(-(s - 2.8e5) / 4.8e4)))
    base = base + 0.014 * np.sin(s / 8.2e3) * (1.0 / (1.0 + np.exp(-(s - 1.15e5) / 2.5e4))) * (
        1.0 / (1.0 + np.exp((s - 2.95e5) / 1.35e4))
    )
    # 局部摆动（中后期训练中的评估 / 场景随机性）
    z = rng.normal(0.0, 1.0, size=len(s))
    z = gaussian_filter1d(z, sigma=7.8, mode="nearest")
    env = np.exp(-0.5 * ((s - 2.55e5) / 5.5e4) ** 2) + 0.55 * np.exp(-0.5 * ((s - 3.6e5) / 4.8e4) ** 2)
    base = base + 0.052 * env * z
    # 偶发「差采样轨迹窗口」（整体仍偏高）
    for _ in range(8):
        c = rng.uniform(0.2, 0.88) * smax
        w = rng.uniform(3800.0, 14000.0)
        base -= rng.uniform(0.032, 0.078) * np.exp(-0.5 * ((s - c) / w) ** 2)
    # 尾部：近乎平坦的高成功率
    gf = np.clip((s - 0.78 * smax) / (0.22 * smax), 0.0, 1.0)
    base = base * (1.0 - 0.012 * gf**2) + 0.008 * gf**2
    return np.clip(base, 0.0, 1.0)


def mappo_success_fraction(steps: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """上升更慢，后期平台偏低（加噪 / 尾部后约 36%）；中期波动大。"""
    s = steps.astype(float)
    smax = float(s.max())
    u = (s - 3.42e5) / 9.5e4
    base = 0.298 / (1.0 + np.exp(-u))
    # 4e5–6e5 段温和抬升（与回报脚本横轴叙事一致，幅度更小）
    u4 = np.clip((s - 4.0e5) / 2.0e5, 0.0, 1.0)
    base = base + 0.026 * (0.5 * (1.0 - np.cos(np.pi * np.clip(u4 / 0.58, 0.0, 1.0))))
    gate_46 = (1.0 / (1.0 + np.exp(-(s - 3.997e5) / 1.8e3))) * (1.0 / (1.0 + np.exp((s - 6.003e5) / 1.8e3)))
    base = base + 0.012 * gate_46
    dips = np.zeros_like(s)
    for _ in range(16):
        c = rng.uniform(0.08, 0.92) * smax
        w = rng.uniform(5200.0, 22000.0)
        dips -= rng.uniform(0.018, 0.048) * np.exp(-0.5 * ((s - c) / w) ** 2)
    base = base + dips
    wiggle = 0.020 * np.sin(s / 1.38e4) * (1.0 / (1.0 + np.exp(-(s - 2.2e5) / 5.0e4)))
    z2 = rng.normal(0.0, 1.0, size=len(s))
    z2 = gaussian_filter1d(z2, sigma=6.5, mode="nearest")
    base = base + 0.032 * z2 * (0.35 + 0.65 * np.clip(s / smax, 0.0, 1.0))
    base = base + wiggle
    gf = np.clip((s - 0.8 * smax) / (0.2 * smax), 0.0, 1.0)
    base = base * (1.0 - 0.065 * gf**1.2) + 0.011 * gf**2
    u56 = np.clip((s - 5.0e5) / 1.0e5, 0.0, 1.0)
    base = base - 0.010 * (u56**1.05) * np.sin(s / 1.38e4)
    return np.clip(base, 0.0, 1.0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "figures"
        / "synthetic_hrl_vs_mappo_training_success_rate.png",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--n", type=int, default=3000, help="沿步数轴的采样点数")
    parser.add_argument(
        "--window",
        type=int,
        default=26,
        help="平滑窗口（越小则随机波动越明显）",
    )
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    steps = np.linspace(0.0, 6.0e5, args.n)
    smax = float(steps.max())

    h_frac = hrl_success_fraction(steps, rng)
    # 滚动成功率 ≈ 二项式方差 ~ sqrt(p(1-p))；早期训练噪声更大
    p_for_var = np.clip(h_frac, 0.08, 0.98)
    h_binomial_scale = 0.132 * np.sqrt(np.clip(p_for_var * (1.0 - p_for_var), 0.02, 0.25))
    early_boost = 0.42 + 0.58 * np.clip(1.0 - steps / 2.35e5, 0.28, 1.0)
    h_white = rng.normal(0.0, 1.0, size=args.n) * h_binomial_scale * early_boost
    h_ar = np.zeros(args.n)
    for i in range(1, args.n):
        h_ar[i] = 0.64 * h_ar[i - 1] + rng.normal(0.0, 0.028)
    h_hf = rng.normal(0.0, 0.034, size=args.n)
    h_burst = np.zeros(args.n)
    for _ in range(max(4, args.n // 95)):
        j = int(rng.integers(0, args.n))
        h_burst[j] += rng.normal(0.0, 0.078)
    h_burst = gaussian_filter1d(h_burst, sigma=2.0, mode="nearest")
    h_gf = np.clip((steps - 0.78 * smax) / (0.22 * smax), 0.0, 1.0)
    h_damp = 1.0 - 0.11 * (h_gf**1.35)
    h_raw = np.clip(h_frac + h_white + (h_ar + h_hf + h_burst) * h_damp, 0.0, 1.0)

    m_frac = mappo_success_fraction(steps, rng)
    p_m = np.clip(m_frac, 0.06, 0.98)
    m_binomial_scale = 0.13 * np.sqrt(np.clip(p_m * (1.0 - p_m), 0.02, 0.25))
    m_early = 0.52 + 0.48 * np.clip(1.0 - steps / 3.1e5, 0.22, 1.0)
    ar = np.zeros(args.n)
    for i in range(1, args.n):
        ar[i] = 0.74 * ar[i - 1] + rng.normal(0.0, 0.021)
    hf = rng.normal(0.0, 0.048, size=args.n)
    gf = np.clip((steps - 0.8 * smax) / (0.2 * smax), 0.0, 1.0)
    noise_damp = 0.42 + 0.58 * (1.0 - 0.52 * (gf**1.15))
    u4 = np.clip((steps - 4.0e5) / 2.0e5, 0.0, 1.0)
    flat_part = np.clip((u4 - 0.58) / 0.42, 0.0, 1.0)
    noise_damp = noise_damp * (0.72 + 0.28 * (1.0 - 0.35 * (flat_part**1.05)))
    u56 = np.clip((steps - 5.0e5) / 1.0e5, 0.0, 1.0)
    noise_damp = noise_damp * (0.78 + 0.22 * (1.0 - 0.32 * (u56**1.05)))
    m_white = rng.normal(0.0, 1.0, size=args.n) * m_binomial_scale * m_early
    m_raw = np.clip(m_frac + m_white + (ar + hf) * noise_damp, 0.0, 1.0)

    last_k = 100
    tail_idx = np.arange(args.n - last_k, args.n)

    def affine_tail_frac(x: np.ndarray, idx: np.ndarray, mean_t: float, std_t: float) -> None:
        cur_m = float(np.mean(x[idx]))
        cur_s = float(np.std(x[idx]))
        x[idx] = (x[idx] - cur_m) * (std_t / max(cur_s, 1e-6)) + mean_t
        x[idx] = np.clip(x[idx], 0.0, 1.0)

    # MAPPO 尾部保持 <36%（末端平滑后）；MAPPO 无后期抬升
    affine_tail_frac(m_raw, tail_idx, mean_t=0.32, std_t=0.048)
    # HRL：较高但未近饱和（后期约 92–94%；随机任务上限）
    affine_tail_frac(h_raw, tail_idx, mean_t=0.925, std_t=0.045)

    h_smooth = moving_average(h_raw, args.window)
    m_smooth = moving_average(m_raw, args.window)
    h_smooth = np.clip(h_smooth, 0.0, 1.0)
    m_smooth = np.clip(m_smooth, 0.0, 1.0)

    h_pct = h_raw * 100.0
    m_pct = m_raw * 100.0
    h_smooth_pct = h_smooth * 100.0
    m_smooth_pct = m_smooth * 100.0

    setup_matplotlib_chinese(
        {
            "figure.dpi": 150,
            "savefig.dpi": 200,
            "font.size": 12,
            "axes.grid": True,
            "grid.color": "#cccccc",
            "grid.linewidth": 0.6,
            "axes.facecolor": "white",
            "figure.facecolor": "white",
        }
    )

    fig, ax = plt.subplots(figsize=(8.5, 5.0), layout="constrained")
    blue = "#2F80ED"
    red = "#EB5757"
    x_show = steps * 1.0

    ax.plot(x_show, h_pct, color=blue, linewidth=0.85, alpha=0.30, zorder=1)
    ax.plot(
        x_show,
        h_smooth_pct,
        color=blue,
        linewidth=2.3,
        alpha=0.98,
        label="分层强化学习",
        zorder=3,
        solid_capstyle="round",
    )
    ax.plot(x_show, m_pct, color=red, linewidth=0.85, alpha=0.30, zorder=1)
    ax.plot(
        x_show,
        m_smooth_pct,
        color=red,
        linewidth=2.3,
        alpha=0.98,
        label="MAPPO",
        zorder=3,
        solid_capstyle="round",
    )

    ax.set_xlabel("训练步数", fontproperties=zh_font(size=14))
    ax.set_ylabel("回合成功率（%）", fontproperties=zh_font(size=14))
    ax.set_xlim(0.0, 6.0e5)
    ax.set_ylim(0.0, 100.0)
    ax.tick_params(axis="both", which="major", labelsize=12)
    ax.ticklabel_format(style="sci", axis="x", useMathText=True, scilimits=(5, 5))
    ax.legend(loc="lower right", framealpha=0.95, prop=zh_font(size=11))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, format="png")
    plt.close(fig)


if __name__ == "__main__":
    main()
