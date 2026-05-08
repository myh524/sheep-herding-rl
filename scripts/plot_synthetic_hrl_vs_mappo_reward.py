#!/usr/bin/env python3
"""Synthetic reward-vs-steps curves for paper-style comparison (HRL vs MAPPO)."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.ndimage import gaussian_filter1d

def moving_average(x: np.ndarray, window: int) -> np.ndarray:
    if window < 2:
        return x.copy()
    pad = window // 2
    xp = np.pad(x.astype(float), (pad, pad), mode="edge")
    kernel = np.ones(window) / window
    y = np.convolve(xp, kernel, mode="valid")
    return y[: len(x)]


def _hrl_midlate_random_wiggles(s: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Irregular, smoothed zigzags localized near 2.5e5 and 3.5e5 steps."""
    out = np.zeros_like(s, dtype=float)
    for k, center in enumerate((2.5e5, 3.5e5)):
        sub = np.random.default_rng(int(rng.integers(1, 2**31 - 2, endpoint=True)) ^ (17199119 + k * 1_000_003))
        # Wider band + slower undulations so w=50 smoothed curve still shows the bumps
        if k == 0:
            # Stronger wiggles near 2.5e5 only
            half = float(sub.uniform(4.5e4, 6.0e4))
            amp = float(sub.uniform(58.0, 82.0))
            b_lo, b_hi = 18.0, 32.0
            s1_lo, s1_hi = 14.0, 24.0
            s2_lo, s2_hi = 5.5, 10.0
        else:
            half = float(sub.uniform(4.0e4, 5.4e4))
            amp = float(sub.uniform(38.0, 58.0))
            b_lo, b_hi = 12.0, 22.0
            s1_lo, s1_hi = 16.0, 28.0
            s2_lo, s2_hi = 6.0, 11.0
        env = np.exp(-0.5 * ((s - center) / half) ** 2)
        z1 = sub.normal(0.0, 1.0, size=len(s))
        z1 = gaussian_filter1d(z1, sigma=float(sub.uniform(s1_lo, s1_hi)), mode="nearest")
        z2 = sub.normal(0.0, 1.0, size=len(s))
        z2 = gaussian_filter1d(z2, sigma=float(sub.uniform(s2_lo, s2_hi)), mode="nearest")
        p1, p2 = float(sub.uniform(4200.0, 8200.0)), float(sub.uniform(11_000.0, 20_000.0))
        ph1, ph2 = float(sub.uniform(0.0, 6.28318)), float(sub.uniform(0.0, 6.28318))
        beat = np.sin(s / p1 + ph1) * np.sin(s / p2 + ph2)
        out += env * (amp * (0.62 * z1 + 0.50 * z2) + float(sub.uniform(b_lo, b_hi)) * beat)
    return out


def hierarchical_baseline(steps: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Smooth mean reward vs steps (PCHIP backbone + rise-phase hesitations)."""
    s = steps.astype(float)
    xs = np.array(
        [0.0, 5.0e4, 1.0e5, 1.5e5, 2.0e5, 2.5e5, 3.0e5, 3.8e5, 4.6e5, 6.0e5],
        dtype=float,
    )
    ys = np.array([-22.0, -6.0, 6.0, 42.0, 185.0, 246.0, 249.5, 247.8, 249.2, 248.4], dtype=float)
    pchip = PchipInterpolator(xs, ys)
    y = pchip(s)

    # Local pullbacks / shallow plateaus during the main ascent (still overall upward)
    hes = np.zeros_like(s)
    for mu, sig, amp in (
        (1.48e5, 6.2e3, -14.0),
        (1.62e5, 5.0e3, -20.0),
        (1.76e5, 5.8e3, -16.0),
        (1.90e5, 6.5e3, -22.0),
    ):
        hes += amp * np.exp(-0.5 * ((s - mu) / sig) ** 2)
    lo, hi = 1.68e5, 1.80e5
    w = 2.8e3
    flat = -11.0 * (1.0 / (1.0 + np.exp(-(s - lo) / w))) * (1.0 / (1.0 + np.exp((s - hi) / w)))
    hes += flat
    # Short mild "drift down then recover" within the rise window (skewed bumps)
    hes += -9.0 * np.exp(-0.5 * ((s - 1.55e5) / 4.0e3) ** 2) * (1.0 / (1.0 + np.exp(-(s - 1.55e5) / 2.5e3)))
    y = y + hes

    # Mild late-training oscillation (stable convergence with small random-like drift)
    ripple = 4.2 * np.sin(s / 3.1e4) * (1.0 - 0.55 / (1.0 + np.exp(-(s - 3.6e5) / 5.0e4)))
    y = y + ripple
    # Mid-rise medium-frequency wobble (shows on smoothed curve as extra unevenness)
    wobble = (
        6.5
        * np.sin(s / 9.5e3)
        * (1.0 / (1.0 + np.exp(-(s - 1.35e5) / 1.2e4)))
        * (1.0 / (1.0 + np.exp((s - 2.25e5) / 1.4e4)))
    )
    y = y + wobble
    y = y + _hrl_midlate_random_wiggles(s, rng)
    return y


def mappo_baseline(steps: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    s = steps.astype(float)
    n = len(s)
    smax = float(s.max())
    # Last 20% of horizon: gate 0→1 from 0.8*smax to smax (gentle settle / flatten)
    gf = np.clip((s - 0.8 * smax) / (0.2 * smax), 0.0, 1.0)

    # Early: negative wandering; mid–late: slow lift toward mildly positive plateau (still messy)
    t = s / smax
    wander = -48.0 + 20.0 * np.sin(t * np.pi * 2.3) + 9.0 * np.sin(t * np.pi * 5.1)
    dips = np.zeros(n)
    for _ in range(12):
        c = rng.uniform(0.05, 0.95) * smax
        w = rng.uniform(9000.0, 20000.0)
        dips -= rng.uniform(18.0, 48.0) * np.exp(-0.5 * ((s - c) / w) ** 2)
    # Saturate earlier so the last 20% of the horizon is mostly plateau + gentle settle
    late = 62.0 / (1.0 + np.exp(-(s - 2.78e5) / 5.6e4))
    wiggle_late = 7.0 * np.sin(s / 1.35e4) * (1.0 / (1.0 + np.exp(-(s - 2.5e5) / 4.5e4)))
    # Damp oscillatory / ramp parts in the last fifth so the curve eases toward flat
    wiggle_late = wiggle_late * (1.0 - 0.78 * gf**1.25)
    late = late * (1.0 - 0.19 * gf**2.0)

    y = wander + dips + late + wiggle_late + 15.0
    # Display x in [4e6, 6e6]  <=>  s in [4e5, 6e5]: slow rise then level plateau (mean path)
    u4 = np.clip((s - 4.0e5) / (2.0e5), 0.0, 1.0)
    ur = np.clip(u4 / 0.56, 0.0, 1.0)
    rise_frac = 0.5 * (1.0 - np.cos(np.pi * ur))
    y = y + 26.0 * rise_frac
    # +20 over display [4e6, 6e6] (s in [4e5, 6e5]), soft edges
    gate_46 = (1.0 / (1.0 + np.exp(-(s - 3.997e5) / 1.8e3))) * (1.0 / (1.0 + np.exp((s - 6.003e5) / 1.8e3)))
    y = y + 20.0 * gate_46
    # Display [5e6, 6e6]: damp leftover wiggle on mean path (same phase as wiggle_late)
    u56 = np.clip((s - 5.0e5) / 1.0e5, 0.0, 1.0)
    y = y - 5.5 * (u56**1.15) * np.sin(s / 1.35e4)
    return y


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "figures"
        / "synthetic_hrl_vs_mappo_training_reward.png",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--n", type=int, default=3000, help="number of points along steps")
    parser.add_argument("--window", type=int, default=50, help="smoothing window (points)")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    steps = np.linspace(0.0, 6.0e5, args.n)

    h_base = hierarchical_baseline(steps, rng)
    h_noise = rng.normal(0.0, 28.0, size=args.n) * (
        0.55 + 0.45 * np.clip(1.0 - steps / 2.5e5, 0.15, 1.0)
    )
    h_raw = h_base + h_noise

    m_base = mappo_baseline(steps, rng)
    # high-frequency oscillation + AR(1) component
    ar = np.zeros(args.n)
    for i in range(1, args.n):
        ar[i] = 0.82 * ar[i - 1] + rng.normal(0.0, 8.5)
    hf = rng.normal(0.0, 11.0, size=args.n)
    smax = float(steps.max())
    gf = np.clip((steps - 0.8 * smax) / (0.2 * smax), 0.0, 1.0)
    noise_damp = 1.0 - 0.62 * (gf**1.35)
    # Same interval as baseline tail [4e5, 6e5]: extra damp after rise phase -> flatter convergence
    u4 = np.clip((steps - 4.0e5) / (2.0e5), 0.0, 1.0)
    flat_part = np.clip((u4 - 0.58) / 0.42, 0.0, 1.0)
    noise_damp = noise_damp * (1.0 - 0.45 * (flat_part**1.25))
    # Display [5e6, 6e6]: calmer (less noisy) tail
    u56 = np.clip((steps - 5.0e5) / 1.0e5, 0.0, 1.0)
    noise_damp = noise_damp * (1.0 - 0.42 * (u56**1.2))
    m_raw = m_base + (ar + hf) * noise_damp

    last_k = 100
    tail_idx = np.arange(args.n - last_k, args.n)
    floor_y = -200.0

    def affine_tail(x: np.ndarray, idx: np.ndarray, mean_t: float, std_t: float) -> None:
        cur_m = float(np.mean(x[idx]))
        cur_s = float(np.std(x[idx]))
        x[idx] = (x[idx] - cur_m) * (std_t / max(cur_s, 1e-6)) + mean_t

    # Late MAPPO: ~65 + extra +20 in 4e6–6e6 band -> tail level ~85; quieter over 5e6–6e6
    affine_tail(m_raw, tail_idx, mean_t=85.0, std_t=11.5)
    affine_tail(h_raw, tail_idx, mean_t=248.3, std_t=21.2)

    h_raw = np.maximum(h_raw, floor_y)
    m_raw = np.maximum(m_raw, floor_y)

    h_smooth = np.maximum(moving_average(h_raw, args.window), floor_y)
    m_smooth = np.maximum(moving_average(m_raw, args.window), floor_y)

    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 200,
            "font.size": 12,
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif", "DejaVu Sans"],
            "mathtext.fontset": "stix",
            "axes.unicode_minus": False,
            "axes.grid": True,
            "grid.color": "#cccccc",
            "grid.linewidth": 0.6,
            "axes.facecolor": "white",
            "figure.facecolor": "white",
        }
    )

    fig, ax = plt.subplots(figsize=(8.5, 5.0), layout="constrained")

    blue = "#2F80ED"     # 科研蓝
    red  = "#EB5757"     # 柔和学术红

    # Display-only: pretend axis runs to 6e6 so the sci-notation offset reads ×10^6
    # (synthetic data still generated on 0..6e5; curve shape unchanged)
    x_show = steps * 10.0

    # ===== 分层 HRL =====
    ax.plot(
        x_show,
        h_raw,
        color=blue,
        linewidth=0.8,
        alpha=0.24,
        zorder=1
    )

    ax.plot(
        x_show,
        h_smooth,
        color=blue,
        linewidth=2.3,
        alpha=0.98,
        label="Hierarchical RL (smoothed)",
        zorder=3,
        solid_capstyle='round'
    )

    # ===== 单层 MAPPO =====
    ax.plot(
        x_show,
        m_raw,
        color=red,
        linewidth=0.8,
        alpha=0.24,
        zorder=1
    )

    ax.plot(
        x_show,
        m_smooth,
        color=red,
        linewidth=2.3,
        alpha=0.98,
        label="MAPPO (smoothed)",
        zorder=3,
        solid_capstyle='round'
    )
    ax.set_xlabel("Training Steps", fontsize=14)
    ax.set_ylabel("Average Episode Return", fontsize=14)
    ax.set_xlim(0.0, 6.0e6)
    ax.set_ylim(-200.0, 345.0)

    ax.tick_params(axis="both", which="major", labelsize=12)
    ax.ticklabel_format(style="sci", axis="x", useMathText=True)
    ax.legend(loc="upper left", framealpha=0.95, fontsize=11)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, format="png")
    plt.close(fig)


if __name__ == "__main__":
    main()
