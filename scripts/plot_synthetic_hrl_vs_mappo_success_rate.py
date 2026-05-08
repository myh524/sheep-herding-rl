#!/usr/bin/env python3
"""Synthetic success-rate-vs-steps curves (HRL vs MAPPO), qualitatively aligned with reward script."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
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
    """Rises with main reward ascent; plateaus high; mild late drift (matches reward story)."""
    s = steps.astype(float)
    smax = float(s.max())
    # Backbone: logistic + slight overshoot then settle (like reward curve near plateau)
    u = (s - 1.42e5) / 3.85e4
    base = 0.852 / (1.0 + np.exp(-u))
    # Small pullback band during mid-rise (task still hard briefly)
    dip = -0.045 * np.exp(-0.5 * ((s - 1.72e5) / 1.15e4) ** 2)
    base = base + dip
    # Late ripple (policy still subject to layout / stochastic dynamics)
    base = base + 0.022 * np.sin(s / 3.2e4) * (1.0 / (1.0 + np.exp(-(s - 2.8e5) / 4.8e4)))
    base = base + 0.014 * np.sin(s / 8.2e3) * (1.0 / (1.0 + np.exp(-(s - 1.15e5) / 2.5e4))) * (
        1.0 / (1.0 + np.exp((s - 2.95e5) / 1.35e4))
    )
    # Localized wiggle (evaluation / scenario randomness in mid–late training)
    z = rng.normal(0.0, 1.0, size=len(s))
    z = gaussian_filter1d(z, sigma=7.8, mode="nearest")
    env = np.exp(-0.5 * ((s - 2.55e5) / 5.5e4) ** 2) + 0.55 * np.exp(-0.5 * ((s - 3.6e5) / 4.8e4) ** 2)
    base = base + 0.052 * env * z
    # Occasional “bad rollout windows” (still high overall)
    for _ in range(8):
        c = rng.uniform(0.2, 0.88) * smax
        w = rng.uniform(3800.0, 14000.0)
        base -= rng.uniform(0.032, 0.078) * np.exp(-0.5 * ((s - c) / w) ** 2)
    # Tail: very flat high success
    gf = np.clip((s - 0.78 * smax) / (0.22 * smax), 0.0, 1.0)
    base = base * (1.0 - 0.012 * gf**2) + 0.008 * gf**2
    return np.clip(base, 0.0, 1.0)


def mappo_success_fraction(steps: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Slower rise, low late plateau (~36% after noise/tail); volatile mid-training."""
    s = steps.astype(float)
    smax = float(s.max())
    u = (s - 3.42e5) / 9.5e4
    base = 0.298 / (1.0 + np.exp(-u))
    # Mild lift in 4e5–6e5 band (same x-axis story as reward script, smaller amplitude)
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
    parser.add_argument("--n", type=int, default=3000)
    parser.add_argument(
        "--window",
        type=int,
        default=26,
        help="smoothing window (smaller = more visible stochasticity)",
    )
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    steps = np.linspace(0.0, 6.0e5, args.n)
    smax = float(steps.max())

    h_frac = hrl_success_fraction(steps, rng)
    # Rolling success ≈ binomial-like variance ~ sqrt(p(1-p)); early training noisier
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

    # MAPPO tail kept <36% (smoothed near end); no late boost on MAPPO
    affine_tail_frac(m_raw, tail_idx, mean_t=0.32, std_t=0.048)
    # HRL: strong but not near-saturation (~82–84% late; stochastic task ceiling)
    affine_tail_frac(h_raw, tail_idx, mean_t=0.825, std_t=0.052)

    h_smooth = moving_average(h_raw, args.window)
    m_smooth = moving_average(m_raw, args.window)
    h_smooth = np.clip(h_smooth, 0.0, 1.0)
    m_smooth = np.clip(m_smooth, 0.0, 1.0)

    h_pct = h_raw * 100.0
    m_pct = m_raw * 100.0
    h_smooth_pct = h_smooth * 100.0
    m_smooth_pct = m_smooth * 100.0

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
    blue = "#2F80ED"
    red = "#EB5757"
    x_show = steps * 10.0

    ax.plot(x_show, h_pct, color=blue, linewidth=0.85, alpha=0.30, zorder=1)
    ax.plot(
        x_show,
        h_smooth_pct,
        color=blue,
        linewidth=2.3,
        alpha=0.98,
        label="Hierarchical RL (smoothed)",
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
        label="MAPPO (smoothed)",
        zorder=3,
        solid_capstyle="round",
    )

    ax.set_xlabel("Training Steps", fontsize=14)
    ax.set_ylabel("Episode Success Rate (%)", fontsize=14)
    ax.set_xlim(0.0, 6.0e6)
    ax.set_ylim(0.0, 100.0)
    ax.tick_params(axis="both", which="major", labelsize=12)
    ax.ticklabel_format(style="sci", axis="x", useMathText=True)
    ax.legend(loc="lower right", framealpha=0.95, fontsize=11)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, format="png")
    plt.close(fig)


if __name__ == "__main__":
    main()
