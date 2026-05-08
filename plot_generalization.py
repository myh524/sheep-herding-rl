"""
泛化评估结果折线图：仅依赖 numpy + matplotlib，可供 evaluate_generalization 与
scripts/plot_generalization_json.py 调用（重绘 JSON 时无需安装 PyTorch）。

图中轴标签与图例为英文（N=羊群规模、H=机械狗数），便于在无 CJK 字体的服务器上导出。
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


def cell_records_to_metric_grid(
    cell_records: List[Dict[str, Any]],
    sheep_list: Sequence[int],
    herder_list: Sequence[int],
    key: str,
) -> np.ndarray:
    """shape (len(herder_list), len(sheep_list))，缺失或 null 为 nan。"""
    idx = {(int(r["num_sheep"]), int(r["num_herders"])): r for r in cell_records}
    g = np.full((len(herder_list), len(sheep_list)), np.nan, dtype=np.float64)
    for j, nh in enumerate(herder_list):
        for i, ns in enumerate(sheep_list):
            r = idx.get((ns, nh))
            if r is None:
                continue
            v = r.get(key)
            if v is None:
                continue
            try:
                fv = float(v)
            except (TypeError, ValueError):
                continue
            if np.isnan(fv):
                continue
            g[j, i] = fv
    return g


def save_generalization_line_figures(
    out_dir: str,
    sheep_list: Sequence[int],
    herder_list: Sequence[int],
    cell_records: List[Dict[str, Any]],
    figure_prefix: str = "",
    suptitle: Optional[str] = None,
) -> List[str]:
    """
    为泛化网格生成折线图：每个指标一张图，左图为「横轴羊数、多条线对应狗数」，
    右图为「横轴狗数、多条线对应羊数」。成功类指标在子集无成功时为 nan，折线可能断开。
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)
    pref = figure_prefix.strip()
    if pref and not pref.endswith("_"):
        pref = pref + "_"

    plt.rcParams["axes.unicode_minus"] = False

    saved: List[str] = []
    # 轴标签用英文，避免无 CJK 字体环境缺字；图例 H=herders, N=sheep flock size
    specs: List[Tuple[str, str, str, Optional[Tuple[float, float]]]] = [
        ("success_rate", "Success rate (%)", "success", (0.0, 100.0)),
        ("mean_steps_on_success", "Mean steps (success ep.)", "steps_success", None),
        ("mean_spread_on_success", "Flock spread at term. (success ep.)", "spread_success", None),
    ]

    sheep_x = [float(x) for x in sheep_list]
    herder_x = [float(x) for x in herder_list]

    for json_key, ylabel, stem, ylim in specs:
        raw = cell_records_to_metric_grid(cell_records, sheep_list, herder_list, json_key)
        if json_key == "success_rate":
            data = raw * 100.0
        else:
            data = raw.copy()

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.3), constrained_layout=True)
        if suptitle:
            fig.suptitle(suptitle, fontsize=10, color="0.35")

        ax1.set_xlabel("Flock size N (#sheep)")
        ax1.set_ylabel(ylabel)
        ax1.grid(True, alpha=0.35)
        ax1.set_xticks(sheep_x)
        for j, nh in enumerate(herder_list):
            y = data[j, :]
            ax1.plot(
                sheep_x,
                y,
                "o-",
                linewidth=2.0,
                markersize=7,
                label=f"H={nh}",
            )
        ax1.legend(title="herders", loc="best", fontsize=9, ncol=2)
        if ylim is not None:
            ax1.set_ylim(ylim[0], ylim[1] * 1.05 if ylim[1] > 0 else ylim[1])

        ax2.set_xlabel("Num herders H")
        ax2.set_ylabel(ylabel)
        ax2.grid(True, alpha=0.35)
        ax2.set_xticks(herder_x)
        for i, ns in enumerate(sheep_list):
            y = data[:, i]
            ax2.plot(
                herder_x,
                y,
                "s-",
                linewidth=2.0,
                markersize=7,
                label=f"N={ns}",
            )
        ax2.legend(title="flock size", loc="best", fontsize=9, ncol=2)
        if ylim is not None:
            ax2.set_ylim(ylim[0], ylim[1] * 1.05 if ylim[1] > 0 else ylim[1])

        out_path = os.path.join(out_dir, f"{pref}gen_lines_{stem}.png")
        fig.savefig(out_path, dpi=160)
        plt.close(fig)
        saved.append(out_path)

    fig2, (bx1, bx2) = plt.subplots(1, 2, figsize=(12.5, 4.3), constrained_layout=True)
    if suptitle:
        fig2.suptitle(suptitle + " (all episodes)", fontsize=10, color="0.35")

    steps_all = cell_records_to_metric_grid(
        cell_records, sheep_list, herder_list, "mean_steps_all"
    )
    spread_all = cell_records_to_metric_grid(
        cell_records, sheep_list, herder_list, "mean_spread_all"
    )

    bx1.set_xlabel("Flock size N (#sheep)")
    bx1.set_ylabel("Mean steps (all ep.)")
    bx1.grid(True, alpha=0.35)
    bx1.set_xticks(sheep_x)
    for j, nh in enumerate(herder_list):
        bx1.plot(sheep_x, steps_all[j, :], "o-", linewidth=2.0, markersize=7, label=f"H={nh}")
    bx1.legend(title="herders", loc="best", fontsize=9, ncol=2)

    bx2.set_xlabel("Num herders H")
    bx2.set_ylabel("Mean steps (all ep.)")
    bx2.grid(True, alpha=0.35)
    bx2.set_xticks(herder_x)
    for i, ns in enumerate(sheep_list):
        bx2.plot(herder_x, steps_all[:, i], "s-", linewidth=2.0, markersize=7, label=f"N={ns}")
    bx2.legend(title="flock size", loc="best", fontsize=9, ncol=2)

    p_steps = os.path.join(out_dir, f"{pref}gen_lines_steps_all.png")
    fig2.savefig(p_steps, dpi=160)
    plt.close(fig2)
    saved.append(p_steps)

    fig3, (cx1, cx2) = plt.subplots(1, 2, figsize=(12.5, 4.3), constrained_layout=True)
    if suptitle:
        fig3.suptitle(suptitle + " (flock spread, all ep.)", fontsize=10, color="0.35")

    cx1.set_xlabel("Flock size N (#sheep)")
    cx1.set_ylabel("Flock spread at term. (all ep.)")
    cx1.grid(True, alpha=0.35)
    cx1.set_xticks(sheep_x)
    for j, nh in enumerate(herder_list):
        cx1.plot(sheep_x, spread_all[j, :], "o-", linewidth=2.0, markersize=7, label=f"H={nh}")
    cx1.legend(title="herders", loc="best", fontsize=9, ncol=2)

    cx2.set_xlabel("Num herders H")
    cx2.set_ylabel("Flock spread at term. (all ep.)")
    cx2.grid(True, alpha=0.35)
    cx2.set_xticks(herder_x)
    for i, ns in enumerate(sheep_list):
        cx2.plot(herder_x, spread_all[:, i], "s-", linewidth=2.0, markersize=7, label=f"N={ns}")
    cx2.legend(title="flock size", loc="best", fontsize=9, ncol=2)

    p_sp = os.path.join(out_dir, f"{pref}gen_lines_spread_all.png")
    fig3.savefig(p_sp, dpi=160)
    plt.close(fig3)
    saved.append(p_sp)

    return saved
