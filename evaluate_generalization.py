#!/usr/bin/env python3
"""
泛化性实验：在多种羊群数量与机械狗数量组合下评估同一 checkpoint。

指标（每个组合）：
  - 平均成功率
  - 平均完成步数：默认在到达目标时提前结束 episode，步数即「完成」所用步数；
    另报告全体 episode 的平均步数（含超时失败则跑满 horizon）。
  - 完成时羊群扩散度：episode 结束瞬间的 flock_spread（质心距离 std）；
    报告「仅成功 episode」与「全体 episode」两套均值，便于对比。

用法示例：
  python evaluate_generalization.py --model_path runs/xxx/model.pt \\
      --num_episodes 50 --seed 0 --output_json results/gen.json \\
      --output_figures results/gen_plots

注意：策略网络 actor 输入维与羊/狗数量无关（观测已归一化），但训练时若未见过某规模，
泛化可能较差；机械狗数量不改变 actor 观测维，可与训练时不同。

启动时会从 checkpoint 权重推断观测维（10 或 13）；若为 13 则自动启用 formation_delta，
避免与训练时不一致导致的 load_state_dict 尺寸错误。

默认在羊群进入目标阈值后提前结束 episode，以便「完成步数」在成功轨迹上有意义；
若需与「始终跑满 horizon」的评估一致，请传入 --run_full_horizon。

默认动力学与常用训练对齐：**不启用圆盘边界**（等价原 `--no-disk-boundary`）、
**`--high_level_interval 3`**、**不启用 `--herder_teleport`**（机械狗用势场运动学，非瞬移）。
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from tqdm import tqdm

from envs import SheepFlockEnv
from envs.defaults import (
    DEFAULT_EVAL_EPISODE_LENGTH,
    DEFAULT_WORLD_SIZE_ARGV,
    FORMATION_DELTA_COVERAGE_MAX,
    FORMATION_DELTA_RADIUS_MAX,
    FORMATION_DELTA_THETA_MAX_DEG,
)
from evaluate_policy import (
    EpisodeResult,
    ImprovedActorCritic,
    load_model,
    run_episode,
)
from plot_generalization import save_generalization_line_figures


DEFAULT_SHEEP_GRID = (5, 10, 15, 20, 25)
DEFAULT_HERDER_GRID = (3, 4, 5, 6)


def _policy_state_dict_from_checkpoint(checkpoint: Any) -> Dict[str, Any]:
    if isinstance(checkpoint, dict) and "policy_state_dict" in checkpoint:
        sd = checkpoint["policy_state_dict"]
    else:
        sd = checkpoint
    return sd if isinstance(sd, dict) else {}


def infer_obs_dim_from_state_dict(sd: Dict[str, Any]) -> Optional[int]:
    """从权重推断 Actor/Critic 的观测维（与 SheepFlockEnv.obs_dim 一致）。"""
    if not sd:
        return None
    for key in ("actor.base.feature_norm.weight", "critic.base.feature_norm.weight"):
        t = sd.get(key)
        if t is not None and getattr(t, "shape", None) is not None and len(t.shape) >= 1:
            return int(t.shape[0])
    w = sd.get("actor.base.mlp.fc1.0.weight")
    if w is not None and getattr(w, "shape", None) is not None and len(w.shape) >= 2:
        return int(w.shape[1])
    for key in ("actor.0.weight", "critic.0.weight"):
        w = sd.get(key)
        if w is not None and getattr(w, "shape", None) is not None and len(w.shape) >= 2:
            return int(w.shape[1])
    return None


def resolve_formation_delta_for_checkpoint(
    obs_dim: Optional[int], cli_formation_delta: bool
) -> bool:
    """
    10 维 = 标准观测；13 维 = formation_delta 增广观测。
    若 checkpoint 为 13 维而命令行未开 formation_delta，则自动开启。
    """
    if obs_dim is None:
        return bool(cli_formation_delta)
    if obs_dim not in (10, 13):
        raise RuntimeError(
            f"从 checkpoint 解析到观测维 {obs_dim}，本脚本仅支持 10（标准）"
            f"与 13（formation_delta）。"
        )
    need_fd = obs_dim == 13
    if cli_formation_delta and not need_fd:
        raise ValueError(
            "已指定 --formation_delta，但 checkpoint 输入维为 10。"
            "请去掉 --formation_delta 后再评估。"
        )
    if need_fd and not cli_formation_delta:
        print(
            f"已从 checkpoint 推断 obs_dim={obs_dim}，自动启用 formation_delta（与训练对齐）。"
        )
    return need_fd


@dataclass
class CellStats:
    num_sheep: int
    num_herders: int
    num_episodes: int
    success_rate: float
    mean_steps_all: float
    std_steps_all: float
    mean_steps_on_success: Optional[float]
    std_steps_on_success: Optional[float]
    n_success: int
    mean_spread_on_success: Optional[float]
    std_spread_on_success: Optional[float]
    mean_spread_all: float
    std_spread_all: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "num_sheep": self.num_sheep,
            "num_herders": self.num_herders,
            "num_episodes": self.num_episodes,
            "success_rate": self.success_rate,
            "mean_steps_all": self.mean_steps_all,
            "std_steps_all": self.std_steps_all,
            "mean_steps_on_success": self.mean_steps_on_success,
            "std_steps_on_success": self.std_steps_on_success,
            "n_success": self.n_success,
            "mean_spread_on_success": self.mean_spread_on_success,
            "std_spread_on_success": self.std_spread_on_success,
            "mean_spread_all": self.mean_spread_all,
            "std_spread_all": self.std_spread_all,
        }


def _build_env(
    *,
    num_sheep: int,
    num_herders: int,
    world_size: Tuple[float, float],
    episode_length: int,
    seed: Optional[int],
    herder_teleport: bool,
    herder_physics_legacy: bool,
    no_disk_boundary: bool,
    herder_init: Optional[str],
    herder_assignment: Optional[str],
    formation_delta: bool,
    formation_delta_theta_max_deg: float,
    formation_delta_radius_max: float,
    formation_delta_coverage_max: float,
    high_level_interval: Optional[int],
    end_episode_when_at_target: bool,
) -> SheepFlockEnv:
    extra_kw: Dict[str, Any] = {}
    if formation_delta:
        extra_kw.update(
            {
                "formation_delta_mode": True,
                "formation_delta_theta_max_rad": float(
                    np.deg2rad(float(formation_delta_theta_max_deg))
                ),
                "formation_delta_radius_max": float(formation_delta_radius_max),
                "formation_delta_coverage_max": float(formation_delta_coverage_max),
            }
        )
    if high_level_interval is not None:
        extra_kw["high_level_interval"] = int(high_level_interval)
    if herder_physics_legacy:
        extra_kw["herder_physics_legacy"] = True
    if no_disk_boundary:
        extra_kw["enforce_disk_boundary"] = False
    hm: Dict[str, Any] = {}
    if herder_init:
        hm["herder_init_mode"] = str(herder_init)
    if herder_assignment:
        hm["herder_slot_assignment"] = str(herder_assignment)
    if hm:
        extra_kw["herder_motion"] = hm

    return SheepFlockEnv(
        world_size=world_size,
        num_sheep=num_sheep,
        num_herders=num_herders,
        episode_length=episode_length,
        random_seed=seed,
        use_herder_kinematics=not herder_teleport,
        end_episode_when_at_target=end_episode_when_at_target,
        **extra_kw,
    )


def _aggregate(results: List[EpisodeResult]) -> CellStats:
    n = len(results)
    ok = [r for r in results if r.success]
    steps = np.array([r.steps for r in results], dtype=np.float64)
    spreads = np.array([r.final_spread for r in results], dtype=np.float64)
    steps_s = np.array([r.steps for r in ok], dtype=np.float64) if ok else None
    spr_s = np.array([r.final_spread for r in ok], dtype=np.float64) if ok else None

    return CellStats(
        num_sheep=0,
        num_herders=0,
        num_episodes=n,
        success_rate=float(sum(r.success for r in results)) / max(n, 1),
        mean_steps_all=float(np.mean(steps)),
        std_steps_all=float(np.std(steps)),
        mean_steps_on_success=float(np.mean(steps_s)) if steps_s is not None and len(steps_s) else None,
        std_steps_on_success=float(np.std(steps_s)) if steps_s is not None and len(steps_s) > 1 else (
            0.0 if steps_s is not None and len(steps_s) == 1 else None
        ),
        n_success=len(ok),
        mean_spread_on_success=float(np.mean(spr_s)) if spr_s is not None and len(spr_s) else None,
        std_spread_on_success=float(np.std(spr_s)) if spr_s is not None and len(spr_s) > 1 else (
            0.0 if spr_s is not None and len(spr_s) == 1 else None
        ),
        mean_spread_all=float(np.mean(spreads)),
        std_spread_all=float(np.std(spreads)),
    )


def _print_matrix(
    rows: Sequence[int],
    cols: Sequence[int],
    cells: Dict[Tuple[int, int], CellStats],
    attr: str,
    title: str,
) -> None:
    print(f"\n{title}")
    header = "羊\\狗".ljust(8) + "".join(f"{c:>10}" for c in cols)
    print(header)
    print("-" * len(header))
    for ns in rows:
        parts = []
        for nh in cols:
            v = getattr(cells[(ns, nh)], attr, None)
            if attr == "success_rate" and isinstance(v, (float, int)):
                parts.append(f"{100.0 * float(v):>9.1f}%")
            else:
                parts.append(f"{_fmt_cell(v):>10}")
        print(f"{ns:<8}" + "".join(parts))


def _fmt_cell(v: Any) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        if np.isnan(v):
            return "—"
        return f"{v:.3f}"
    return str(v)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="多规模泛化评估（羊数 × 狗数网格）")
    p.add_argument("--model_path", type=str, required=True, help="checkpoint 路径")
    p.add_argument("--num_episodes", type=int, default=50, help="每个组合评估的 episode 数")
    p.add_argument("--deterministic", action="store_true", default=True)
    p.add_argument(
        "--sheep_counts",
        type=int,
        nargs="+",
        default=list(DEFAULT_SHEEP_GRID),
        help="羊群数量列表，默认 5 10 15 20 25",
    )
    p.add_argument(
        "--herder_counts",
        type=int,
        nargs="+",
        default=list(DEFAULT_HERDER_GRID),
        help="机械狗数量列表，默认 3 4 5 6",
    )
    p.add_argument(
        "--world_size",
        type=float,
        nargs=2,
        default=list(DEFAULT_WORLD_SIZE_ARGV),
        help="场地 (W,H)",
    )
    p.add_argument("--episode_length", type=int, default=DEFAULT_EVAL_EPISODE_LENGTH)
    p.add_argument("--seed", type=int, default=None, help="每个组合内 reset 用固定种子时传入")
    p.add_argument(
        "--per_combo_seed",
        action="store_true",
        help="若设置，则组合 (ns,nh) 使用 seed + ns*1000 + nh，episode 间仍随机",
    )
    p.add_argument(
        "--run_full_horizon",
        action="store_true",
        default=False,
        help="设置后不因到达目标提前结束（每 episode 固定跑满 episode_length）",
    )
    p.add_argument(
        "--herder_teleport",
        action="store_true",
        default=False,
        help="机械狗瞬移到编队点（关闭势场运动学）。默认关闭，即低层为势场动力学。",
    )
    p.add_argument("--herder_physics_legacy", action="store_true", default=False)
    p.add_argument(
        "--disk-boundary",
        action="store_true",
        default=False,
        help="启用圆盘场地裁剪与边界斥力。默认关闭（与常用无圆盘边界训练一致）。",
    )
    p.add_argument(
        "--no-disk-boundary",
        dest="no_disk_boundary_flag",
        action="store_true",
        default=False,
        help="兼容其它脚本的显式写法；默认已是「无圆盘边界」，可不传。勿与 --disk-boundary 同用。",
    )
    p.add_argument("--herder_init", type=str, default=None, choices=["random_disk", "fixed_arc"])
    p.add_argument("--herder_assignment", type=str, default=None, choices=["min_cost", "ordered"])
    p.add_argument(
        "--formation_delta",
        action="store_true",
        default=False,
        help="强制开启编队增量观测；通常不必指定，脚本会从 checkpoint 自动对齐 10/13 维",
    )
    p.add_argument("--formation_delta_theta_max_deg", type=float, default=FORMATION_DELTA_THETA_MAX_DEG)
    p.add_argument("--formation_delta_radius_max", type=float, default=FORMATION_DELTA_RADIUS_MAX)
    p.add_argument("--formation_delta_coverage_max", type=float, default=FORMATION_DELTA_COVERAGE_MAX)
    p.add_argument(
        "--high_level_interval",
        type=int,
        default=3,
        help="高层编队决策刷新间隔（环境步）。默认 3。",
    )
    p.add_argument("--hidden_size", type=int, default=256)
    p.add_argument("--layer_N", type=int, default=3)
    p.add_argument("--use_ReLU", action="store_true", default=True)
    p.add_argument("--use_orthogonal", action="store_true", default=True)
    p.add_argument("--use_feature_normalization", action="store_true", default=True)
    p.add_argument("--recurrent_N", type=int, default=1)
    p.add_argument("--use_recurrent_policy", action="store_true", default=False)
    p.add_argument("--use_naive_recurrent_policy", action="store_true", default=False)
    p.add_argument("--gain", type=float, default=0.01)
    p.add_argument("--use_policy_active_masks", action="store_true", default=True)
    p.add_argument("--use_popart", action="store_true", default=False)
    p.add_argument("--use_improved", action="store_true", default=False)
    p.add_argument("--stacked_frames", type=int, default=1)
    p.add_argument("--output_json", type=str, default=None, help="汇总结果 JSON 路径")
    p.add_argument(
        "--output_figures",
        type=str,
        default=None,
        metavar="DIR",
        help="将指标画成折线图（双视角）并保存 PNG 到此目录；可与 --output_json 同时使用",
    )
    p.add_argument(
        "--figure_prefix",
        type=str,
        default="",
        help="PNG 文件名前缀（建议以 _ 结尾或脚本会自动补 _）",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    end_on_target = not bool(args.run_full_horizon)
    if bool(args.disk_boundary) and bool(getattr(args, "no_disk_boundary_flag", False)):
        raise SystemExit("错误: 不能同时使用 --disk-boundary 与 --no-disk-boundary。")
    no_disk_boundary = not bool(args.disk_boundary)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    world_size = (float(args.world_size[0]), float(args.world_size[1]))

    ckpt = torch.load(args.model_path, map_location="cpu", weights_only=False)
    sd = _policy_state_dict_from_checkpoint(ckpt)
    obs_dim_ckpt = infer_obs_dim_from_state_dict(sd)
    use_formation_delta = resolve_formation_delta_for_checkpoint(
        obs_dim_ckpt, args.formation_delta
    )
    if obs_dim_ckpt is None and not use_formation_delta:
        print(
            "警告: 未能从 checkpoint 解析观测维，formation_delta 保持关闭；"
            "若加载权重失败请检查模型结构或手动传入 --formation_delta。"
        )

    ref_ns, ref_nh = int(args.sheep_counts[0]), int(args.herder_counts[0])
    ref_env = _build_env(
        num_sheep=ref_ns,
        num_herders=ref_nh,
        world_size=world_size,
        episode_length=args.episode_length,
        seed=args.seed,
        herder_teleport=args.herder_teleport,
        herder_physics_legacy=args.herder_physics_legacy,
        no_disk_boundary=no_disk_boundary,
        herder_init=args.herder_init,
        herder_assignment=args.herder_assignment,
        formation_delta=use_formation_delta,
        formation_delta_theta_max_deg=args.formation_delta_theta_max_deg,
        formation_delta_radius_max=args.formation_delta_radius_max,
        formation_delta_coverage_max=args.formation_delta_coverage_max,
        high_level_interval=int(args.high_level_interval),
        end_episode_when_at_target=end_on_target,
    )

    policy, args_loaded = load_model(args.model_path, ref_env, device, args)
    is_improved = isinstance(policy, ImprovedActorCritic)
    ref_env.close()

    sheep_list = [int(x) for x in args.sheep_counts]
    herder_list = [int(x) for x in args.herder_counts]
    cells: Dict[Tuple[int, int], CellStats] = {}
    all_records: List[Dict[str, Any]] = []

    print(f"设备: {device}")
    print(f"模型: {args.model_path}")
    _od = obs_dim_ckpt if obs_dim_ckpt is not None else "未知"
    print(f"观测维(推断): {_od}；formation_delta={use_formation_delta}")
    print(
        f"环境: high_level_interval={int(args.high_level_interval)}；"
        f"圆盘边界={'启用' if args.disk_boundary else '关闭'}；"
        f"机械狗={'瞬移' if args.herder_teleport else '势场运动学'}"
    )
    print(
        f"网格: 羊 {sheep_list} × 狗 {herder_list}；每格 {args.num_episodes} episodes；"
        f"到达目标提前结束={end_on_target}"
    )

    for ns in sheep_list:
        for nh in herder_list:
            combo_seed = None
            if args.seed is not None and args.per_combo_seed:
                combo_seed = int(args.seed) + ns * 1000 + nh

            env = _build_env(
                num_sheep=ns,
                num_herders=nh,
                world_size=world_size,
                episode_length=args.episode_length,
                seed=combo_seed,
                herder_teleport=args.herder_teleport,
                herder_physics_legacy=args.herder_physics_legacy,
                no_disk_boundary=no_disk_boundary,
                herder_init=args.herder_init,
                herder_assignment=args.herder_assignment,
                formation_delta=use_formation_delta,
                formation_delta_theta_max_deg=args.formation_delta_theta_max_deg,
                formation_delta_radius_max=args.formation_delta_radius_max,
                formation_delta_coverage_max=args.formation_delta_coverage_max,
                high_level_interval=int(args.high_level_interval),
                end_episode_when_at_target=end_on_target,
            )

            results: List[EpisodeResult] = []
            desc = f"羊{ns}_狗{nh}"
            for _ in tqdm(range(args.num_episodes), desc=desc, leave=False):
                results.append(
                    run_episode(env, policy, device, args_loaded, is_improved, False)
                )

            st = _aggregate(results)
            st.num_sheep = ns
            st.num_herders = nh
            cells[(ns, nh)] = st
            all_records.append(st.to_dict())
            env.close()

    print("\n" + "=" * 72)
    print("泛化评估汇总表（逐格）")
    print("=" * 72)
    hdr = (
        f"{'羊':>4} {'狗':>4} {'成功率':>8} {'成功数':>6} "
        f"{'成功均步':>10} {'全体均步':>10} {'成功末扩散':>12} {'全体末扩散':>12}"
    )
    print(hdr)
    print("-" * len(hdr))
    for ns in sheep_list:
        for nh in herder_list:
            c = cells[(ns, nh)]
            ms = c.mean_steps_on_success
            mss = c.mean_spread_on_success
            print(
                f"{ns:4d} {nh:4d} {c.success_rate * 100:7.1f}% {c.n_success:6d} "
                f"{_fmt_cell(ms):>10} {_fmt_cell(c.mean_steps_all):>10} "
                f"{_fmt_cell(mss):>12} {_fmt_cell(c.mean_spread_all):>12}"
            )

    _print_matrix(
        sheep_list,
        herder_list,
        cells,
        "success_rate",
        "矩阵：成功率（0–1）",
    )
    _print_matrix(
        sheep_list,
        herder_list,
        cells,
        "mean_steps_on_success",
        "矩阵：成功 episode 平均完成步数（无成功则为 —）",
    )
    _print_matrix(
        sheep_list,
        herder_list,
        cells,
        "mean_spread_on_success",
        "矩阵：成功 episode 结束时平均羊群扩散度",
    )

    saved_figure_paths: Optional[List[str]] = None
    if args.output_figures:
        fig_title = os.path.basename(args.model_path)
        saved_figure_paths = save_generalization_line_figures(
            args.output_figures,
            sheep_list,
            herder_list,
            all_records,
            figure_prefix=args.figure_prefix,
            suptitle=fig_title,
        )
        print("\n折线图已保存:")
        for fp in saved_figure_paths:
            print(f"  {fp}")

    if args.output_json:
        out = {
            "model_path": os.path.abspath(args.model_path),
            "device": str(device),
            "obs_dim_inferred": obs_dim_ckpt,
            "formation_delta": use_formation_delta,
            "high_level_interval": int(args.high_level_interval),
            "enforce_disk_boundary": bool(args.disk_boundary),
            "herder_teleport": bool(args.herder_teleport),
            "end_episode_when_at_target": end_on_target,
            "episode_length": args.episode_length,
            "world_size": list(world_size),
            "sheep_counts": sheep_list,
            "herder_counts": herder_list,
            "num_episodes_per_cell": args.num_episodes,
            "figure_paths": (
                [os.path.abspath(p) for p in saved_figure_paths]
                if saved_figure_paths
                else None
            ),
            "cells": all_records,
        }
        path = os.path.abspath(args.output_json)
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        print(f"\n已写入 JSON: {path}")


if __name__ == "__main__":
    main()
