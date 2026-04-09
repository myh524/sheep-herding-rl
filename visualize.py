"""
模型可视化运行脚本
加载训练好的模型并实时渲染环境。

Linux / SSH 无 DISPLAY 时默认使用 Agg 后端（不弹窗），用 time.sleep 代替 plt.pause；
本地桌面请安装 python3-tk（TkAgg）或 PyQt5（Qt5Agg）并设置 DISPLAY；也可设置 MPLBACKEND。
"""

import argparse
import os
import sys
import time
import warnings

import importlib
import numpy as np
import torch


def _gui_deps_available(backend: str) -> bool:
    """matplotlib.use 成功不等于能弹窗；例如未装 python3-tk 时 TkAgg 会在首帧才崩。"""
    b = backend.lower()
    if b == "agg":
        return True
    if b == "tkagg":
        try:
            import tkinter  # noqa: F401
        except ImportError:
            return False
        return True
    if b == "qt5agg":
        try:
            importlib.import_module("matplotlib.backends.backend_qt5agg")
        except Exception:
            return False
        return True
    if b == "qtagg":
        try:
            importlib.import_module("matplotlib.backends.backend_qtagg")
        except Exception:
            return False
        return True
    if b == "gtk3agg":
        try:
            importlib.import_module("matplotlib.backends.backend_gtk3agg")
        except Exception:
            return False
        return True
    return True


def _configure_matplotlib_backend(force_headless: bool) -> str:
    """必须在 import matplotlib.pyplot 之前调用。"""
    import matplotlib

    if force_headless:
        matplotlib.use("Agg", force=True)
        return "Agg"

    env_backend = os.environ.get("MPLBACKEND")
    if env_backend:
        try:
            matplotlib.use(env_backend, force=True)
            if env_backend.lower() != "agg" and not _gui_deps_available(env_backend):
                raise ImportError(f"backend {env_backend} dependencies missing")
            return env_backend
        except Exception:
            print(
                f"警告: MPLBACKEND={env_backend} 不可用（例如 TkAgg 需安装 python3-tk），"
                "已回退到 Agg。要弹窗请: sudo apt install python3-tk 或安装 PyQt5 后使用 Qt5Agg。",
                file=sys.stderr,
            )
            matplotlib.use("Agg", force=True)
            return "Agg"

    has_display = bool(
        os.environ.get("DISPLAY")
        or os.environ.get("WAYLAND_DISPLAY")
    )
    if not has_display:
        matplotlib.use("Agg", force=True)
        return "Agg"

    for name in ("TkAgg", "Qt5Agg", "QtAgg", "GTK3Agg"):
        try:
            matplotlib.use(name, force=True)
            if not _gui_deps_available(name):
                continue
            return name
        except Exception:
            continue

    matplotlib.use("Agg", force=True)
    return "Agg"


def _parse_early_args():
    """仅解析 --force_headless，以便在导入 pyplot 前选后端。"""
    fh = False
    i = 1
    while i < len(sys.argv):
        if sys.argv[i] == "--force_headless":
            fh = True
            break
        if sys.argv[i].startswith("--force_headless="):
            fh = sys.argv[i].split("=", 1)[1].lower() in ("1", "true", "yes")
            break
        i += 1
    return fh


_FORCE_HEADLESS = _parse_early_args()
_MATPLOTLIB_BACKEND = _configure_matplotlib_backend(_FORCE_HEADLESS)

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.widgets import Button
from typing import Any, Dict, List, Optional

if _MATPLOTLIB_BACKEND.lower() == "agg":
    warnings.filterwarnings(
        "ignore",
        message="FigureCanvasAgg is non-interactive",
        category=UserWarning,
    )


def _matplotlib_is_interactive() -> bool:
    return plt.get_backend().lower() != "agg"


from dataclasses import dataclass

from envs import SheepFlockEnv
from onpolicy.algorithms.ppo_actor_critic import PPOActorCritic, ImprovedActorCritic


@dataclass
class VisualizationState:
    current_step: int = 0
    total_reward: float = 0.0
    episode_done: bool = False
    success: bool = False
    action_history: List[np.ndarray] = None
    
    def __post_init__(self):
        if self.action_history is None:
            self.action_history = []


def parse_args():
    parser = argparse.ArgumentParser(description='可视化运行训练好的PPO模型')
    
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--num_episodes', type=int, default=5)
    parser.add_argument('--render_delay', type=int, default=30)
    
    parser.add_argument('--num_sheep', type=int, default=10)
    parser.add_argument('--num_herders', type=int, default=3)
    parser.add_argument(
        '--world_size',
        type=float,
        nargs=2,
        default=[100.0, 100.0],
        help='场地 (W,H)：圆形半径 R=min/2，目标在圆心',
    )
    parser.add_argument('--episode_length', type=int, default=150)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument(
        '--herder_teleport',
        action='store_true',
        default=False,
        help='机械狗直接瞬移到编队目标（关闭运动学），与 train_ppo --herder_teleport 一致',
    )
    parser.add_argument(
        '--formation_delta',
        action='store_true',
        default=False,
        help='与 train_ppo --formation_delta 一致：观测 13 维、a[0:3] 为编队增量',
    )
    parser.add_argument(
        '--formation_delta_theta_max_deg',
        type=float,
        default=22.5,
    )
    parser.add_argument('--formation_delta_radius_max', type=float, default=3.0)
    parser.add_argument('--formation_delta_coverage_max', type=float, default=0.15)
    parser.add_argument(
        '--high_level_interval',
        type=int,
        default=None,
        help='与 train_ppo 一致；默认 None 使用环境自动默认',
    )
    parser.add_argument(
        '--stochastic_policy',
        action='store_true',
        default=False,
        help='策略按分布随机采样动作（与 train_ppo 采集时 deterministic=False 一致）；默认关闭则为确定性均值动作',
    )
    
    parser.add_argument('--hidden_size', type=int, default=256)
    parser.add_argument('--layer_N', type=int, default=3)
    parser.add_argument('--use_ReLU', action='store_true', default=True)
    parser.add_argument('--use_orthogonal', action='store_true', default=True)
    parser.add_argument('--use_feature_normalization', action='store_true', default=True)
    parser.add_argument('--recurrent_N', type=int, default=1)
    parser.add_argument('--use_recurrent_policy', action='store_true', default=False)
    parser.add_argument('--use_naive_recurrent_policy', action='store_true', default=False)
    parser.add_argument('--gain', type=float, default=0.01)
    parser.add_argument('--use_policy_active_masks', action='store_true', default=True)
    parser.add_argument('--use_popart', action='store_true', default=False)
    parser.add_argument('--use_improved', action='store_true', default=False)
    parser.add_argument('--stacked_frames', type=int, default=1)
    
    parser.add_argument('--save_gif', type=str, default=None)
    parser.add_argument('--save_video', type=str, default=None)
    parser.add_argument(
        '--force_headless',
        action='store_true',
        help='强制使用无窗口 Agg 后端（需在命令中放在靠前位置以便生效，或设置 MPLBACKEND=Agg）',
    )
    parser.add_argument(
        '--headless_progress_interval',
        type=int,
        default=30,
        help='无图形窗口时每 N 步打印一行进度；设为 0 关闭',
    )

    return parser.parse_args()


def create_args_from_checkpoint(checkpoint_path: str, use_improved: bool = False) -> argparse.Namespace:
    args = argparse.Namespace()
    args.hidden_size = 256
    args.layer_N = 3
    args.use_ReLU = True
    args.use_orthogonal = True
    args.use_feature_normalization = True
    args.recurrent_N = 1
    args.use_recurrent_policy = False
    args.use_naive_recurrent_policy = False
    args.gain = 0.01
    args.use_policy_active_masks = True
    args.use_popart = False
    args.stacked_frames = 1
    return args


def detect_model_type(checkpoint: dict) -> str:
    if 'policy_state_dict' in checkpoint:
        state_dict = checkpoint['policy_state_dict']
    else:
        state_dict = checkpoint
    
    for key in state_dict.keys():
        if 'actor.0' in key or 'critic.0' in key:
            return 'improved'
        if 'actor.base.mlp.fc1' in key:
            return 'ppo'
    
    if any('feature_norm' in k for k in state_dict.keys()):
        return 'improved'
    
    return 'ppo'


def detect_layer_N(checkpoint: dict) -> int:
    if 'policy_state_dict' in checkpoint:
        state_dict = checkpoint['policy_state_dict']
    else:
        state_dict = checkpoint
    
    fc2_indices = set()
    for key in state_dict.keys():
        if 'actor.base.mlp.fc2.' in key:
            parts = key.split('.')
            for i, part in enumerate(parts):
                if part == 'fc2' and i + 1 < len(parts):
                    try:
                        idx = int(parts[i + 1])
                        fc2_indices.add(idx)
                    except ValueError:
                        pass
    
    if fc2_indices:
        return len(fc2_indices)
    return 3


def detect_policy_obs_dim(checkpoint: dict):
    """从 checkpoint 推断 Actor 输入维数（与 env.obs_dim 对齐检查用）。"""
    sd = checkpoint.get("policy_state_dict", checkpoint)
    w = sd.get("actor.base.mlp.fc1.0.weight")
    if w is not None:
        return int(w.shape[1])
    w = sd.get("actor.base.feature_norm.weight")
    if w is not None:
        return int(w.shape[0])
    return None


def detect_hidden_size(checkpoint: dict) -> int:
    if 'policy_state_dict' in checkpoint:
        state_dict = checkpoint['policy_state_dict']
    else:
        state_dict = checkpoint
    
    for key in state_dict.keys():
        if 'actor.base.mlp.fc1.0.weight' in key:
            return state_dict[key].shape[0]
        if 'actor.base.mlp.fc_h.0.weight' in key:
            return state_dict[key].shape[0]
    return 256


def load_model(model_path: str, env: SheepFlockEnv, device: torch.device, 
               args: argparse.Namespace = None):
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    model_type = detect_model_type(checkpoint)
    
    if args is None:
        args = create_args_from_checkpoint(model_path, model_type == 'improved')
    
    args.layer_N = detect_layer_N(checkpoint)
    args.hidden_size = detect_hidden_size(checkpoint)
    
    if model_type == 'improved':
        print(f"Detected ImprovedActorCritic")
        policy = ImprovedActorCritic(
            args, env.observation_space, env.action_space, device,
            hidden_size=args.hidden_size,
        ).to(device)
    else:
        print(f"Detected PPOActorCritic (layer_N={args.layer_N}, hidden_size={args.hidden_size})")
        policy = PPOActorCritic(
            args, env.observation_space, env.action_space, device,
        ).to(device)

    ckpt_obs = detect_policy_obs_dim(checkpoint)
    env_obs = int(env.observation_space.shape[0])
    if ckpt_obs is not None and ckpt_obs != env_obs:
        raise RuntimeError(
            f"观测维数不一致：checkpoint 的 Actor 输入维为 {ckpt_obs}，当前环境为 {env_obs}。\n"
            f"  - 若模型是旧版（10 维）训练：可视化请去掉 --formation_delta。\n"
            f"  - 若是增量编队训练（13 维）：请加 --formation_delta（或 VIS_FORMATION_DELTA=1）。"
        )

    if 'policy_state_dict' in checkpoint:
        policy.load_state_dict(checkpoint['policy_state_dict'])
        print(f"Loaded model: step {checkpoint.get('total_num_steps', 'unknown')}")
    else:
        policy.load_state_dict(checkpoint)
    
    policy.eval()
    return policy, args


class Visualizer:
    def __init__(self, env: SheepFlockEnv, policy, device: torch.device, args: argparse.Namespace):
        self.env = env
        self.policy = policy
        self.device = device
        self.args = args
        
        self.state = VisualizationState()
        self.fig = None
        self.ax = None
        self.paused = False
        self.frames = []
        
        self.rnn_states_actor = None
        self.rnn_states_critic = None
        self.masks = None
        self.current_obs = None
        
        self.is_improved = isinstance(policy, ImprovedActorCritic)
        self._interactive = _matplotlib_is_interactive()
        self._gui_shown = False

    def setup_figure(self):
        self.fig, self.ax = plt.subplots(figsize=(10, 10))
        R = float(self.env.scenario.world_radius)
        pad = max(R * 0.06, 1.0)
        self.ax.set_xlim(-R - pad, R + pad)
        self.ax.set_ylim(-R - pad, R + pad)
        self.ax.set_aspect('equal')
        self.ax.set_facecolor('#f5f5f5')
        self.ax.add_patch(
            patches.Circle(
                (0.0, 0.0), R, fill=False, edgecolor="0.5", linewidth=1.2, linestyle="--"
            )
        )

        if self._interactive:
            ax_pause = plt.axes([0.85, 0.01, 0.12, 0.04])
            self.btn_pause = Button(ax_pause, 'Pause/Resume')
            self.btn_pause.on_clicked(self.toggle_pause)
            self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        else:
            self.btn_pause = None
        
    def toggle_pause(self, event):
        self.paused = not self.paused
        
    def on_key_press(self, event):
        if event.key == ' ':
            self.paused = not self.paused
        
    def reset_episode(self):
        self.state = VisualizationState()
        self.current_obs = self.env.reset()
        
        if not self.is_improved:
            self.rnn_states_actor = np.zeros(
                (1, self.args.recurrent_N, self.args.hidden_size), dtype=np.float32
            )
            self.rnn_states_critic = np.zeros(
                (1, self.args.recurrent_N, self.args.hidden_size), dtype=np.float32
            )
        else:
            self.rnn_states_actor = np.zeros((1, 1), dtype=np.float32)
            self.rnn_states_critic = np.zeros((1, 1), dtype=np.float32)
        self.masks = np.ones((1, 1), dtype=np.float32)
        
    def get_action(self, obs: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            obs_tensor = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)
            rnn_states_actor_tensor = torch.from_numpy(self.rnn_states_actor).to(self.device)
            rnn_states_critic_tensor = torch.from_numpy(self.rnn_states_critic).to(self.device)
            masks_tensor = torch.from_numpy(self.masks).to(self.device)
            
            deterministic = not getattr(self.args, "stochastic_policy", False)
            value, action, action_log_prob, rnn_states_actor_out, rnn_states_critic_out = \
                self.policy.get_actions(
                    obs_tensor, rnn_states_actor_tensor, rnn_states_critic_tensor,
                    masks_tensor, deterministic=deterministic,
                )
            
            self.rnn_states_actor = rnn_states_actor_out.cpu().numpy()
            self.rnn_states_critic = rnn_states_critic_out.cpu().numpy()
            
            return action.cpu().numpy()[0]
    
    def render_env(self):
        self.ax.clear()
        
        R = float(self.env.scenario.world_radius)
        pad = max(R * 0.06, 1.0)
        self.ax.set_xlim(-R - pad, R + pad)
        self.ax.set_ylim(-R - pad, R + pad)
        self.ax.set_aspect('equal')
        self.ax.set_facecolor('#f5f5f5')

        arena = patches.Circle(
            (0.0, 0.0), R, fill=False, edgecolor="0.5", linewidth=1.2, linestyle="--"
        )
        self.ax.add_patch(arena)
        
        target = self.env.scenario.get_target_position()
        target_circle = patches.Circle(target, 3.0, color='green', alpha=0.3)
        self.ax.add_patch(target_circle)
        self.ax.plot(target[0], target[1], 'g*', markersize=20)

        # 与当前仿真一致：[4]–[7] 为轴对齐四向极值 e0..e3/R，与 r_envelope 包络矩形相同
        obs_vec = self.env.scenario.get_observation()
        if obs_vec is not None and len(obs_vec) >= 8:
            o4, o5, o6, o7 = (
                float(obs_vec[4]),
                float(obs_vec[5]),
                float(obs_vec[6]),
                float(obs_vec[7]),
            )
            tx, ty = float(target[0]), float(target[1])
            w_rect = (o4 + o6) * R
            h_rect = (o5 + o7) * R
            x0 = tx - o6 * R
            y0 = ty - o7 * R
            if len(self.env.scenario.sheep) > 0 and w_rect > 1e-4 and h_rect > 1e-4:
                env_rect = patches.Rectangle(
                    (x0, y0),
                    w_rect,
                    h_rect,
                    linewidth=1.6,
                    edgecolor="darkviolet",
                    facecolor="mediumpurple",
                    alpha=0.14,
                    zorder=2,
                )
                self.ax.add_patch(env_rect)
            area_bb = w_rect * h_rect
            area_norm = area_bb / max((2.0 * R) ** 2, 1e-6)
            obs_txt = (
                "观测 [4]–[7]：轴对齐四向极值 e0..e3 / R（与包络惩罚一致）\n"
                f"[4] e0=max(Δx+)/R = {o4:.3f}  [5] e1=max(Δy+)/R = {o5:.3f}\n"
                f"[6] e2=max(Δx−)/R = {o6:.3f}  [7] e3=max(Δy−)/R = {o7:.3f}\n"
                f"AABB 面积 A = {area_bb:.1f}  (A/r_scale² = {area_norm:.3f}, r_scale=2R)"
            )
            self.ax.text(
                0.02,
                0.98,
                obs_txt,
                transform=self.ax.transAxes,
                va="top",
                ha="left",
                fontsize=8,
                family="monospace",
                color="#222",
                bbox=dict(boxstyle="round,pad=0.35", facecolor="white", alpha=0.88, edgecolor="0.6"),
                zorder=10,
            )
        
        flock_state = self.env.scenario.get_flock_state()
        sheep_positions = flock_state['positions']
        
        if len(sheep_positions) > 0:
            sheep_positions = np.array(sheep_positions)
            self.ax.scatter(sheep_positions[:, 0], sheep_positions[:, 1], 
                           c='gray', s=80, alpha=0.7, marker='o', edgecolors='black')
            
            flock_center = self.env.scenario.get_flock_center()
            self.ax.plot(flock_center[0], flock_center[1], 'k+', markersize=12, markeredgewidth=2)
        
        herder_positions = self.env.scenario.herder_positions.copy()
        
        if hasattr(self.env.scenario, 'herder_targets'):
            herder_targets = self.env.scenario.herder_targets
            target_pos = self.env.scenario.get_target_position()
            # 与仿真一致：羊质心/弧来自「上次高层刷新」时的 decode，而非每步策略输出
            # （env.high_level_interval 内 herder_targets 不变，但 action_history 每步都在变）
            viz_dec = getattr(self.env, "_last_formation_decoded", None)
            flock_c = None
            radius = None
            coverage = None
            theta_mid = None
            if viz_dec is not None:
                flock_c = np.asarray(viz_dec["flock_center"], dtype=float)
                radius = float(viz_dec["radius"])
                coverage = float(viz_dec["coverage"])
                theta_mid = float(viz_dec["theta_mid_rad"])
            elif len(self.state.action_history) > 0 and not getattr(
                self.env, "formation_delta_mode", False
            ):
                last_action = self.state.action_history[-1]
                fc = self.env.scenario.get_flock_center()
                decoded = self.env.action_decoder.decode_action(
                    last_action,
                    fc,
                    self.env.world_size,
                )
                flock_c = np.asarray(decoded["flock_center"], dtype=float)
                radius = float(decoded["radius"])
                coverage = float(decoded["coverage"])
                theta_mid = float(decoded["theta_mid_rad"])

            if flock_c is not None and radius is not None:
                dec = self.env.action_decoder
                O = flock_c.astype(float)
                theta_mid_deg = float(np.degrees(theta_mid))
                span_rad, at_max_span = dec.effective_span_radians(
                    coverage, self.env.num_herders
                )
                span_deg = float(np.degrees(span_rad))

                arc_ring = patches.Circle(
                    O, radius, fill=False, color="orange", linewidth=2, alpha=0.7
                )
                self.ax.add_patch(arc_ring)

                self.ax.plot(O[0], O[1], "o", color="red", markersize=12,
                            markeredgecolor="darkred", markeredgewidth=2)

                if span_deg > 0.5:
                    wedge = patches.Wedge(
                        O,
                        radius,
                        theta_mid_deg - span_deg / 2.0,
                        theta_mid_deg + span_deg / 2.0,
                        width=radius * 0.15,
                        facecolor="orange",
                        alpha=0.2,
                        edgecolor="darkorange",
                        linewidth=1.5,
                    )
                    self.ax.add_patch(wedge)
                
                for i in range(self.env.num_herders):
                    target_pos_i = herder_targets[i]
                    
                    self.ax.plot(target_pos_i[0], target_pos_i[1], 'o', color='darkorange', 
                                markersize=8, alpha=0.6, markeredgecolor='black', markeredgewidth=1)
                    
                    if i < len(herder_positions):
                        hpos = herder_positions[i]
                        direction = target_pos_i - hpos
                        dist = np.linalg.norm(direction)
                        
                        if dist > 0.5:
                            direction = direction / dist
                            arrow_len = min(dist * 0.4, 3.0)
                            self.ax.annotate('', 
                                            xy=(hpos[0] + direction[0] * arrow_len, 
                                                hpos[1] + direction[1] * arrow_len),
                                            xytext=hpos,
                                            arrowprops=dict(arrowstyle='->', color='green', 
                                                           lw=1.5, alpha=0.8))
                
                mode = self.env.action_decoder.get_formation_mode(coverage)
                mx = " MAX" if at_max_span else ""
                self.ax.text(
                    O[0],
                    O[1] - 2,
                    f"{mode}{mx} | R={radius:.1f} cov={coverage:.2f} Θ={span_deg:.0f}°",
                    ha="center",
                    fontsize=9,
                    color="darkorange",
                    fontweight="bold",
                )
        
        evasion_r = float(
            self.env.scenario.sheep_config.get("evasion_radius", 8.0)
        )
        for i, hpos in enumerate(herder_positions):
            evasion_circle = patches.Circle(
                hpos,
                evasion_r,
                fill=True,
                facecolor="blue",
                alpha=0.08,
                edgecolor="blue",
                linestyle="--",
                linewidth=1,
            )
            self.ax.add_patch(evasion_circle)
        
        self.ax.scatter(herder_positions[:, 0], herder_positions[:, 1],
                       c='blue', s=120, alpha=0.9, marker='s', edgecolors='darkblue',
                       label='Herder')
        
        legend_elements = [
            plt.Line2D([0], [0], marker='*', color='w', markerfacecolor='green', 
                      markersize=15, label='Target'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='gray',
                      markersize=10, markeredgecolor='black', label='Sheep'),
            plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='blue',
                      markersize=10, markeredgecolor='darkblue', label='Herder'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red',
                      markersize=10, markeredgecolor='darkred', label='Arc center (flock)'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='darkorange',
                      markersize=8, markeredgecolor='black', label='Sampled Target'),
            patches.Patch(facecolor='orange', alpha=0.2, edgecolor='darkorange',
                         label='Station Region'),
            patches.Patch(facecolor='blue', alpha=0.08, edgecolor='blue',
                         linestyle='--', label='Evasion Zone'),
        ]
        self.ax.legend(handles=legend_elements, loc='upper right', fontsize=8)
        
        status = "PAUSED" if self.paused else "RUNNING"
        self.ax.set_title(f'Step {self.state.current_step} | Reward: {self.state.total_reward:.1f} | {status}', 
                         fontsize=12, fontweight='bold')
        self.ax.grid(True, alpha=0.3)
        
    def run_episode(self, episode_num: int, total_episodes: int):
        print(f"\n{'='*50}")
        print(f"Episode {episode_num}/{total_episodes}")
        print(f"{'='*50}")
        
        self.reset_episode()
        self.state.episode_done = False
        
        if self.fig is None:
            self.setup_figure()
        
        for step in range(self.env.episode_length):
            while self.paused:
                if self._interactive:
                    plt.pause(0.1)
                else:
                    time.sleep(0.1)

            action = self.get_action(self.current_obs)

            actions_env = np.zeros((self.env.num_herders, len(action)), dtype=np.float32)
            for i in range(self.env.num_herders):
                actions_env[i] = action.copy()

            obs, reward, done, info = self.env.step(actions_env)

            self.state.current_step += 1
            self.state.total_reward += reward
            self.state.action_history.append(action.copy())

            self.render_env()

            if self._interactive and not self._gui_shown:
                plt.show(block=False)
                self._gui_shown = True

            if self.args.save_gif or self.args.save_video:
                self.fig.canvas.draw()
                self.frames.append(np.array(self.fig.canvas.renderer.buffer_rgba()))

            delay_s = self.args.render_delay / 1000.0
            if self._interactive:
                plt.pause(delay_s)
            else:
                time.sleep(delay_s)
                iv = self.args.headless_progress_interval
                if iv > 0 and self.state.current_step % iv == 0:
                    print(
                        f"  step {self.state.current_step}/{self.env.episode_length}",
                        flush=True,
                    )

            self.current_obs = obs

            if isinstance(info, dict):
                self.state.success = bool(info.get('is_success', False))
            if done:
                break

        print(f"\nEpisode {episode_num} finished:")
        print(f"  Steps: {self.state.current_step}, Reward: {self.state.total_reward:.3f}")
        print(f"  Result: {'SUCCESS!' if self.state.success else 'FAILED'}")
        
        return self.state.success, self.state.total_reward, self.state.current_step
    
    def run(self):
        successes = []
        total_rewards = []
        total_steps = []
        
        for ep in range(self.args.num_episodes):
            success, reward, steps = self.run_episode(ep + 1, self.args.num_episodes)
            successes.append(success)
            total_rewards.append(reward)
            total_steps.append(steps)
        
        print(f"\n{'='*50}")
        print("Visualization completed!")
        print(f"Success rate: {sum(successes)}/{len(successes)} ({sum(successes)/len(successes)*100:.1f}%)")
        print(f"Average reward: {np.mean(total_rewards):.3f}")
        
        if self.args.save_gif:
            self.save_gif(self.args.save_gif)
        if self.args.save_video:
            self.save_video(self.args.save_video)
        
        plt.close('all')
    
    def save_gif(self, path: str):
        try:
            import imageio
            imageio.mimsave(path, self.frames, fps=20)
            print(f"GIF saved to: {path}")
        except ImportError:
            print("Warning: imageio required to save GIF")
    
    def save_video(self, path: str):
        try:
            import imageio
            imageio.mimsave(path, self.frames, fps=20)
            print(f"Video saved to: {path}")
        except ImportError:
            print("Warning: imageio required to save video")


def main():
    args = parse_args()

    print(f"Model: {args.model_path}")
    print(f"Config: {args.num_sheep} sheep, {args.num_herders} herders, world {args.world_size}")
    mode = "interactive" if _matplotlib_is_interactive() else "headless (无弹窗)"
    print(f"Matplotlib backend: {_MATPLOTLIB_BACKEND} ({mode})")
    if not _matplotlib_is_interactive():
        print(
            "提示: 无窗口运行可加 --save_gif out.gif；要弹窗请先 sudo apt install python3-tk "
            "（或安装 PyQt5），再在有 DISPLAY 的会话中运行。",
        )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if args.stochastic_policy:
        print("Policy: 随机采样动作 (--stochastic_policy，与训练 rollout 一致)")
        if args.seed is not None:
            s = int(args.seed)
            torch.manual_seed(s)
            if device.type == "cuda":
                torch.cuda.manual_seed_all(s)
    else:
        print("Policy: 确定性动作（默认；加 --stochastic_policy 可改为随机采样）")
    
    extra_kw = {}
    if getattr(args, "formation_delta", False):
        extra_kw.update(
            {
                "formation_delta_mode": True,
                "formation_delta_theta_max_rad": float(
                    np.deg2rad(float(args.formation_delta_theta_max_deg))
                ),
                "formation_delta_radius_max": float(args.formation_delta_radius_max),
                "formation_delta_coverage_max": float(args.formation_delta_coverage_max),
            }
        )
    if getattr(args, "high_level_interval", None) is not None:
        extra_kw["high_level_interval"] = int(args.high_level_interval)
    env = SheepFlockEnv(
        world_size=tuple(args.world_size),
        num_sheep=args.num_sheep,
        num_herders=args.num_herders,
        episode_length=args.episode_length,
        random_seed=args.seed,
        use_herder_kinematics=not args.herder_teleport,
        **extra_kw,
    )
    print(
        f"Env: formation_delta_mode={env.formation_delta_mode}, "
        f"high_level_interval={env.high_level_interval}, obs_dim={env.obs_dim}"
    )
    if not env.formation_delta_mode:
        parts = [
            "说明: 当前为**绝对动作**（默认），弧由 a[0:3] 直接解码。",
            "若训练用了 --formation_delta，须加 --formation_delta 且 obs=13。",
        ]
        if args.stochastic_policy:
            parts.append("已开 --stochastic_policy：每步从分布采样，占位易剧烈跳动；纯观察可关掉。")
        print(" ".join(parts))

    policy, args = load_model(args.model_path, env, device, args)
    
    visualizer = Visualizer(env, policy, device, args)
    visualizer.run()
    
    env.close()


if __name__ == '__main__':
    main()
