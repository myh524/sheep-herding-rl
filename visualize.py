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
from matplotlib.ticker import FuncFormatter
from matplotlib.widgets import Button
from typing import Any, Dict, List, Optional, Tuple

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
from envs.defaults import (
    DEFAULT_NUM_HERDERS,
    DEFAULT_NUM_SHEEP,
    DEFAULT_VIS_EPISODE_LENGTH,
    DEFAULT_WORLD_SIZE_ARGV,
    FORMATION_DELTA_COVERAGE_MAX,
    FORMATION_DELTA_RADIUS_MAX,
    FORMATION_DELTA_THETA_MAX_DEG,
)
from onpolicy.algorithms.ppo_actor_critic import PPOActorCritic, ImprovedActorCritic

# 编队采样目标点上的小圆（世界坐标半径）；alpha 越小越透明（0 全透明，1 不透明）
FORMATION_SAMPLE_CIRCLE_RADIUS = 1.2
FORMATION_SAMPLE_CIRCLE_ALPHA = 0.45
FORMATION_SAMPLE_CIRCLE_ZORDER = 3
# 机械狗应在采样圆之上；箭头略低于方块以免完全盖住箭头尖
HERDER_ARROW_ZORDER = 5
HERDER_SCATTER_ZORDER = 6
EVASION_ZONE_ZORDER = 2

# 仅改坐标轴刻度「读数」：数据仍为真实 world 米制；场地图形边界在刻度上读成 ± 该值（总跨度 2×）
VIZ_AXIS_LABEL_DISPLAY_HALF = 250.0


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


_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))


def _rewrite_root_figures_typo_path(path: Optional[str]) -> Optional[str]:
    """把误写的根路径 /figures/... 改为仓库内 figures/...，避免在磁盘根目录建文件夹失败。"""
    if not path:
        return None
    raw = str(path).strip()
    if raw == "/figures" or raw.startswith("/figures/"):
        rel = raw[1:].lstrip(os.sep)
        return os.path.normpath(os.path.join(_REPO_ROOT, rel))
    return raw


def parse_args():
    parser = argparse.ArgumentParser(description='可视化运行训练好的PPO模型')
    
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--num_episodes', type=int, default=5)
    parser.add_argument('--render_delay', type=int, default=30)
    
    parser.add_argument('--num_sheep', type=int, default=DEFAULT_NUM_SHEEP)
    parser.add_argument('--num_herders', type=int, default=DEFAULT_NUM_HERDERS)
    parser.add_argument(
        '--world_size',
        type=float,
        nargs=2,
        default=list(DEFAULT_WORLD_SIZE_ARGV),
        help='场地 (W,H)：圆形半径 R=min/2，目标在圆心',
    )
    parser.add_argument('--episode_length', type=int, default=DEFAULT_VIS_EPISODE_LENGTH)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument(
        '--herder_teleport',
        action='store_true',
        default=False,
        help='机械狗直接瞬移到编队目标（关闭运动学），与 train_ppo --herder_teleport 一致',
    )
    parser.add_argument(
        '--herder_physics_legacy',
        action='store_true',
        default=False,
        help='与 train_ppo --herder_physics_legacy 一致',
    )
    parser.add_argument(
        '--no-disk-boundary',
        action='store_true',
        default=False,
        help='与 train_ppo --no-disk-boundary 一致：羊/狗不受圆盘约束；坐标轴按 world_size 矩形',
    )
    parser.add_argument(
        '--herder_init',
        type=str,
        default=None,
        choices=['random_disk', 'fixed_arc'],
    )
    parser.add_argument(
        '--herder_assignment',
        type=str,
        default=None,
        choices=['min_cost', 'ordered'],
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
        default=FORMATION_DELTA_THETA_MAX_DEG,
    )
    parser.add_argument(
        '--formation_delta_radius_max',
        type=float,
        default=FORMATION_DELTA_RADIUS_MAX,
    )
    parser.add_argument(
        '--formation_delta_coverage_max',
        type=float,
        default=FORMATION_DELTA_COVERAGE_MAX,
    )
    parser.add_argument(
        '--high_level_interval',
        type=int,
        default=None,
        help='与 train_ppo 一致；默认 None 使用环境自动默认',
    )
    parser.add_argument(
        '--low-level-model-dir',
        type=str,
        default=None,
        help='指定后启用联合高低层：加载 InforMARL 低层 Graph MAPPO（目录含 actor.pt 或单个 model_*.pt），主步内子仿真驱动牧者',
    )
    parser.add_argument(
        '--low-level-substeps',
        type=int,
        default=None,
        help='每主环境步内低层子步数；默认 round(dt/0.1)，与 train_ppo 一致',
    )
    parser.add_argument(
        '--low-level-device',
        type=str,
        default='cpu',
        help='低层策略推理设备：cpu 或 cuda（可与高层可视化 device 不同）',
    )
    parser.add_argument(
        '--low-level-world-size',
        type=float,
        default=None,
        help='低层 MPE world_size；默认 min(W,H)',
    )
    parser.add_argument(
        '--low-level-num-obstacles',
        type=int,
        default=1,
        help='低层障碍数，应与低层训练一致',
    )
    parser.add_argument(
        '--low-level-max-speed',
        type=float,
        default=2.0,
        help='低层 max_speed，应与低层训练一致',
    )
    parser.add_argument(
        '--low-level-max-edge-dist',
        type=float,
        default=None,
        help='低层 GNN max_edge_dist；默认与 train_ppo 一致（未设时用环境默认 30）',
    )
    parser.add_argument(
        '--low-level-no-shepherd',
        action='store_true',
        default=False,
        help='关闭低层 use_shepherd_env（默认开启，与 InforMARL 训练脚本对齐）',
    )
    parser.add_argument(
        '--low-level-stochastic',
        action='store_true',
        default=False,
        help='低层策略按分布采样动作（默认确定性）',
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
        '--save-sheep-trajectory-dir',
        type=str,
        default=None,
        metavar='DIR',
        help='若指定目录，则每个 episode 结束后在该目录保存一张「全部羊轨迹」PNG（自动创建目录）。若以 /figures/ 开头会按仓库内 figures/ 解析',
    )
    parser.add_argument(
        '--save-visual-every',
        type=int,
        default=None,
        metavar='K',
        help='若为正整数，则每 K 个环境 step 将当前 matplotlib 界面保存为 PNG（需已 render）',
    )
    parser.add_argument(
        '--save-visual-dir',
        type=str,
        default=None,
        metavar='DIR',
        help='快照保存根目录；不设则使用仓库内 figures/viz_snapshots/。若以 /figures/ 开头会按仓库内 figures/ 解析（勿与根目录 /figures 混淆）',
    )
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


def _low_level_sheep_env_kwargs(args: argparse.Namespace) -> Dict[str, Any]:
    """与 train_ppo._extra_sheep_env_kwargs 中的低层字段一致；未传 --low-level-model-dir 时返回空字典。"""
    kw: Dict[str, Any] = {}
    mdir = getattr(args, "low_level_model_dir", None)
    if not mdir:
        return kw
    kw["low_level_model_dir"] = str(mdir)
    if getattr(args, "low_level_substeps", None) is not None:
        kw["low_level_substeps"] = int(args.low_level_substeps)
    kw["low_level_device"] = str(args.low_level_device)
    if getattr(args, "low_level_world_size", None) is not None:
        kw["low_level_world_size"] = float(args.low_level_world_size)
    kw["low_level_num_obstacles"] = int(args.low_level_num_obstacles)
    kw["low_level_max_speed"] = float(args.low_level_max_speed)
    if getattr(args, "low_level_max_edge_dist", None) is not None:
        kw["low_level_max_edge_dist"] = float(args.low_level_max_edge_dist)
    kw["low_level_use_shepherd"] = not bool(
        getattr(args, "low_level_no_shepherd", False)
    )
    kw["low_level_deterministic"] = not bool(
        getattr(args, "low_level_stochastic", False)
    )
    return kw


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
        self._sheep_traj_buffer: Optional[List[np.ndarray]] = None
        self._visual_snapshot_dir: Optional[str] = None

    def _init_visual_snapshots_if_needed(self) -> None:
        k = getattr(self.args, "save_visual_every", None)
        if k is None or int(k) <= 0:
            self._visual_snapshot_dir = None
            return

        ns = int(self.env.num_sheep)
        nh = int(self.env.num_herders)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        sub = f"sheep{ns}_herder{nh}_{stamp}"
        root = getattr(self.args, "save_visual_dir", None)
        if not root:
            parent = os.path.join(_REPO_ROOT, "figures", "viz_snapshots")
        else:
            parent = os.path.abspath(str(root).strip())
        self._visual_snapshot_dir = os.path.join(parent, sub)
        os.makedirs(self._visual_snapshot_dir, exist_ok=True)
        print(
            f"视觉快照: 每 {int(k)} 步保存 → {self._visual_snapshot_dir}",
            flush=True,
        )

    def _maybe_save_visual_snapshot(self, episode_num: int) -> None:
        k = getattr(self.args, "save_visual_every", None)
        if k is None or int(k) <= 0 or self._visual_snapshot_dir is None:
            return
        if self.fig is None:
            return
        if int(self.state.current_step) % int(k) != 0:
            return
        ns = int(self.env.num_sheep)
        nh = int(self.env.num_herders)
        name = (
            f"sheep{ns}_herder{nh}_ep{int(episode_num):02d}_"
            f"step{int(self.state.current_step):06d}.png"
        )
        path = os.path.join(self._visual_snapshot_dir, name)
        self.fig.savefig(path, dpi=150)

    def _arena_view(self):
        """返回 (xlim, ylim, 是否绘制圆盘边界)。取景按真实 world_size / world_radius；刻度读数见 _apply_axis_display_tick_labels。"""
        sc = self.env.scenario
        if getattr(sc, "enforce_disk_boundary", True):
            R = float(sc.world_radius)
            pad = max(R * 0.06, 1.0)
            return (-R - pad, R + pad), (-R - pad, R + pad), True
        Wx, Wy = float(self.env.world_size[0]), float(self.env.world_size[1])
        hx, hy = Wx / 2.0, Wy / 2.0
        pad = max(Wx, Wy) * 0.06
        return (-hx - pad, hx + pad), (-hy - pad, hy + pad), False

    def _axis_label_display_scales(self) -> Tuple[float, float]:
        """真实坐标 → 轴刻度显示值 的乘子；使场边界在刻度上约为 ±VIZ_AXIS_LABEL_DISPLAY_HALF。"""
        half = float(VIZ_AXIS_LABEL_DISPLAY_HALF)
        sc = self.env.scenario
        if getattr(sc, "enforce_disk_boundary", True):
            R = max(float(sc.world_radius), 1e-9)
            s = half / R
            return (s, s)
        Wx, Wy = float(self.env.world_size[0]), float(self.env.world_size[1])
        hx = max(Wx / 2.0, 1e-9)
        hy = max(Wy / 2.0, 1e-9)
        return (half / hx, half / hy)

    def _apply_axis_display_tick_labels(self, ax) -> None:
        sx, sy = self._axis_label_display_scales()
        ax.xaxis.set_major_formatter(
            FuncFormatter(lambda v, _p, sx=sx: f"{v * sx:g}")
        )
        ax.yaxis.set_major_formatter(
            FuncFormatter(lambda v, _p, sy=sy: f"{v * sy:g}")
        )

    def setup_figure(self):
        self.fig, self.ax = plt.subplots(figsize=(10, 10))
        (x0, x1), (y0, y1), draw_disk = self._arena_view()
        self.ax.set_xlim(x0, x1)
        self.ax.set_ylim(y0, y1)
        self.ax.set_aspect('equal')
        self.ax.set_facecolor('#f5f5f5')
        if draw_disk:
            R = float(self.env.scenario.world_radius)
            self.ax.add_patch(
                patches.Circle(
                    (0.0, 0.0), R, fill=False, edgecolor="0.5", linewidth=1.2, linestyle="--"
                )
            )
        else:
            Wx, Wy = float(self.env.world_size[0]), float(self.env.world_size[1])
            hx, hy = Wx / 2.0, Wy / 2.0
            self.ax.add_patch(
                patches.Rectangle(
                    (-hx, -hy),
                    Wx,
                    Wy,
                    fill=False,
                    edgecolor="0.65",
                    linewidth=1.0,
                    linestyle="--",
                )
            )
        self._apply_axis_display_tick_labels(self.ax)

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
        
        (x0, x1), (y0, y1), draw_disk = self._arena_view()
        self.ax.set_xlim(x0, x1)
        self.ax.set_ylim(y0, y1)
        self.ax.set_aspect('equal')
        self.ax.set_facecolor('#f5f5f5')

        R = float(self.env.scenario.world_radius)
        if draw_disk:
            arena = patches.Circle(
                (0.0, 0.0), R, fill=False, edgecolor="0.5", linewidth=1.2, linestyle="--"
            )
            self.ax.add_patch(arena)
        else:
            Wx, Wy = float(self.env.world_size[0]), float(self.env.world_size[1])
            hx, hy = Wx / 2.0, Wy / 2.0
            self.ax.add_patch(
                patches.Rectangle(
                    (-hx, -hy),
                    Wx,
                    Wy,
                    fill=False,
                    edgecolor="0.65",
                    linewidth=1.0,
                    linestyle="--",
                )
            )
        
        target = self.env.scenario.get_target_position()
        target_circle = patches.Circle(target, 5.0, color='green', alpha=0.3)
        self.ax.add_patch(target_circle)
        self.ax.plot(target[0], target[1], 'g*', markersize=20)

        flock_state = self.env.scenario.get_flock_state()
        sheep_positions = flock_state['positions']
        
        if len(sheep_positions) > 0:
            sheep_positions = np.array(sheep_positions)
            self.ax.scatter(sheep_positions[:, 0], sheep_positions[:, 1], 
                           c='gray', s=50, alpha=0.7, marker='o', edgecolors='black')
            
            flock_center = self.env.scenario.get_flock_center()
            self.ax.plot(flock_center[0], flock_center[1], 'k+', markersize=12, markeredgewidth=2)
        
        herder_positions = self.env.scenario.herder_positions.copy()
        
        if hasattr(self.env.scenario, "herder_targets"):
            herder_targets = self.env.scenario.herder_targets
            if herder_targets is not None:
                for i in range(self.env.num_herders):
                    target_pos_i = np.asarray(herder_targets[i], dtype=float).reshape(2)
                    # 暂时关闭：编队采样点金色圆（Sampled target）。恢复显示时取消下面整块注释。
                    # self.ax.add_patch(
                    #     patches.Circle(
                    #         (float(target_pos_i[0]), float(target_pos_i[1])),
                    #         FORMATION_SAMPLE_CIRCLE_RADIUS,
                    #         facecolor="gold",
                    #         edgecolor="darkorange",
                    #         linewidth=1.0,
                    #         alpha=FORMATION_SAMPLE_CIRCLE_ALPHA,
                    #         zorder=FORMATION_SAMPLE_CIRCLE_ZORDER,
                    #     )
                    # )
                    if i < len(herder_positions):
                        hpos = herder_positions[i]
                        direction = target_pos_i - hpos
                        dist = float(np.linalg.norm(direction))
                        if dist > 0.5:
                            direction = direction / dist
                            arrow_len = min(dist * 0.4, 3.0)
                            self.ax.annotate(
                                "",
                                xy=(
                                    hpos[0] + direction[0] * arrow_len,
                                    hpos[1] + direction[1] * arrow_len,
                                ),
                                xytext=hpos,
                                zorder=HERDER_ARROW_ZORDER,
                                arrowprops=dict(
                                    arrowstyle="->",
                                    color="green",
                                    lw=1.5,
                                    alpha=0.8,
                                    zorder=HERDER_ARROW_ZORDER,
                                ),
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
                zorder=EVASION_ZONE_ZORDER,
            )
            self.ax.add_patch(evasion_circle)
        
        self.ax.scatter(
            herder_positions[:, 0],
            herder_positions[:, 1],
            c="blue",
            s=80,
            alpha=0.9,
            marker="s",
            edgecolors="darkblue",
            label="Herder",
            zorder=HERDER_SCATTER_ZORDER,
        )
        
        legend_elements = [
            plt.Line2D([0], [0], marker='*', color='w', markerfacecolor='green', 
                      markersize=15, label='Target'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='gray',
                      markersize=10, markeredgecolor='black', label='Sheep'),
            plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='blue',
                      markersize=10, markeredgecolor='darkblue', label='Herder'),
            # 与上方金色圆一并关闭图例项；恢复圆时取消注释。
            # patches.Patch(
            #     facecolor="gold",
            #     alpha=FORMATION_SAMPLE_CIRCLE_ALPHA,
            #     edgecolor="darkorange",
            #     linewidth=1.0,
            #     label="Sampled target",
            # ),
            patches.Patch(facecolor='blue', alpha=0.08, edgecolor='blue',
                         linestyle='--', label='Evasion Zone'),
        ]
        self.ax.legend(handles=legend_elements, loc='upper right', fontsize=8)
        self.ax.grid(True, alpha=0.3)
        self._apply_axis_display_tick_labels(self.ax)

    def _sheep_positions_now(self) -> np.ndarray:
        flock = self.env.scenario.get_flock_state()
        pos = flock.get("positions")
        if pos is None or len(pos) == 0:
            return np.zeros((0, 2), dtype=np.float64)
        return np.asarray(pos, dtype=np.float64).reshape(-1, 2)

    @staticmethod
    def _smooth_xy_traj(xy: np.ndarray, half_window: Optional[int] = None) -> np.ndarray:
        """对 (T,2) 路径做各维独立的时间滑动平均；half_window 为单侧邻域点数，None 时按轨迹长度自适应。"""
        xy = np.asarray(xy, dtype=np.float64)
        t = int(xy.shape[0])
        if t < 3:
            return xy.copy()
        if half_window is None:
            hw = max(1, min(8, max(1, t // 8)))
        else:
            hw = max(1, int(half_window))
        hw = min(hw, t // 2)
        out = np.zeros_like(xy)
        for i in range(t):
            lo, hi = max(0, i - hw), min(t, i + hw + 1)
            out[i] = xy[lo:hi].mean(axis=0)
        return out

    @staticmethod
    def _allocate_sheep_trajectory_path(out_dir: str, model_stem: str, episode_num: int) -> str:
        """在 out_dir 下生成不覆盖已存在文件的 PNG 路径。"""
        base = f"sheep_trajectory_{model_stem}_ep{int(episode_num):03d}"
        path = os.path.join(out_dir, f"{base}.png")
        if not os.path.isfile(path):
            return path
        k = 1
        while True:
            path = os.path.join(out_dir, f"{base}_{k}.png")
            if not os.path.isfile(path):
                return path
            k += 1

    def _save_sheep_trajectory_figure(self, episode_num: int) -> None:
        out_dir = getattr(self.args, "save_sheep_trajectory_dir", None)
        if not out_dir or self._sheep_traj_buffer is None:
            return
        buf = self._sheep_traj_buffer
        if len(buf) == 0:
            print("  跳过羊轨迹图: 无采样点")
            return
        traj = np.stack(buf, axis=0)
        n_sheep = int(traj.shape[1])
        if n_sheep == 0:
            print("  跳过羊轨迹图: 当前无羊")
            return
        os.makedirs(out_dir, exist_ok=True)
        stem = os.path.splitext(os.path.basename(self.args.model_path))[0]
        path = self._allocate_sheep_trajectory_path(out_dir, stem, episode_num)

        fig_traj, ax_traj = plt.subplots(figsize=(10, 10))
        (x0, x1), (y0, y1), draw_disk = self._arena_view()
        ax_traj.set_xlim(x0, x1)
        ax_traj.set_ylim(y0, y1)
        ax_traj.set_aspect("equal")
        ax_traj.set_facecolor("#f5f5f5")
        R = float(self.env.scenario.world_radius)
        if draw_disk:
            ax_traj.add_patch(
                patches.Circle(
                    (0.0, 0.0),
                    R,
                    fill=False,
                    edgecolor="0.5",
                    linewidth=1.2,
                    linestyle="--",
                )
            )
        else:
            Wx, Wy = float(self.env.world_size[0]), float(self.env.world_size[1])
            hx, hy = Wx / 2.0, Wy / 2.0
            ax_traj.add_patch(
                patches.Rectangle(
                    (-hx, -hy),
                    Wx,
                    Wy,
                    fill=False,
                    edgecolor="0.65",
                    linewidth=1.0,
                    linestyle="--",
                )
            )
        target = self.env.scenario.get_target_position()
        ax_traj.add_patch(
            patches.Circle(
                (float(target[0]), float(target[1])),
                5.0,
                color="green",
                alpha=0.3,
                zorder=2,
            )
        )
        ax_traj.scatter(
            float(target[0]),
            float(target[1]),
            c="green",
            s=200,
            marker="*",
            zorder=8,
            edgecolors="darkgreen",
            linewidths=0.8,
            label="Target",
        )

        _tab = plt.get_cmap("tab10")
        # tab10 索引 3 为红色，留给质心专用
        sheep_colors = [_tab(j) for j in range(10) if j != 3]
        for i in range(n_sheep):
            color = sheep_colors[i % len(sheep_colors)]
            ax_traj.plot(
                traj[:, i, 0],
                traj[:, i, 1],
                "--",
                color=color,
                lw=1.6,
                alpha=0.88,
                label=f"Sheep {i}",
                zorder=3,
            )
            ax_traj.scatter(
                traj[0, i, 0],
                traj[0, i, 1],
                color=color,
                s=42,
                marker="o",
                edgecolors="black",
                linewidths=0.8,
                zorder=6,
            )
            ax_traj.scatter(
                traj[-1, i, 0],
                traj[-1, i, 1],
                color=color,
                s=48,
                marker="s",
                edgecolors="black",
                linewidths=0.8,
                zorder=7,
            )

        centroid = np.mean(traj, axis=1)
        centroid_s = self._smooth_xy_traj(centroid)
        ax_traj.plot(
            centroid_s[:, 0],
            centroid_s[:, 1],
            "-",
            color="crimson",
            lw=2.4,
            alpha=0.92,
            solid_capstyle="round",
            solid_joinstyle="round",
            label="Flock centroid (smoothed)",
            zorder=5,
        )
        # 沿质心轨迹箭头表示运动方向（随长度自适应数量）
        Tc = int(centroid_s.shape[0])
        if Tc >= 2:
            n_seg = min(14, max(4, Tc // 12))
            step = max(1, (Tc - 1) // n_seg)
            for i in range(0, Tc - 1, step):
                j = min(i + step, Tc - 1)
                if j <= i:
                    j = min(i + 1, Tc - 1)
                x1, y1 = float(centroid_s[i, 0]), float(centroid_s[i, 1])
                x2, y2 = float(centroid_s[j, 0]), float(centroid_s[j, 1])
                ax_traj.annotate(
                    "",
                    xy=(x2, y2),
                    xytext=(x1, y1),
                    arrowprops=dict(
                        arrowstyle="->",
                        color="crimson",
                        lw=2.2,
                        alpha=0.85,
                        shrinkA=0,
                        shrinkB=0,
                        mutation_scale=24,
                    ),
                    zorder=5,
                )
        ax_traj.scatter(
            centroid_s[0, 0],
            centroid_s[0, 1],
            c="crimson",
            s=58,
            marker="D",
            zorder=8,
            edgecolors="white",
            linewidths=0.9,
        )
        ax_traj.scatter(
            centroid_s[-1, 0],
            centroid_s[-1, 1],
            c="crimson",
            s=62,
            marker="P",
            zorder=8,
            edgecolors="white",
            linewidths=0.9,
        )

        ax_traj.grid(True, alpha=0.3)
        self._apply_axis_display_tick_labels(ax_traj)
        ax_traj.legend(loc="upper right", fontsize=8, ncol=2)
        fig_traj.tight_layout()
        fig_traj.savefig(path, dpi=150)
        plt.close(fig_traj)
        print(f"  羊轨迹图已保存: {path}")
        
    def run_episode(self, episode_num: int, total_episodes: int):
        print(f"\n{'='*50}")
        print(f"Episode {episode_num}/{total_episodes}")
        print(f"{'='*50}")
        
        self.reset_episode()
        self.state.episode_done = False
        if getattr(self.args, "save_sheep_trajectory_dir", None):
            os.makedirs(self.args.save_sheep_trajectory_dir, exist_ok=True)
            self._sheep_traj_buffer = [self._sheep_positions_now()]
        else:
            self._sheep_traj_buffer = None
        
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
            if self._sheep_traj_buffer is not None:
                self._sheep_traj_buffer.append(self._sheep_positions_now())

            self.state.current_step += 1
            self.state.total_reward += reward
            self.state.action_history.append(action.copy())

            self.render_env()
            self._maybe_save_visual_snapshot(episode_num)

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
        if getattr(self.args, "save_sheep_trajectory_dir", None):
            self._save_sheep_trajectory_figure(episode_num)
        
        return self.state.success, self.state.total_reward, self.state.current_step
    
    def run(self):
        self._init_visual_snapshots_if_needed()
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
    args.save_sheep_trajectory_dir = _rewrite_root_figures_typo_path(
        getattr(args, "save_sheep_trajectory_dir", None)
    )
    args.save_visual_dir = _rewrite_root_figures_typo_path(
        getattr(args, "save_visual_dir", None)
    )

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
    if getattr(args, "herder_physics_legacy", False):
        extra_kw["herder_physics_legacy"] = True
    if getattr(args, "no_disk_boundary", False):
        extra_kw["enforce_disk_boundary"] = False
    hm = {}
    if getattr(args, "herder_init", None):
        hm["herder_init_mode"] = str(args.herder_init)
    if getattr(args, "herder_assignment", None):
        hm["herder_slot_assignment"] = str(args.herder_assignment)
    if hm:
        extra_kw["herder_motion"] = hm
    extra_kw.update(_low_level_sheep_env_kwargs(args))
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
    if getattr(args, "low_level_model_dir", None):
        print(
            "联合高低层: 已启用 LowLevelMappoBridge（高层策略 + 低层 Graph MAPPO 子步）。"
            f" low_level_model_dir={args.low_level_model_dir}, "
            f"low_level_device={args.low_level_device}, "
            f"low_level_deterministic={not getattr(args, 'low_level_stochastic', False)}"
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
