"""
GraphMPE 的 matplotlib 可视化。

后端选择与弹窗节奏与仓库根目录 ``visualize.py`` 保持一致（同序逻辑，便于维护）：
``_gui_deps_available`` / ``configure_mpl_backend`` 对应 ``visualize._gui_deps_available`` /
``visualize._configure_matplotlib_backend``；``MatplotlibGraphViewer`` 对应 ``Visualizer`` 中
``setup_figure`` + 首帧 ``render_env`` 后 ``plt.show(block=False)`` + ``plt.pause`` 的流程。
"""

from __future__ import annotations

import importlib
import os
import sys
import time
import warnings
from typing import Optional, Tuple

import numpy as np

# -----------------------------------------------------------------------------
# 后端配置（必须在 import pyplot 之前完成；逻辑与 visualize.py 对齐）
# -----------------------------------------------------------------------------

_CONFIGURED = False
_BACKEND_NAME: Optional[str] = None


def _gui_deps_available(backend: str) -> bool:
    """与 visualize.py 相同：matplotlib.use 成功不等于能弹窗。"""
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


def configure_mpl_backend(force_headless: bool = False) -> str:
    """
    与仓库根目录 visualize.py 中 ``_configure_matplotlib_backend`` 相同逻辑；
    必须在任何 ``matplotlib.pyplot`` 导入之前调用。返回后端名称。
    """
    global _CONFIGURED, _BACKEND_NAME
    if _CONFIGURED:
        return _BACKEND_NAME or "Agg"

    import matplotlib

    if force_headless:
        matplotlib.use("Agg", force=True)
        _BACKEND_NAME = "Agg"
        _CONFIGURED = True
        return _BACKEND_NAME

    env_backend = os.environ.get("MPLBACKEND")
    if env_backend:
        try:
            matplotlib.use(env_backend, force=True)
            if env_backend.lower() != "agg" and not _gui_deps_available(env_backend):
                raise ImportError(f"backend {env_backend} dependencies missing")
            _BACKEND_NAME = env_backend
            _CONFIGURED = True
            return _BACKEND_NAME
        except Exception:
            print(
                f"警告: MPLBACKEND={env_backend} 不可用（例如 TkAgg 需安装 python3-tk），"
                "已回退到 Agg。要弹窗请: sudo apt install python3-tk 或安装 PyQt5 后使用 Qt5Agg。",
                file=sys.stderr,
            )
            matplotlib.use("Agg", force=True)
            _BACKEND_NAME = "Agg"
            _CONFIGURED = True
            return _BACKEND_NAME

    has_display = bool(
        os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")
    )
    if not has_display:
        matplotlib.use("Agg", force=True)
        _BACKEND_NAME = "Agg"
        _CONFIGURED = True
        return _BACKEND_NAME

    for name in ("TkAgg", "Qt5Agg", "QtAgg", "GTK3Agg"):
        try:
            matplotlib.use(name, force=True)
            if not _gui_deps_available(name):
                continue
            _BACKEND_NAME = name
            _CONFIGURED = True
            return _BACKEND_NAME
        except Exception:
            continue

    matplotlib.use("Agg", force=True)
    _BACKEND_NAME = "Agg"
    _CONFIGURED = True
    return _BACKEND_NAME


def is_interactive_backend() -> bool:
    """与 visualize.py 中 ``_matplotlib_is_interactive`` 相同。"""
    import matplotlib.pyplot as plt

    return plt.get_backend().lower() != "agg"


def _rgb(entity_color) -> Tuple[float, float, float]:
    if entity_color is None:
        return (0.45, 0.45, 0.45)
    c = np.asarray(entity_color, dtype=float).ravel()
    c = np.clip(c[:3], 0.0, 1.0)
    if c.size < 3:
        return (0.4, 0.7, 0.4)
    return (float(c[0]), float(c[1]), float(c[2]))


def draw_graph_navigation(ax, world, cam_range: float = 2.0, title: str = "") -> None:
    """从 world 重绘 ax（语义与 pyglet 一致；边 O(E)、实体用 scatter 以降低卡顿）。"""
    from matplotlib.collections import LineCollection
    from matplotlib.patches import Polygon

    ax.clear()
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-cam_range, cam_range)
    ax.set_ylim(-cam_range, cam_range)
    ax.set_axisbelow(True)
    ax.set_facecolor("#f5f5f5")
    ax.grid(True, alpha=0.18, linewidth=0.5)
    ax.tick_params(labelsize=8)

    for w in getattr(world, "walls", []) or []:
        corners = (
            (float(w.axis_pos - 0.5 * w.width), float(w.endpoints[0])),
            (float(w.axis_pos - 0.5 * w.width), float(w.endpoints[1])),
            (float(w.axis_pos + 0.5 * w.width), float(w.endpoints[1])),
            (float(w.axis_pos + 0.5 * w.width), float(w.endpoints[0])),
        )
        if w.orient == "H":
            corners = tuple((float(c[1]), float(c[0])) for c in corners)
        poly = Polygon(
            corners,
            closed=True,
            facecolor=_rgb(w.color),
            edgecolor="k",
            linewidth=0.5,
            alpha=0.85,
        )
        ax.add_patch(poly)

    if getattr(world, "graph_mode", False) and world.edge_list is not None:
        id2e = {e.global_id: e for e in world.entities}
        segs = []
        for col in world.edge_list.T:
            ga, gb = int(col[0]), int(col[1])
            ea, eb = id2e.get(ga), id2e.get(gb)
            if ea is None or eb is None:
                continue
            p0, p1 = ea.state.p_pos, eb.state.p_pos
            segs.append(
                [[float(p0[0]), float(p0[1])], [float(p1[0]), float(p1[1])]]
            )
        if segs:
            ax.add_collection(
                LineCollection(
                    segs,
                    colors=(0.35, 0.35, 0.35, 0.38),
                    linewidths=0.75,
                    zorder=1,
                )
            )

    ents = list(world.entities)
    if ents:
        p = np.stack([e.state.p_pos for e in ents], axis=0).astype(np.float32, copy=False)
        # scatter 的 s 为「点」上的面积（pt²）；适当放大，避免圆几乎看不见
        _pt_scale = 1000.0
        _s_min = 100.0
        s = np.array(
            [
                max(
                    _s_min,
                    max(float(getattr(e, "size", 0.05)), 0.02) ** 2 * _pt_scale,
                )
                for e in ents
            ],
            dtype=np.float32,
        )
        rgba = np.zeros((len(ents), 4), dtype=np.float32)
        for i, e in enumerate(ents):
            r, g, b = _rgb(e.color)
            rgba[i, 0] = r
            rgba[i, 1] = g
            rgba[i, 2] = b
            rgba[i, 3] = 0.55 if "agent" in e.name else 0.85
        ax.scatter(
            p[:, 0],
            p[:, 1],
            s=s,
            c=rgba,
            edgecolors="0.25",
            linewidths=0.55,
            zorder=2,
        )
    if title:
        ax.set_title(title, fontsize=10)


class MatplotlibGraphViewer:
    """
    与 visualize.Visualizer 相同的弹窗节奏（无 plt.ion；首帧绘制后 plt.show(block=False)；
    帧间 plt.pause / Agg 时 time.sleep）。仅绘图内容不同（GraphMPE / draw_graph_navigation）。
    """

    def __init__(
        self,
        cam_range: float = 2.0,
        figsize: Tuple[float, float] = (6.0, 6.0),
        autosave_path: Optional[str] = None,
        autosave_every: int = 20,
    ):
        # 若 eval 已在 import pyplot 前调用过 configure_mpl_backend，此处不再覆盖。
        if not _CONFIGURED:
            configure_mpl_backend(force_headless=False)

        import matplotlib.pyplot as plt

        self._plt = plt
        if plt.get_backend().lower() == "agg":
            warnings.filterwarnings(
                "ignore",
                message="FigureCanvasAgg is non-interactive",
                category=UserWarning,
            )

        # 与 visualize.Visualizer：self._interactive = _matplotlib_is_interactive()
        self._interactive = is_interactive_backend()
        self._gui_shown = False
        self.cam_range = float(cam_range)
        self._autosave_path = autosave_path
        self._autosave_every = max(0, int(autosave_every))
        self._autosave_hint_printed = False
        self._frame_idx = 0

        # 与 visualize.Visualizer.setup_figure：仅 subplots，此处不在空图上 show
        self.fig, self.ax = plt.subplots(figsize=figsize)
        try:
            mgr = self.fig.canvas.manager
            if mgr is not None and hasattr(mgr, "set_window_title"):
                mgr.set_window_title("GraphMPE (matplotlib)")
        except Exception:
            pass

    def update(self, world, title: str = "") -> None:
        self._frame_idx += 1
        draw_graph_navigation(self.ax, world, cam_range=self.cam_range, title=title)
        if self._interactive:
            self.fig.canvas.draw_idle()
        else:
            self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        # 与 visualize.Visualizer.run_episode：render_env 之后首次 plt.show(block=False)
        if self._interactive and not self._gui_shown:
            self._plt.show(block=False)
            self._gui_shown = True
        if (
            self._autosave_path
            and self._autosave_every > 0
            and (self._frame_idx % self._autosave_every == 0)
        ):
            if not self._autosave_hint_printed:
                print(
                    f"[GraphMPE matplotlib] 每 {self._autosave_every} 帧写入: "
                    f"{self._autosave_path}（可调 --mpl_autosave_every，0=不写盘）",
                    flush=True,
                )
                self._autosave_hint_printed = True
            try:
                d = os.path.dirname(self._autosave_path)
                if d:
                    os.makedirs(d, exist_ok=True)
                # 不用 bbox_inches=tight（每帧很慢）；dpi 适中减小 IO
                self.fig.savefig(self._autosave_path, dpi=80)
            except Exception as e:
                print(f"[GraphMPE matplotlib] savefig 失败: {e}", file=sys.stderr, flush=True)

    def pause(self, seconds: float) -> None:
        """与 visualize.Visualizer.run_episode 中 delay 分支一致。"""
        s = max(float(seconds), 1e-4)
        if self._interactive:
            self._plt.pause(s)
        else:
            time.sleep(s)

    def block_until_closed(self) -> None:
        """交互后端下阻塞直到用户关闭窗口（便于确认是否真能弹窗）。"""
        if not self._interactive:
            return
        print(
            "[GraphMPE matplotlib] 阻塞模式：关闭图形窗口后继续。",
            flush=True,
        )
        self._plt.show(block=True)

    def close(self) -> None:
        try:
            self._plt.close(self.fig)
        except Exception:
            pass
