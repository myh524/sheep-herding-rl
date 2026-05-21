"""为 matplotlib 配置可用的中文字体（按字体文件路径加载，避免方框）。"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from matplotlib import font_manager

# 由 setup_matplotlib_chinese 填充；图例等建议用 zh_font() 显式指定 fname
_FONT_PATH: Optional[str] = None
_FONT_FAMILY: Optional[str] = None

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 常见系统字体路径（Linux / WSL 挂载 Windows / macOS / 仓库内可选字体）
_CANDIDATE_FONT_FILES = (
    os.path.join(_REPO_ROOT, "assets", "fonts", "NotoSansSC-Regular.otf"),
    os.path.join(_REPO_ROOT, "assets", "fonts", "wqy-microhei.ttc"),
    "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJKsc-Regular.otf",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansSC-Regular.ttf",
    "/usr/share/fonts/opentype/noto/NotoSansSC-Regular.otf",
    "/mnt/c/Windows/Fonts/msyh.ttc",
    "/mnt/c/Windows/Fonts/msyhbd.ttc",
    "/mnt/c/Windows/Fonts/simhei.ttf",
    "C:/Windows/Fonts/msyh.ttc",
    "C:/Windows/Fonts/simhei.ttf",
    "/System/Library/Fonts/PingFang.ttc",
    "/System/Library/Fonts/STHeiti Light.ttc",
)

_NAME_KEYWORDS = (
    "WenQuanYi",
    "Noto Sans CJK",
    "Noto Sans SC",
    "Source Han",
    "SimHei",
    "Microsoft YaHei",
    "PingFang",
    "Heiti",
    "STSong",
)


def _resolve_font_path() -> Optional[str]:
    for path in _CANDIDATE_FONT_FILES:
        if os.path.isfile(path):
            return path
    for entry in font_manager.fontManager.ttflist:
        name = entry.name or ""
        if any(k in name for k in _NAME_KEYWORDS):
            return entry.fname
    try:
        for path in font_manager.findSystemFonts(fontext="ttf"):
            low = path.lower()
            if any(k in low for k in ("wqy", "noto", "cjk", "simhei", "yahei", "pingfang")):
                return path
        for path in font_manager.findSystemFonts(fontext="otf"):
            low = path.lower()
            if any(k in low for k in ("noto", "cjk", "sourcehan")):
                return path
    except Exception:
        pass
    return None


def setup_matplotlib_chinese(extra_rc: Optional[Dict] = None) -> str:
    """注册并启用中文字体，返回 matplotlib 使用的字体族名。"""
    global _FONT_PATH, _FONT_FAMILY

    font_path = _resolve_font_path()
    if font_path is None:
        raise RuntimeError(
            "未找到中文字体。Ubuntu/WSL 可安装: sudo apt install fonts-wqy-microhei "
            "或 fonts-noto-cjk"
        )

    _FONT_PATH = font_path
    try:
        font_manager.fontManager.addfont(font_path)
    except Exception:
        pass

    prop = font_manager.FontProperties(fname=font_path)
    _FONT_FAMILY = prop.get_name()

    import matplotlib.pyplot as plt

    existing = [
        f
        for f in plt.rcParams.get("font.sans-serif", [])
        if f and f != _FONT_FAMILY
    ]
    sans = [_FONT_FAMILY, *existing]

    rc: Dict = {
        "font.family": "sans-serif",
        "font.sans-serif": sans,
        "axes.unicode_minus": False,
    }
    if extra_rc:
        rc.update(extra_rc)
    plt.rcParams.update(rc)
    return _FONT_FAMILY


def zh_font(**kwargs) -> font_manager.FontProperties:
    """返回基于字体文件路径的 FontProperties（图例/文字最稳妥，避免缺字方框）。"""
    if _FONT_PATH is None:
        setup_matplotlib_chinese()
    return font_manager.FontProperties(fname=_FONT_PATH, **kwargs)


# 与 visualize.py 主视图 / formation_sliders 右上角图例一致
VIZ_LEGEND_FONTSIZE = 22
VIZ_LEGEND_MARKER_SIZE_TARGET = 30
VIZ_LEGEND_MARKER_SIZE_DEFAULT = 20
VIZ_LEGEND_HANDLEHEIGHT = 1.4
VIZ_LEGEND_HANDLELENGTH = 4.0
VIZ_LEGEND_MARKERSCALE = 2.0

# 羊轨迹 PNG（条目多、双列）
TRAJECTORY_LEGEND_FONTSIZE = VIZ_LEGEND_FONTSIZE
TRAJECTORY_LEGEND_MARKERSCALE = VIZ_LEGEND_MARKERSCALE
TRAJECTORY_LEGEND_HANDLEHEIGHT = VIZ_LEGEND_HANDLEHEIGHT
TRAJECTORY_LEGEND_HANDLELENGTH = VIZ_LEGEND_HANDLELENGTH


def viz_panel_legend(ax, handles: List[Any], *, ncol: int = 1, loc: str = "upper right", **overrides: Any):
    """交互/主视图固定样式图例（visualize.py、formation_sliders.py）。"""
    kw: Dict[str, Any] = {
        "handles": handles,
        "loc": loc,
        "ncol": ncol,
        "prop": zh_font(size=VIZ_LEGEND_FONTSIZE),
        "handleheight": VIZ_LEGEND_HANDLEHEIGHT,
        "handlelength": VIZ_LEGEND_HANDLELENGTH,
    }
    kw.update(overrides)
    return ax.legend(**kw)


def sheep_trajectory_legend(ax, **overrides: Any):
    """羊轨迹 PNG 图例（visualize.sh / 合成轨迹脚本共用）。"""
    kw: Dict[str, Any] = {
        "loc": "upper right",
        "prop": zh_font(size=TRAJECTORY_LEGEND_FONTSIZE),
        "ncol": 2,
        "markerscale": TRAJECTORY_LEGEND_MARKERSCALE,
        "handleheight": TRAJECTORY_LEGEND_HANDLEHEIGHT,
        "handlelength": TRAJECTORY_LEGEND_HANDLELENGTH,
    }
    kw.update(overrides)
    return ax.legend(**kw)
