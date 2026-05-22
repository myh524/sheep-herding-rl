#!/usr/bin/env python
"""
InforMARL 训练结果可视化入口（对齐仓库根目录 visualize.py 的用法与 Matplotlib 路径）。

实现要点（与根目录 visualize.py / onpolicy/utils/graph_mpe_matplotlib.py 一致）：
- 在 import torch 之前根据参数选择 Matplotlib 后端，避免 Agg 被过早锁定；
- 交互后端首帧后 show(block=False)，无 DISPLAY 时用 Agg + sleep；
- 实际绘图由 GraphMPE 的 MatplotlibGraphViewer（runner.render）完成。

默认开启 ``--render_with_matplotlib``（与根项目默认用 Matplotlib 看策略一致）。
若要用 pyglet 窗口，请加 ``--pyglet-render``。

典型用法::

    cd InforMARL
    python visualize.py --model_dir onpolicy/results/.../models \\
        --num_episodes 3 --render_delay 60

等价于调用 ``onpolicy/scripts/eval_mpe.py`` 并带上 ``--render_with_matplotlib`` 等标志。

与根目录 ``scripts/visualize.sh`` 一致：请勿在 shell 里默认 ``export MPLBACKEND=TkAgg``（未装
``python3-tk`` 时易退回 Agg）。后端选择与根目录 ``visualize.py`` 中
``_configure_matplotlib_backend`` 相同。

弹窗节奏与根目录 ``visualize.Visualizer`` 相同；若需在进程退出前阻塞直到关窗，可设
``INFORMARL_MPL_BLOCK_END=1``（见 ``graph_mpe_runner.render``）。
"""

from __future__ import annotations

import importlib.util
import os
import sys

_INFORMARL_ROOT = os.path.dirname(os.path.abspath(__file__))
if _INFORMARL_ROOT not in sys.path:
    sys.path.insert(0, _INFORMARL_ROOT)


def _normalize_argv_for_eval(argv: list[str]) -> list[str]:
    """
    将根目录 visualize.py 风格参数映射到 eval_mpe / get_config 的名称。
    """
    out: list[str] = []
    i = 0
    n = len(argv)
    have_render_episodes = any(
        x == "--render_episodes" or x.startswith("--render_episodes=") for x in argv
    )
    have_human_sleep = any(
        x == "--human_render_sleep" or x.startswith("--human_render_sleep=")
        for x in argv
    )
    skip_next = False

    while i < n:
        if skip_next:
            skip_next = False
            i += 1
            continue

        a = argv[i]

        if a == "--force_headless":
            out.append("--force_mpl_headless")
            i += 1
            continue

        if a == "--num_episodes":
            if not have_render_episodes and i + 1 < n:
                out.extend(["--render_episodes", argv[i + 1]])
                skip_next = True
            i += 1
            continue
        if a.startswith("--num_episodes="):
            if not have_render_episodes:
                out.append("--render_episodes=" + a.split("=", 1)[1])
            i += 1
            continue

        if a == "--render_delay":
            if not have_human_sleep and i + 1 < n:
                ms = float(argv[i + 1])
                out.extend(["--human_render_sleep", str(max(ms, 0.0) / 1000.0)])
                skip_next = True
            i += 1
            continue
        if a.startswith("--render_delay="):
            if not have_human_sleep:
                ms = float(a.split("=", 1)[1])
                out.append("--human_render_sleep=" + str(max(ms, 0.0) / 1000.0))
            i += 1
            continue

        if a == "--pyglet-render":
            # 仅作标记，不把字面量传给 eval（eval 不认识）
            i += 1
            continue

        out.append(a)
        i += 1

    use_pyglet = "--pyglet-render" in argv
    save_gifs = "--save_gifs" in argv
    if use_pyglet:
        out = [x for x in out if x != "--render_with_matplotlib"]
    elif not save_gifs and "--render_with_matplotlib" not in out:
        out.insert(0, "--render_with_matplotlib")

    return out


def _print_usage_hint():
    print(
        "InforMARL visualize.py\n"
        "  常用: python visualize.py --model_dir <训练输出 models 目录> [与 eval_mpe 相同的其它参数]\n"
        "  别名: --force_headless → --force_mpl_headless；"
        "--num_episodes → --render_episodes；"
        "--render_delay <毫秒> → --human_render_sleep\n"
        "  默认使用 Matplotlib 渲染；要 pyglet 请加 --pyglet-render。\n",
        flush=True,
    )


def main() -> None:
    if len(sys.argv) <= 1:
        _print_usage_hint()
        sys.exit(1)

    raw = sys.argv[1:]
    if raw in (["-h"], ["--help"]):
        _print_usage_hint()
        print(
            "完整参数列表: python onpolicy/scripts/eval_mpe.py --help",
            flush=True,
        )
        sys.exit(0)
    sys.argv = [sys.argv[0]] + _normalize_argv_for_eval(list(raw))

    eval_path = os.path.join(
        _INFORMARL_ROOT, "onpolicy", "scripts", "eval_mpe.py"
    )
    if not os.path.isfile(eval_path):
        raise FileNotFoundError(f"找不到 eval 脚本: {eval_path}")

    spec = importlib.util.spec_from_file_location(
        "informarl_eval_mpe_visualize", eval_path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载模块: {eval_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    if not hasattr(mod, "main"):
        raise RuntimeError("eval_mpe.py 中缺少 main()")

    mod.main(sys.argv[1:])


if __name__ == "__main__":
    main()
