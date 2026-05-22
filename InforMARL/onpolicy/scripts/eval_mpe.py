#!/usr/bin/env python
"""
Eval / render for MPE. When using --render_with_matplotlib, matplotlib backend must
be set before import torch (see visualize.py); torch may otherwise lock Agg early.
"""
import os
import sys

# InforMARL 根目录（不依赖当前工作目录，与从任意 cwd 调用兼容）
_EVAL_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_INFORMARL_ROOT = os.path.abspath(os.path.join(_EVAL_SCRIPT_DIR, "..", ".."))
for _p in (_INFORMARL_ROOT, os.path.abspath(os.getcwd())):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _configure_matplotlib_before_heavy_imports():
    """在 import torch 等重依赖之前配置 matplotlib 后端（对齐 visualize.py）。"""
    argv = sys.argv[1:]
    if "--render_with_matplotlib" not in argv:
        return
    if "matplotlib.pyplot" in sys.modules:
        print(
            "[GraphMPE eval] 警告: matplotlib.pyplot 在配置后端之前已被其它模块导入，"
            "GUI 后端可能无法切换，窗口常出不来。请避免在环境里预装会抢先 import pyplot 的包。",
            flush=True,
        )
    fh = "--force_mpl_headless" in argv
    from onpolicy.utils.graph_mpe_matplotlib import (
        configure_mpl_backend,
        is_interactive_backend,
    )

    b = configure_mpl_backend(force_headless=fh)
    inter = is_interactive_backend()
    print(
        f"[GraphMPE eval] matplotlib backend={b} "
        f"({'interactive' if inter else 'Agg/无窗口'})",
        flush=True,
    )
    if not inter and not fh:
        print(
            "提示: 当前为 Agg 或无交互后端。要弹窗请: sudo apt install python3-tk "
            "或安装 PyQt5 后设置 MPLBACKEND=Qt5Agg，并在有 DISPLAY/WAYLAND 的会话中运行；"
            "无桌面可加 --force_mpl_headless 仅用 sleep 推进仿真。",
            flush=True,
        )


_configure_matplotlib_before_heavy_imports()

import argparse
from distutils.util import strtobool
from typing import Dict

import numpy as np
from pathlib import Path
import torch

from utils.utils import print_args, print_box
from onpolicy.config import get_config
from multiagent.MPE_env import MPEEnv, GraphMPEEnv
from onpolicy.envs.env_wrappers import (
    SubprocVecEnv,
    DummyVecEnv,
    GraphSubprocVecEnv,
    GraphDummyVecEnv,
)


def make_render_env(all_args: argparse.Namespace):
    def get_env_fn(rank: int):
        def init_env():
            if all_args.env_name == "MPE":
                env = MPEEnv(all_args)
            elif all_args.env_name == "GraphMPE":
                env = GraphMPEEnv(all_args)
            else:
                print(f"Can not support the {all_args.env_name} environment.")
                raise NotImplementedError
            env.seed(all_args.seed + rank * 1000)
            return env

        return init_env

    if all_args.n_rollout_threads == 1:
        if all_args.env_name == "GraphMPE":
            return GraphDummyVecEnv([get_env_fn(0)])
        return DummyVecEnv([get_env_fn(0)])
    else:
        if all_args.env_name == "GraphMPE":
            return GraphSubprocVecEnv(
                [get_env_fn(i) for i in range(all_args.n_rollout_threads)]
            )
        return SubprocVecEnv([get_env_fn(i) for i in range(all_args.n_rollout_threads)])


def parse_args(args, parser):
    parser.add_argument(
        "--scenario_name",
        type=str,
        default="simple_spread",
        help="Which scenario to run on",
    )
    parser.add_argument("--num_landmarks", type=int, default=3)
    parser.add_argument("--num_agents", type=int, default=2, help="number of players")
    parser.add_argument(
        "--num_obstacles", type=int, default=3, help="Number of obstacles"
    )
    parser.add_argument(
        "--collaborative",
        type=lambda x: bool(strtobool(x)),
        default=True,
        help="Number of agents in the env",
    )
    parser.add_argument(
        "--max_speed",
        type=float,
        default=2,
        help="Max speed for agents. NOTE that if this is None, "
        "then max_speed is 2 with discrete action space",
    )
    parser.add_argument(
        "--collision_rew",
        type=float,
        default=5,
        help="The reward to be negated for collisions with other "
        "agents and obstacles",
    )
    parser.add_argument(
        "--goal_rew",
        type=float,
        default=5,
        help="The reward to be added if agent reaches the goal",
    )
    parser.add_argument(
        "--min_dist_thresh",
        type=float,
        default=0.05,
        help="The minimum distance threshold to classify whether "
        "agent has reached the goal or not",
    )
    parser.add_argument(
        "--use_dones",
        type=lambda x: bool(strtobool(x)),
        default=False,
        help="Whether we want to use the 'done=True' "
        "when agent has reached the goal or just return False like "
        "the `simple.py` or `simple_spread.py`",
    )

    all_args = parser.parse_known_args(args)[0]

    return all_args, parser


def modify_args(
    model_dir: str,
    args: argparse.Namespace,
    exclude_args: list = [
        "model_dir",
        "num_agents",
        "num_obstacles",
        "num_landmarks",
        "render_episodes",
        "world_size",
        "seed",
        # Do not let training yaml force GIF-only mode (no pyglet window).
        "save_gifs",
        # Eval-only pacing for human window (override without regenerating yaml).
        "human_render_sleep",
        "render_with_matplotlib",
        "force_mpl_headless",
        "mpl_autosave_every",
        "mpl_figsize",
    ],
):
    """
    Modify the args used to train the model
    """
    import yaml

    config_path = os.path.join(str(model_dir), "config.yaml")
    if os.path.isfile(config_path):
        with open(config_path, encoding="utf-8") as f:
            ydict = yaml.safe_load(f)
        if not ydict:
            raise ValueError(f"{config_path} is empty or invalid YAML.")

        print("_" * 50)
        for k, v in ydict.items():
            if k in exclude_args:
                print(f"Using {k} = {vars(args)[k]}")
                continue
            if type(v) == dict and "value" in v.keys():
                setattr(args, k, v["value"])
        print("_" * 50)
    else:
        print(
            "_" * 50
            + "\nWARNING: config.yaml not found under model_dir; "
            "using CLI defaults + your flags only (must match training). "
            "Re-save checkpoints after training with an updated codebase, or run:\n"
            "  python onpolicy/scripts/write_eval_config_yaml.py --dump_to <model_dir> "
            "<same flags as train_mpe.py>\n"
            + "_" * 50
        )

    # set some args manually
    args.cuda = False
    args.use_wandb = False
    args.use_render = True
    # save_gifs: excluded from yaml above; only --save_gifs enables GIF path.
    args.n_rollout_threads = 1

    return args


def main(args):
    # model_dir = 'trained_models/navigation/Navigation/rmappo/wandb/offline-run-20210720_220614-1eqhk4l1/files'
    # matplotlib 后端已在模块加载阶段、import torch 之前配置（见文件顶部）。
    parser = get_config()
    all_args, parser = parse_args(args, parser)
    if all_args.env_name == "GraphMPE":
        from onpolicy.config import graph_config

        all_args, parser = graph_config(args, parser)
    all_args = modify_args(all_args.model_dir, all_args)

    disp = os.environ.get("DISPLAY") or ""
    way = os.environ.get("WAYLAND_DISPLAY") or ""
    if not all_args.save_gifs and not (disp or way):
        print(
            "WARNING: DISPLAY / WAYLAND_DISPLAY is unset; pyglet usually cannot show "
            "a window here. Use a desktop terminal, `ssh -X`/VNC/WSLg, or pass "
            "`--save_gifs` to write render.gif under model_dir."
        )
    elif (
        not all_args.save_gifs
        and os.environ.get("SSH_CONNECTION")
        and disp in (":0", ":0.0")
        and not way
    ):
        # Very common: SSH to a Linux box that has a local X server on :0; the
        # window is created on that server's console, not forwarded to your laptop.
        print(
            "[InforMARL eval] 检测到 SSH 且 DISPLAY 为 :0 —— pyglet 窗口会画在"
            "「这台 Linux 机器接的显示器 / 本地图形会话」上，不会出现在你当前"
            "SSH 终端所在的电脑上。若你只在远程终端里跑，就会感觉「完全不显示」。"
            "可选方案：在本机对服务器使用 `ssh -X`/`ssh -Y`（并安装客户端 X 服务"
            "器）、用 VNC/远程桌面登录该机器再运行、或在 eval 命令上加 `--save_gifs`"
            "生成 model_dir/render.gif 再拷回本地查看。"
        )

    if all_args.algorithm_name == "rmappo" or all_args.algorithm_name == "rmappg":
        assert (
            all_args.use_recurrent_policy or all_args.use_naive_recurrent_policy
        ), "check recurrent policy!"
    elif all_args.algorithm_name == "mappo" or all_args.algorithm_name == "mappg":
        assert (
            all_args.use_recurrent_policy and all_args.use_naive_recurrent_policy
        ) == False, "check recurrent policy!"
    else:
        raise NotImplementedError

    assert all_args.use_render, "Need to set use_render be True"
    assert not (
        all_args.model_dir == None or all_args.model_dir == ""
    ), "set model_dir first"
    assert all_args.n_rollout_threads == 1, "only support to use 1 env to render."

    device = torch.device("cpu")

    # run dir
    # run_dir = Path(os.path.split(os.path.dirname(os.path.abspath(__file__)))[0]
    # if not run_dir.exists():
    #     os.makedirs(str(run_dir))

    # seed
    torch.manual_seed(all_args.seed)
    np.random.seed(all_args.seed)

    # env init
    envs = make_render_env(all_args)
    eval_envs = None
    num_agents = all_args.num_agents
    run_dir = None

    config = {
        "all_args": all_args,
        "envs": envs,
        "eval_envs": eval_envs,
        "num_agents": num_agents,
        "device": device,
        "run_dir": run_dir,
    }

    # run experiments
    if all_args.share_policy:
        if all_args.env_name == "GraphMPE":
            from onpolicy.runner.shared.graph_mpe_runner import GMPERunner as Runner
        else:
            from onpolicy.runner.shared.mpe_runner import MPERunner as Runner
    else:
        if all_args.env_name == "GraphMPE":
            raise NotImplementedError
        from onpolicy.runner.separated.mpe_runner import MPERunner as Runner

    # print_args(config['all_args'])

    runner = Runner(config)
    # actor_state_dict = torch.load(str(model_dir) + '/actor.pt')
    # runner.policy.actor.load_state_dict(actor_state_dict)
    # get_metrics=True 会跳过所有画面（含 matplotlib）；可视化必须传 False。
    runner.render(False)

    # post process
    envs.close()


if __name__ == "__main__":
    main(sys.argv[1:])
