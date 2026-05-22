"""
将 InforMARL（Graph MAPPO）作为高层 SheepFlockEnv 的牧者动力学子步。

InforMARL 经主仓库小幅 patch（``World.external_goals`` / ``Scenario.apply_external_state`` /
``GraphMPEEnv().scenario`` / config 新增两开关）后，与原 ``mappo_layered_shepherd`` 接口等价。

低层局部观测（``navigation_graph.observation``）：6 维基础量 + 每个障碍 2 维相对位置；
``num_obstacles=1`` 且牧羊模式时为 8 维。旧 ``actor.pt``（6 维 obs）需重新训练。

``agent_id`` 仍由环境/buffer 提供，仅在策略层用于 GNN gather 下标，不作为 Actor MLP 输入。

sys.path：把 InforMARL 置于首位，避免与仓库根目录的 onpolicy 包冲突。
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
_MAPPO_ROOT = _REPO_ROOT / "InforMARL"


def _ensure_mappo_on_path() -> None:
    p = str(_MAPPO_ROOT)
    if p not in sys.path:
        sys.path.insert(0, p)


def default_low_level_substeps(high_dt: float, low_dt: float = 0.1) -> int:
    return max(1, int(round(float(high_dt) / float(low_dt))))


def default_low_level_max_edge_dist(_world_size: float) -> float:
    """大场地（如 ``world_size≈100``）上 GNN 连边半径的宿主默认；与训练不一致时请显式传入 ``low_level_max_edge_dist``。"""
    return 30.0


def _parse_train_mpe_env_argv(argv: List[str], parser: Any) -> Tuple[Any, Any]:
    """
    与 onpolicy/scripts/train_mpe.py 中 parse_args 等价：注册 MPE 场景专用参数并解析 argv。
    必须在 graph_config 之前调用，否则 num_agents / scenario_name 等会被 parse_known_args 丢弃。
    """
    from distutils.util import strtobool

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

    all_args = parser.parse_known_args(argv)[0]
    return all_args, parser


def build_graph_mpe_args(
    num_agents: int,
    world_size: float,
    *,
    num_obstacles: int = 1,
    max_speed: float = 2.0,
    max_edge_dist: float = 30.0,
    graph_feat_type: str = "relative",
    use_shepherd_env: bool = True,
    external_goals: bool = True,
    episode_length: int = 10_000,
    seed: int = 0,
    device: str = "cpu",
) -> Any:
    """构造与 train.sh / GraphMPE 训练一致的 argparse.Namespace（推理用）。"""
    import torch

    _ensure_mappo_on_path()
    from onpolicy.config import get_config, graph_config

    argv: List[str] = [
        "--env_name",
        "GraphMPE",
        "--algorithm_name",
        "rmappo",
        "--scenario_name",
        "navigation_graph",
        "--world_size",
        str(float(world_size)),
        "--num_agents",
        str(int(num_agents)),
        "--num_obstacles",
        str(int(num_obstacles)),
        "--max_speed",
        str(float(max_speed)),
        "--graph_feat_type",
        str(graph_feat_type),
        "--max_edge_dist",
        str(float(max_edge_dist)),
        "--episode_length",
        str(int(episode_length)),
        "--n_rollout_threads",
        "1",
        "--n_eval_rollout_threads",
        "1",
        "--n_render_rollout_threads",
        "1",
        "--num_env_steps",
        "1",
        "--seed",
        str(int(seed)),
        "--collision_rew",
        "5",
        "--goal_rew",
        "5",
        "--min_dist_thresh",
        "0.05",
        "--use_dones",
        "False",
        "--collaborative",
        "True",
    ]
    if use_shepherd_env:
        argv.append("--use_shepherd_env")
    if external_goals:
        argv.append("--external_goals")

    parser = get_config()
    all_args, parser = _parse_train_mpe_env_argv(argv, parser)
    all_args, _parser = graph_config(argv, parser)
    all_args.cuda = device.startswith("cuda") and torch.cuda.is_available()
    all_args.external_goals = bool(external_goals)
    all_args.use_shepherd_env = bool(use_shepherd_env)
    all_args.max_edge_dist = float(max_edge_dist)
    # train_mpe.parse_args 中的场景项不在 get_config 里，补默认以免缺属性
    _scenario_defaults: Dict[str, Any] = {
        "collision_rew": 5.0,
        "goal_rew": 5.0,
        "min_dist_thresh": 0.05,
        "use_dones": False,
        "collaborative": True,
    }
    for _k, _v in _scenario_defaults.items():
        if not hasattr(all_args, _k):
            setattr(all_args, _k, _v)
    return all_args


def _resolve_actor_state_dict(model_dir: Path) -> Dict[str, Any]:
    """支持 .../models/actor.pt 或 .../checkpoints/model_*.pt（含 actor_state_dict）。"""
    import torch

    model_dir = model_dir.resolve()
    if model_dir.is_file() and model_dir.suffix == ".pt":
        ckpt = torch.load(str(model_dir), map_location="cpu")
        if isinstance(ckpt, dict) and "actor_state_dict" in ckpt:
            return ckpt["actor_state_dict"]
        raise ValueError(f"无法从 {model_dir} 解析 actor_state_dict")
    actor_pt = model_dir / "actor.pt"
    if actor_pt.is_file():
        return torch.load(str(actor_pt), map_location="cpu")
    raise FileNotFoundError(f"未找到 actor 权重：{model_dir}（期望 actor.pt 或 model_*.pt）")


class LowLevelMappoBridge:
    """
    持有一个 GraphMPEEnv + GR_MAPPOPolicy，在每个高层步内执行若干低层离散动作子步，
    并把 agent 位置写回 SheepScenario.herder_positions。
    """

    def __init__(
        self,
        model_dir: str,
        num_herders: int,
        world_size: float,
        *,
        num_obstacles: int = 1,
        max_speed: float = 2.0,
        max_edge_dist: float = 30.0,
        graph_feat_type: str = "relative",
        use_shepherd_env: bool = True,
        device: str = "cpu",
        seed: int = 0,
    ) -> None:
        import torch

        _ensure_mappo_on_path()
        self.num_agents = int(num_herders)
        self.world_size = float(world_size)
        self.device = torch.device(
            device if device.startswith("cuda") and torch.cuda.is_available() else "cpu"
        )
        self._all_args = build_graph_mpe_args(
            self.num_agents,
            self.world_size,
            num_obstacles=num_obstacles,
            max_speed=max_speed,
            max_edge_dist=max_edge_dist,
            graph_feat_type=graph_feat_type,
            use_shepherd_env=use_shepherd_env,
            external_goals=True,
            seed=seed,
            device=str(self.device),
        )
        self._all_args.model_dir = None

        from multiagent.MPE_env import GraphMPEEnv
        from onpolicy.algorithms.graph_MAPPOPolicy import GR_MAPPOPolicy

        self.env = GraphMPEEnv(self._all_args)
        self.env.seed(int(seed))
        self.env.reset()

        share_obs_space = (
            self.env.share_observation_space[0]
            if self._all_args.use_centralized_V
            else self.env.observation_space[0]
        )
        self.policy = GR_MAPPOPolicy(
            self._all_args,
            self.env.observation_space[0],
            share_obs_space,
            self.env.node_observation_space[0],
            self.env.edge_observation_space[0],
            self.env.action_space[0],
            device=self.device,
        )
        sd = _resolve_actor_state_dict(Path(model_dir))
        self.policy.actor.load_state_dict(sd)
        self.policy.actor.eval()

        self._recurrent_N = int(self._all_args.recurrent_N)
        self._hidden_size = int(self._all_args.hidden_size)
        self._use_cent = bool(self._all_args.use_centralized_V)
        self._low_dt = float(self.env.world.dt)
        self._rnn_actor: Optional[np.ndarray] = None
        self._rnn_critic: Optional[np.ndarray] = None

    def reset_rnn(self) -> None:
        """主环境 episode reset 时清空低层 RNN 状态。"""
        self._rnn_actor = None
        self._rnn_critic = None

    def _pack_batch(
        self,
        obs: np.ndarray,
        agent_id: np.ndarray,
        node_obs: np.ndarray,
        adj: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """(1, n, *) -> concatenate 第一维得到 (n, *)，及 share_obs / share_agent_id。"""
        n_rollout = 1
        n = self.num_agents
        if self._use_cent:
            share_obs = obs.reshape(n_rollout, -1)
            share_obs = np.expand_dims(share_obs, 1).repeat(n, axis=1)
            share_agent_id = agent_id.reshape(n_rollout, -1)
            share_agent_id = np.expand_dims(share_agent_id, 1).repeat(n, axis=1)
        else:
            share_obs = obs
            share_agent_id = agent_id

        cent = np.concatenate(share_obs, axis=0)
        ob = np.concatenate(obs, axis=0)
        node = np.concatenate(node_obs, axis=0)
        ad = np.concatenate(adj, axis=0)
        aid = np.concatenate(agent_id, axis=0)
        said = np.concatenate(share_agent_id, axis=0)
        return cent, ob, node, ad, aid, said

    def _get_obs_batch(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        obs_l: List[np.ndarray] = []
        aid_l: List[np.ndarray] = []
        node_l: List[np.ndarray] = []
        adj_l: List[np.ndarray] = []
        for agent in self.env.agents:
            obs_l.append(self.env._get_obs(agent))
            aid_l.append(self.env._get_id(agent))
            node, adj = self.env._get_graph_obs(agent)
            node_l.append(node)
            adj_l.append(adj)
        obs = np.stack(obs_l, axis=0)[None, ...]
        agent_id = np.stack(aid_l, axis=0)[None, ...]
        node_obs = np.stack(node_l, axis=0)[None, ...]
        adj = np.stack(adj_l, axis=0)[None, ...]
        return obs, agent_id, node_obs, adj

    def _actions_tensor_to_env(self, actions: Any) -> List[np.ndarray]:
        """与 graph_mpe_runner.collect 中 Discrete 分支一致。"""
        act_space = self.env.action_space[0]
        acts = np.asarray(actions.detach().cpu().numpy(), dtype=np.int64).reshape(-1)
        if act_space.__class__.__name__ != "Discrete":
            raise NotImplementedError(
                f"低层桥接当前仅支持 Discrete 动作，当前为 {act_space.__class__.__name__}"
            )
        n = act_space.n
        one_hot = np.eye(n, dtype=np.float32)[acts]
        return [one_hot[i] for i in range(one_hot.shape[0])]

    def run_substeps(
        self,
        herder_positions: np.ndarray,
        herder_targets: np.ndarray,
        flock_center: np.ndarray,
        num_substeps: int,
        *,
        deterministic: bool = True,
    ) -> np.ndarray:
        """
        Args:
            herder_positions: (N, 2) 主环境米制坐标
            herder_targets: (N, 2) set_herder_targets 之后每 agent 的目标（已分配槽位）
            flock_center: (2,) 羊群质心，写入 obstacle[0]（牧羊模式）
            num_substeps: 低层步数
        """
        import torch

        scenario = self.env.scenario
        world = self.env.world
        world.current_time_step = 0
        self.env.current_step = 0

        sheep = (
            np.asarray(flock_center, dtype=np.float64).reshape(2)
            if world.use_shepherd_env and len(world.obstacles) > 0
            else None
        )
        scenario.apply_external_state(
            world,
            np.asarray(herder_positions, dtype=np.float64),
            np.asarray(herder_targets, dtype=np.float64),
            sheep_position=sheep,
        )

        if self._rnn_actor is None:
            self._rnn_actor = np.zeros(
                (1, self.num_agents, self._recurrent_N, self._hidden_size),
                dtype=np.float32,
            )
            self._rnn_critic = np.zeros_like(self._rnn_actor)
        rnn_a = self._rnn_actor
        rnn_c = self._rnn_critic
        masks = np.ones((1, self.num_agents, 1), dtype=np.float32)

        with torch.no_grad():
            for _ in range(int(num_substeps)):
                obs, agent_id, node_obs, adj = self._get_obs_batch()
                cent, ob, node, ad, aid, said = self._pack_batch(
                    obs, agent_id, node_obs, adj
                )
                rnn_a_flat = np.concatenate(rnn_a, axis=0)
                rnn_c_flat = np.concatenate(rnn_c, axis=0)
                m_flat = np.concatenate(masks, axis=0)

                _v, actions, _lp, rnn_a_next, rnn_c_next = self.policy.get_actions(
                    cent,
                    ob,
                    node,
                    ad,
                    aid,
                    said,
                    rnn_a_flat,
                    rnn_c_flat,
                    m_flat,
                    deterministic=deterministic,
                )
                rnn_a[:] = rnn_a_next.detach().cpu().numpy().reshape(
                    1, self.num_agents, self._recurrent_N, self._hidden_size
                )
                rnn_c[:] = rnn_c_next.detach().cpu().numpy().reshape(
                    1, self.num_agents, self._recurrent_N, self._hidden_size
                )

                act_list = self._actions_tensor_to_env(actions)
                self.env.step(act_list)

        return np.stack(
            [a.state.p_pos.copy() for a in world.agents], axis=0
        ).astype(np.float32)
