#!/usr/bin/env python
"""
Write config.yaml into a model directory for eval_mpe.py.

Use the same CLI flags as train_mpe.py for that run (omit --dump_to).

From InforMARL repo root, example:

  python onpolicy/scripts/write_eval_config_yaml.py --dump_to \\
    onpolicy/results/GraphMPE/navigation_graph/rmappo/informarl/run2/models \\
    --use_valuenorm --use_popart --project_name informarl --env_name GraphMPE \\
    --algorithm_name rmappo --seed 0 --experiment_name informarl \\
    --scenario_name navigation_graph --num_agents 3 --collision_rew 5 \\
    --n_training_threads 1 --n_rollout_threads 128 --num_mini_batch 1 \\
    --episode_length 25 --num_env_steps 2000000 \\
    --ppo_epoch 10 --use_ReLU --gain 0.01 --lr 7e-4 --critic_lr 7e-4 \\
    --user_name marl --use_cent_obs False --graph_feat_type relative \\
    --auto_mini_batch_size --target_mini_batch_size 128 --use_wandb
"""

from __future__ import annotations

import os
import sys

sys.path.append(os.path.abspath(os.getcwd()))

from onpolicy.config import get_config, graph_config
from onpolicy.scripts.train_mpe import parse_args as train_mpe_parse_args
from onpolicy.utils.wandb_config_yaml import save_namespace_as_eval_config_yaml


def main() -> None:
    if "--dump_to" not in sys.argv:
        print("Required: --dump_to /path/to/models_dir", file=sys.stderr)
        sys.exit(1)
    idx = sys.argv.index("--dump_to")
    if idx + 1 >= len(sys.argv):
        print("--dump_to requires a directory path", file=sys.stderr)
        sys.exit(1)
    dump_to = os.path.abspath(sys.argv[idx + 1])
    train_argv = sys.argv[1:idx] + sys.argv[idx + 2 :]

    parser = get_config()
    all_args, parser = train_mpe_parse_args(train_argv, parser)
    if all_args.env_name == "GraphMPE":
        all_args, parser = graph_config(train_argv, parser)

    os.makedirs(dump_to, exist_ok=True)
    out = os.path.join(dump_to, "config.yaml")
    save_namespace_as_eval_config_yaml(all_args, out)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
