#!/usr/bin/env python3
"""
分层牧羊训练入口：与 train_ppo.py 共用解析与 PPOTrainer，通过 CLI 启用低层 MAPPO 动力学。

典型用法（低层冻结，仅训高层）::

    python train_hierarchical.py --low-level-model-dir path/to/run/models \\
        --num-env-steps 500000 ...

交替占位（周期性打印低层微调提示）::

    python train_hierarchical.py --low-level-model-dir ... \\
        --hierarchical-phase alternating_stub --low-level-finetune-every 50 ...
"""

from train_ppo import main

if __name__ == "__main__":
    main()
