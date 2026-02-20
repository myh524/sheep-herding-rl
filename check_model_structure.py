"""
检查模型结构
"""

import torch
from pathlib import Path

# 找到最新的模型
results_dir = Path("results/sheep_herding/default/ppo/seed1")
model_files = list(results_dir.rglob("models/model_*.pt"))

if model_files:
    latest_model = max(model_files, key=lambda x: x.stat().st_mtime)
    print(f"最新模型: {latest_model}")
    
    checkpoint = torch.load(latest_model, map_location='cpu', weights_only=False)
    
    if 'policy_state_dict' in checkpoint:
        state_dict = checkpoint['policy_state_dict']
        print(f"\nCheckpoint包含的keys:")
        for key in sorted(state_dict.keys()):
            print(f"  {key}: {state_dict[key].shape}")
    else:
        print(f"\nCheckpoint直接是state_dict")
        for key in sorted(checkpoint.keys()):
            print(f"  {key}: {checkpoint[key].shape}")
    
    print(f"\n其他信息:")
    for key in checkpoint.keys():
        if key != 'policy_state_dict':
            print(f"  {key}: {checkpoint[key]}")
