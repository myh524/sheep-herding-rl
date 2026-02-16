"""
策略评估脚本
运行多个episode并输出性能统计
"""

import argparse
import os
import torch
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, field
from tqdm import tqdm

from envs import SheepFlockEnv
from onpolicy.algorithms.ppo_actor_critic import PPOActorCritic, ImprovedActorCritic


OBS_LABELS = [
    '目标距离',
    '方向cos',
    '方向sin',
    '形状λ1',
    '形状λ2',
    '主方向cos',
    '主方向sin',
    '羊群扩散',
    '站位半径均值',
    '站位半径std',
    '角度分布',
    '羊群数量'
]

ACTION_LABELS = [
    '半径均值',
    '半径std',
    '相对角度',
    '集中度'
]


@dataclass
class EpisodeResult:
    total_reward: float
    steps: int
    success: bool
    final_distance: float
    final_spread: float
    rewards_history: List[float] = field(default_factory=list)
    actions_history: List[np.ndarray] = field(default_factory=list)
    obs_history: List[np.ndarray] = field(default_factory=list)
    values_history: List[float] = field(default_factory=list)


def parse_args():
    parser = argparse.ArgumentParser(description='评估训练好的PPO模型')
    
    parser.add_argument('--model_path', type=str, required=True,
                        help='模型checkpoint路径')
    parser.add_argument('--num_episodes', type=int, default=100,
                        help='评估的episode数量')
    parser.add_argument('--deterministic', action='store_true', default=True,
                        help='使用确定性策略')
    
    parser.add_argument('--num_sheep', type=int, default=10)
    parser.add_argument('--num_herders', type=int, default=3)
    parser.add_argument('--world_size', type=float, nargs=2, default=[50.0, 50.0])
    parser.add_argument('--episode_length', type=int, default=100)
    parser.add_argument('--seed', type=int, default=None)
    
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
    parser.add_argument('--use_improved', action='store_true', default=False,
                        help='使用ImprovedActorCritic网络架构')
    parser.add_argument('--stacked_frames', type=int, default=1)
    
    parser.add_argument('--verbose', action='store_true', default=False,
                        help='输出每个episode的详细信息')
    parser.add_argument('--save_results', type=str, default=None,
                        help='保存评估结果到文件')
    
    return parser.parse_args()


def create_args_from_checkpoint(checkpoint_path: str, use_improved: bool = False) -> argparse.Namespace:
    """从checkpoint路径推断参数"""
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
    """检测模型类型"""
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
    """从checkpoint检测layer_N参数"""
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


def detect_hidden_size(checkpoint: dict) -> int:
    """从checkpoint检测hidden_size参数"""
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
    """加载模型，自动检测模型类型"""
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    model_type = detect_model_type(checkpoint)
    
    if args is None:
        args = create_args_from_checkpoint(model_path, model_type == 'improved')
    
    args.layer_N = detect_layer_N(checkpoint)
    args.hidden_size = detect_hidden_size(checkpoint)
    
    if model_type == 'improved':
        print(f"检测到 ImprovedActorCritic 网络架构")
        policy = ImprovedActorCritic(
            args,
            env.observation_space,
            env.action_space,
            device,
            hidden_size=args.hidden_size,
        ).to(device)
    else:
        print(f"检测到 PPOActorCritic 网络架构 (layer_N={args.layer_N}, hidden_size={args.hidden_size})")
        policy = PPOActorCritic(
            args,
            env.observation_space,
            env.action_space,
            device,
        ).to(device)
    
    if 'policy_state_dict' in checkpoint:
        policy.load_state_dict(checkpoint['policy_state_dict'])
        print(f"加载模型: step {checkpoint.get('total_num_steps', 'unknown')}")
    else:
        policy.load_state_dict(checkpoint)
    
    policy.eval()
    return policy, args


def run_episode(env: SheepFlockEnv, policy, 
                device: torch.device, args: argparse.Namespace,
                is_improved: bool = False,
                collect_history: bool = False) -> EpisodeResult:
    """运行单个episode"""
    result = EpisodeResult(
        total_reward=0.0,
        steps=0,
        success=False,
        final_distance=0.0,
        final_spread=0.0,
    )
    
    obs = env.reset()
    
    if not is_improved:
        rnn_states_actor = np.zeros(
            (1, args.recurrent_N, args.hidden_size), dtype=np.float32
        )
        rnn_states_critic = np.zeros(
            (1, args.recurrent_N, args.hidden_size), dtype=np.float32
        )
    else:
        rnn_states_actor = np.zeros((1, 1), dtype=np.float32)
        rnn_states_critic = np.zeros((1, 1), dtype=np.float32)
    masks = np.ones((1, 1), dtype=np.float32)
    
    for step in range(env.episode_length):
        with torch.no_grad():
            obs_tensor = torch.from_numpy(obs).float().unsqueeze(0).to(device)
            rnn_states_actor_tensor = torch.from_numpy(rnn_states_actor).to(device)
            rnn_states_critic_tensor = torch.from_numpy(rnn_states_critic).to(device)
            masks_tensor = torch.from_numpy(masks).to(device)
            
            value, action, action_log_prob, rnn_states_actor_out, rnn_states_critic_out = \
                policy.get_actions(
                    obs_tensor,
                    rnn_states_actor_tensor,
                    rnn_states_critic_tensor,
                    masks_tensor,
                    deterministic=args.deterministic,
                )
            
            rnn_states_actor = rnn_states_actor_out.cpu().numpy()
            rnn_states_critic = rnn_states_critic_out.cpu().numpy()
            
            action_np = action.cpu().numpy()[0]
            value_np = value.cpu().numpy()[0, 0]
        
        actions_env = np.zeros((env.num_herders, len(action_np)), dtype=np.float32)
        for i in range(env.num_herders):
            actions_env[i] = action_np.copy()
        
        next_obs, reward, done, info = env.step(actions_env)
        
        result.total_reward += reward
        result.steps = step + 1
        
        if collect_history:
            result.rewards_history.append(reward)
            result.actions_history.append(action_np.copy())
            result.obs_history.append(obs.copy())
            result.values_history.append(value_np)
        
        obs = next_obs
        
        if done:
            break
    
    result.success = info.get('is_success', False)
    result.final_distance = info.get('distance_to_target', 0)
    result.final_spread = info.get('flock_spread', 0)
    
    return result


def print_statistics(results: List[EpisodeResult], verbose: bool = False):
    """打印统计信息"""
    successes = [r.success for r in results]
    rewards = [r.total_reward for r in results]
    steps = [r.steps for r in results]
    distances = [r.final_distance for r in results]
    spreads = [r.final_spread for r in results]
    
    print(f"\n{'='*60}")
    print("评估结果统计")
    print(f"{'='*60}")
    print(f"总Episode数: {len(results)}")
    print(f"\n成功率: {sum(successes)}/{len(successes)} ({sum(successes)/len(successes)*100:.1f}%)")
    
    print(f"\n奖励统计:")
    print(f"  平均: {np.mean(rewards):.3f} ± {np.std(rewards):.3f}")
    print(f"  最大: {np.max(rewards):.3f}")
    print(f"  最小: {np.min(rewards):.3f}")
    print(f"  中位数: {np.median(rewards):.3f}")
    
    print(f"\n步数统计:")
    print(f"  平均: {np.mean(steps):.1f} ± {np.std(steps):.1f}")
    print(f"  最大: {np.max(steps)}")
    print(f"  最小: {np.min(steps)}")
    
    print(f"\n最终距离统计:")
    print(f"  平均: {np.mean(distances):.2f} ± {np.std(distances):.2f}")
    
    print(f"\n最终扩散统计:")
    print(f"  平均: {np.mean(spreads):.2f} ± {np.std(spreads):.2f}")
    
    success_rewards = [r.total_reward for r in results if r.success]
    fail_rewards = [r.total_reward for r in results if not r.success]
    
    if success_rewards:
        print(f"\n成功Episode平均奖励: {np.mean(success_rewards):.3f}")
    if fail_rewards:
        print(f"失败Episode平均奖励: {np.mean(fail_rewards):.3f}")
    
    if verbose:
        print(f"\n{'='*60}")
        print("各Episode详情")
        print(f"{'='*60}")
        for i, r in enumerate(results):
            status = "成功" if r.success else "失败"
            print(f"Episode {i+1:3d}: {status:4s} | 奖励: {r.total_reward:7.3f} | "
                  f"步数: {r.steps:3d} | 距离: {r.final_distance:5.2f}")


def analyze_actions(results: List[EpisodeResult]):
    """分析动作分布"""
    all_actions = []
    for r in results:
        all_actions.extend(r.actions_history)
    
    if not all_actions:
        return
    
    actions = np.array(all_actions)
    
    print(f"\n{'='*60}")
    print("动作分布分析")
    print(f"{'='*60}")
    
    for i, label in enumerate(ACTION_LABELS):
        print(f"\n{label}:")
        print(f"  均值: {np.mean(actions[:, i]):.3f}")
        print(f"  标准差: {np.std(actions[:, i]):.3f}")
        print(f"  范围: [{np.min(actions[:, i]):.3f}, {np.max(actions[:, i]):.3f}]")


def analyze_observations(results: List[EpisodeResult]):
    """分析观测分布"""
    all_obs = []
    for r in results:
        all_obs.extend(r.obs_history)
    
    if not all_obs:
        return
    
    obs = np.array(all_obs)
    
    print(f"\n{'='*60}")
    print("观测分布分析")
    print(f"{'='*60}")
    
    for i, label in enumerate(OBS_LABELS):
        print(f"\n{label}:")
        print(f"  均值: {np.mean(obs[:, i]):.3f}")
        print(f"  标准差: {np.std(obs[:, i]):.3f}")
        print(f"  范围: [{np.min(obs[:, i]):.3f}, {np.max(obs[:, i]):.3f}]")


def save_results(results: List[EpisodeResult], path: str):
    """保存评估结果"""
    import json
    
    data = {
        'summary': {
            'num_episodes': len(results),
            'success_rate': sum(r.success for r in results) / len(results),
            'mean_reward': np.mean([r.total_reward for r in results]),
            'std_reward': np.std([r.total_reward for r in results]),
            'mean_steps': np.mean([r.steps for r in results]),
            'std_steps': np.std([r.steps for r in results]),
        },
        'episodes': [
            {
                'total_reward': r.total_reward,
                'steps': r.steps,
                'success': r.success,
                'final_distance': r.final_distance,
                'final_spread': r.final_spread,
            }
            for r in results
        ]
    }
    
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"\n结果已保存到: {path}")


def main():
    args = parse_args()
    
    print(f"模型路径: {args.model_path}")
    print(f"评估配置: {args.num_episodes} episodes, "
          f"{args.num_sheep}只羊, {args.num_herders}只机械狗")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    env = SheepFlockEnv(
        world_size=tuple(args.world_size),
        num_sheep=args.num_sheep,
        num_herders=args.num_herders,
        episode_length=args.episode_length,
        random_seed=args.seed,
    )
    
    policy, args = load_model(args.model_path, env, device, args)
    
    is_improved = isinstance(policy, ImprovedActorCritic)
    
    results = []
    collect_history = args.verbose
    
    print(f"\n开始评估...")
    for i in tqdm(range(args.num_episodes), desc="评估进度"):
        result = run_episode(env, policy, device, args, is_improved, collect_history)
        results.append(result)
    
    print_statistics(results, args.verbose)
    
    if collect_history:
        analyze_actions(results)
        analyze_observations(results)
    
    if args.save_results:
        save_results(results, args.save_results)
    
    env.close()


if __name__ == '__main__':
    main()
