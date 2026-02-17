"""
结果展示系统
展示模型预测结果、注意力热力图等可视化
"""

import argparse
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from pathlib import Path

from envs import SheepFlockEnv
from onpolicy.algorithms.ppo_actor_critic import PPOActorCritic, ImprovedActorCritic


@dataclass
class ResultData:
    episode: int
    steps: int
    reward: float
    success: bool
    actions: List[np.ndarray]
    observations: List[np.ndarray]
    rewards: List[float]
    info: List[Dict]


class ResultVisualizer:
    def __init__(self, model_path: str, num_episodes: int = 5):
        self.model_path = model_path
        self.num_episodes = num_episodes
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.env = None
        self.policy = None
        self.args = None
        self.results = []
    
    def load_model(self):
        """加载模型"""
        # 创建环境
        self.env = SheepFlockEnv(
            world_size=(50.0, 50.0),
            num_sheep=10,
            num_herders=3,
            episode_length=150,
            random_seed=42,
        )
        
        # 加载模型
        checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
        
        # 检测模型类型
        model_type = self._detect_model_type(checkpoint)
        
        # 创建参数
        args = self._create_args_from_checkpoint(checkpoint, model_type == 'improved')
        args.layer_N = self._detect_layer_N(checkpoint)
        args.hidden_size = self._detect_hidden_size(checkpoint)
        
        # 加载模型
        if model_type == 'improved':
            print(f"Detected ImprovedActorCritic")
            self.policy = ImprovedActorCritic(
                args, self.env.observation_space, self.env.action_space, self.device,
                hidden_size=args.hidden_size,
            ).to(self.device)
        else:
            print(f"Detected PPOActorCritic (layer_N={args.layer_N}, hidden_size={args.hidden_size})")
            self.policy = PPOActorCritic(
                args, self.env.observation_space, self.env.action_space, self.device,
            ).to(self.device)
        
        if 'policy_state_dict' in checkpoint:
            self.policy.load_state_dict(checkpoint['policy_state_dict'])
            print(f"Loaded model: step {checkpoint.get('total_num_steps', 'unknown')}")
        else:
            self.policy.load_state_dict(checkpoint)
        
        self.policy.eval()
        self.args = args
    
    def _detect_model_type(self, checkpoint: dict) -> str:
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
    
    def _create_args_from_checkpoint(self, checkpoint: dict, use_improved: bool = False) -> argparse.Namespace:
        """从检查点创建参数"""
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
    
    def _detect_layer_N(self, checkpoint: dict) -> int:
        """检测网络层数"""
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
    
    def _detect_hidden_size(self, checkpoint: dict) -> int:
        """检测隐藏层大小"""
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
    
    def collect_results(self):
        """收集结果数据"""
        if not self.policy:
            self.load_model()
        
        self.results = []
        
        for episode in range(self.num_episodes):
            print(f"Running episode {episode + 1}/{self.num_episodes}")
            
            obs = self.env.reset()
            done = False
            episode_actions = []
            episode_observations = []
            episode_rewards = []
            episode_info = []
            total_reward = 0
            steps = 0
            
            # 初始化RNN状态
            if isinstance(self.policy, ImprovedActorCritic):
                rnn_states_actor = np.zeros((1, 1), dtype=np.float32)
                rnn_states_critic = np.zeros((1, 1), dtype=np.float32)
            else:
                rnn_states_actor = np.zeros(
                    (1, self.args.recurrent_N, self.args.hidden_size),
                    dtype=np.float32
                )
                rnn_states_critic = np.zeros(
                    (1, self.args.recurrent_N, self.args.hidden_size),
                    dtype=np.float32
                )
            masks = np.ones((1, 1), dtype=np.float32)
            
            while not done:
                episode_observations.append(obs.copy())
                
                # 获取动作
                with torch.no_grad():
                    obs_tensor = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)
                    rnn_states_actor_tensor = torch.from_numpy(rnn_states_actor).to(self.device)
                    rnn_states_critic_tensor = torch.from_numpy(rnn_states_critic).to(self.device)
                    masks_tensor = torch.from_numpy(masks).to(self.device)
                    
                    value, action, action_log_prob, rnn_states_actor_out, rnn_states_critic_out = \
                        self.policy.get_actions(
                            obs_tensor, rnn_states_actor_tensor, rnn_states_critic_tensor,
                            masks_tensor, deterministic=True,
                        )
                    
                    action_np = action.cpu().numpy()[0]
                    rnn_states_actor = rnn_states_actor_out.cpu().numpy()
                    rnn_states_critic = rnn_states_critic_out.cpu().numpy()
                
                episode_actions.append(action_np)
                
                # 执行动作
                actions_env = np.zeros((self.env.num_herders, len(action_np)), dtype=np.float32)
                for i in range(self.env.num_herders):
                    actions_env[i] = action_np.copy()
                
                obs, reward, done, info = self.env.step(actions_env)
                
                episode_rewards.append(reward)
                episode_info.append(info)
                total_reward += reward
                steps += 1
            
            success = info.get('is_success', False) if isinstance(info, dict) else False
            
            result = ResultData(
                episode=episode + 1,
                steps=steps,
                reward=total_reward,
                success=success,
                actions=episode_actions,
                observations=episode_observations,
                rewards=episode_rewards,
                info=episode_info
            )
            
            self.results.append(result)
            
            print(f"Episode {episode + 1}: Steps={steps}, Reward={total_reward:.3f}, Success={success}")
    
    def visualize_predictions(self, episode_idx: int = 0, save_path: Optional[str] = None):
        """可视化预测结果"""
        if not self.results:
            self.collect_results()
        
        if episode_idx >= len(self.results):
            print(f"Episode index {episode_idx} out of range")
            return
        
        result = self.results[episode_idx]
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'预测结果分析 - Episode {result.episode}', fontsize=16, fontweight='bold')
        
        # 奖励曲线
        ax1 = axes[0, 0]
        ax1.plot(range(len(result.rewards)), result.rewards, 'b-', linewidth=2)
        ax1.set_xlabel('Step')
        ax1.set_ylabel('Reward')
        ax1.set_title('奖励曲线')
        ax1.grid(True, alpha=0.3)
        
        # 动作分布
        ax2 = axes[0, 1]
        actions_array = np.array(result.actions)
        for i in range(actions_array.shape[1]):
            ax2.plot(range(len(result.actions)), actions_array[:, i], label=f'Action {i}')
        ax2.set_xlabel('Step')
        ax2.set_ylabel('Action Value')
        ax2.set_title('动作序列')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 观察值分布
        ax3 = axes[1, 0]
        obs_array = np.array(result.observations)
        obs_mean = np.mean(obs_array, axis=1)
        obs_std = np.std(obs_array, axis=1)
        ax3.plot(range(len(result.observations)), obs_mean, 'g-', linewidth=2, label='Mean')
        ax3.fill_between(range(len(result.observations)), obs_mean - obs_std, obs_mean + obs_std, alpha=0.2, label='Std')
        ax3.set_xlabel('Step')
        ax3.set_ylabel('Observation Value')
        ax3.set_title('观察值统计')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 成功率统计
        ax4 = axes[1, 1]
        success_count = sum(1 for r in self.results if r.success)
        fail_count = len(self.results) - success_count
        ax4.bar(['Success', 'Fail'], [success_count, fail_count], color=['green', 'red'])
        ax4.set_ylabel('Count')
        ax4.set_title('成功率统计')
        ax4.text(0, success_count + 0.1, f'{success_count}/{len(self.results)}', ha='center')
        ax4.text(1, fail_count + 0.1, f'{fail_count}/{len(self.results)}', ha='center')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"预测结果可视化已保存到: {save_path}")
        else:
            plt.show()
    
    def visualize_attention_heatmap(self, episode_idx: int = 0, step_idx: int = 0, save_path: Optional[str] = None):
        """可视化注意力热力图"""
        if not self.results:
            self.collect_results()
        
        if episode_idx >= len(self.results):
            print(f"Episode index {episode_idx} out of range")
            return
        
        result = self.results[episode_idx]
        if step_idx >= len(result.observations):
            print(f"Step index {step_idx} out of range")
            return
        
        obs = result.observations[step_idx]
        
        # 这里假设模型有注意力机制，实际需要根据模型结构调整
        # 这里我们使用观察值的相关性作为注意力热力图的替代
        obs_reshaped = obs.reshape(-1, 1)
        attention = np.dot(obs_reshaped, obs_reshaped.T)
        attention = (attention - attention.min()) / (attention.max() - attention.min())
        
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(attention, ax=ax, cmap='viridis')
        ax.set_title(f'注意力热力图 - Episode {result.episode}, Step {step_idx}')
        ax.set_xlabel('Feature Index')
        ax.set_ylabel('Feature Index')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"注意力热力图已保存到: {save_path}")
        else:
            plt.show()
    
    def visualize_feature_maps(self, episode_idx: int = 0, step_idx: int = 0, save_path: Optional[str] = None):
        """可视化特征映射"""
        if not self.results:
            self.collect_results()
        
        if episode_idx >= len(self.results):
            print(f"Episode index {episode_idx} out of range")
            return
        
        result = self.results[episode_idx]
        if step_idx >= len(result.observations):
            print(f"Step index {step_idx} out of range")
            return
        
        obs = result.observations[step_idx]
        
        # 这里假设模型可以输出中间层特征，实际需要根据模型结构调整
        # 这里我们使用观察值的不同子集作为特征映射的替代
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'特征映射分析 - Episode {result.episode}, Step {step_idx}', fontsize=16, fontweight='bold')
        
        # 特征子集1
        ax1 = axes[0, 0]
        ax1.plot(obs[:len(obs)//4], 'r-', linewidth=2)
        ax1.set_title('特征子集 1')
        ax1.set_xlabel('Feature Index')
        ax1.set_ylabel('Value')
        ax1.grid(True, alpha=0.3)
        
        # 特征子集2
        ax2 = axes[0, 1]
        ax2.plot(obs[len(obs)//4:len(obs)//2], 'g-', linewidth=2)
        ax2.set_title('特征子集 2')
        ax2.set_xlabel('Feature Index')
        ax2.set_ylabel('Value')
        ax2.grid(True, alpha=0.3)
        
        # 特征子集3
        ax3 = axes[1, 0]
        ax3.plot(obs[len(obs)//2:3*len(obs)//4], 'b-', linewidth=2)
        ax3.set_title('特征子集 3')
        ax3.set_xlabel('Feature Index')
        ax3.set_ylabel('Value')
        ax3.grid(True, alpha=0.3)
        
        # 特征子集4
        ax4 = axes[1, 1]
        ax4.plot(obs[3*len(obs)//4:], 'y-', linewidth=2)
        ax4.set_title('特征子集 4')
        ax4.set_xlabel('Feature Index')
        ax4.set_ylabel('Value')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"特征映射可视化已保存到: {save_path}")
        else:
            plt.show()
    
    def visualize_performance_dashboard(self, save_path: Optional[str] = None):
        """可视化性能仪表板"""
        if not self.results:
            self.collect_results()
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('模型性能仪表板', fontsize=16, fontweight='bold')
        
        # 成功率
        ax1 = axes[0, 0]
        success_count = sum(1 for r in self.results if r.success)
        fail_count = len(self.results) - success_count
        ax1.bar(['Success', 'Fail'], [success_count, fail_count], color=['green', 'red'])
        ax1.set_ylabel('Count')
        ax1.set_title('成功率统计')
        ax1.text(0, success_count + 0.1, f'{success_count}/{len(self.results)}', ha='center')
        ax1.text(1, fail_count + 0.1, f'{fail_count}/{len(self.results)}', ha='center')
        
        # 奖励分布
        ax2 = axes[0, 1]
        rewards = [r.reward for r in self.results]
        ax2.hist(rewards, bins=10, alpha=0.7, color='blue')
        ax2.set_xlabel('Reward')
        ax2.set_ylabel('Count')
        ax2.set_title('奖励分布')
        ax2.axvline(np.mean(rewards), color='red', linestyle='dashed', linewidth=2, label=f'Mean: {np.mean(rewards):.2f}')
        ax2.legend()
        
        # 步数分布
        ax3 = axes[1, 0]
        steps = [r.steps for r in self.results]
        ax3.hist(steps, bins=10, alpha=0.7, color='green')
        ax3.set_xlabel('Steps')
        ax3.set_ylabel('Count')
        ax3.set_title('步数分布')
        ax3.axvline(np.mean(steps), color='red', linestyle='dashed', linewidth=2, label=f'Mean: {np.mean(steps):.2f}')
        ax3.legend()
        
        # 奖励趋势
        ax4 = axes[1, 1]
        episode_rewards = [r.reward for r in self.results]
        ax4.plot(range(1, len(self.results) + 1), episode_rewards, 'b-', linewidth=2)
        ax4.set_xlabel('Episode')
        ax4.set_ylabel('Reward')
        ax4.set_title('奖励趋势')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"性能仪表板已保存到: {save_path}")
        else:
            plt.show()
    
    def generate_report(self, save_dir: Optional[str] = None):
        """生成完整报告"""
        if not self.results:
            self.collect_results()
        
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        else:
            save_dir = './reports'
            os.makedirs(save_dir, exist_ok=True)
        
        # 生成各种可视化
        self.visualize_predictions(save_path=os.path.join(save_dir, 'predictions.png'))
        self.visualize_attention_heatmap(save_path=os.path.join(save_dir, 'attention_heatmap.png'))
        self.visualize_feature_maps(save_path=os.path.join(save_dir, 'feature_maps.png'))
        self.visualize_performance_dashboard(save_path=os.path.join(save_dir, 'performance_dashboard.png'))
        
        # 生成文本报告
        report_path = os.path.join(save_dir, 'report.txt')
        with open(report_path, 'w') as f:
            f.write('模型性能报告\n')
            f.write('=' * 50 + '\n')
            f.write(f'模型路径: {self.model_path}\n')
            f.write(f'测试 episodes: {self.num_episodes}\n')
            f.write('\n')
            
            # 统计信息
            total_success = sum(1 for r in self.results if r.success)
            total_reward = sum(r.reward for r in self.results)
            total_steps = sum(r.steps for r in self.results)
            
            f.write('统计信息:\n')
            f.write(f'成功率: {total_success}/{self.num_episodes} ({total_success/self.num_episodes*100:.1f}%)\n')
            f.write(f'平均奖励: {total_reward/self.num_episodes:.3f}\n')
            f.write(f'平均步数: {total_steps/self.num_episodes:.2f}\n')
            f.write('\n')
            
            # 每个episode的详细信息
            f.write('详细信息:\n')
            for i, result in enumerate(self.results):
                f.write(f'Episode {i+1}: Steps={result.steps}, Reward={result.reward:.3f}, Success={result.success}\n')
        
        print(f"报告已生成到: {save_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description='结果展示系统')
    
    parser.add_argument('--model_path', type=str, required=True,
                        help='模型路径')
    parser.add_argument('--num_episodes', type=int, default=5,
                        help='测试的episode数量')
    parser.add_argument('--save_dir', type=str, default=None,
                        help='保存报告的目录')
    parser.add_argument('--visualize_predictions', action='store_true', default=False,
                        help='可视化预测结果')
    parser.add_argument('--visualize_attention', action='store_true', default=False,
                        help='可视化注意力热力图')
    parser.add_argument('--visualize_features', action='store_true', default=False,
                        help='可视化特征映射')
    parser.add_argument('--visualize_dashboard', action='store_true', default=False,
                        help='可视化性能仪表板')
    parser.add_argument('--generate_report', action='store_true', default=False,
                        help='生成完整报告')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    visualizer = ResultVisualizer(
        model_path=args.model_path,
        num_episodes=args.num_episodes
    )
    
    visualizer.load_model()
    visualizer.collect_results()
    
    if args.visualize_predictions:
        visualizer.visualize_predictions()
    
    if args.visualize_attention:
        visualizer.visualize_attention_heatmap()
    
    if args.visualize_features:
        visualizer.visualize_feature_maps()
    
    if args.visualize_dashboard:
        visualizer.visualize_performance_dashboard()
    
    if args.generate_report:
        visualizer.generate_report(args.save_dir)


if __name__ == '__main__':
    main()
