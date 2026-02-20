"""
可视化训练过程
快速演示训练过程，使用较少的步数
"""

import argparse
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
from datetime import datetime
from collections import deque

from envs import SheepFlockEnv
from onpolicy.algorithms.ppo_actor_critic import PPOActorCritic
from onpolicy.utils.ppo_buffer import PPOReplayBuffer


class TrainingVisualizer:
    """训练过程可视化器"""
    
    def __init__(self, env, policy, args):
        self.env = env
        self.policy = policy
        self.args = args
        
        # 训练指标历史
        self.episode_rewards = []
        self.episode_lengths = []
        self.value_losses = []
        self.policy_losses = []
        self.entropy_losses = []
        self.timesteps = []
        
        # 当前episode数据
        self.current_episode_reward = 0
        self.current_episode_length = 0
        
        # 设置可视化
        self.setup_visualization()
        
    def setup_visualization(self):
        """设置可视化界面"""
        self.fig = plt.figure(figsize=(16, 10))
        
        # 环境可视化 (左侧)
        self.ax_env = self.fig.add_subplot(2, 2, 1)
        
        # 奖励曲线 (右上)
        self.ax_reward = self.fig.add_subplot(2, 2, 2)
        
        # 损失曲线 (左下)
        self.ax_loss = self.fig.add_subplot(2, 2, 3)
        
        # 训练信息 (右下)
        self.ax_info = self.fig.add_subplot(2, 2, 4)
        self.ax_info.axis('off')
        
        plt.tight_layout()
        
    def render_env(self, obs):
        """渲染环境状态"""
        self.ax_env.clear()
        
        self.ax_env.set_xlim(0, self.env.world_size[0])
        self.ax_env.set_ylim(0, self.env.world_size[1])
        self.ax_env.set_aspect('equal')
        self.ax_env.set_facecolor('#f5f5f5')
        
        # 目标
        target = self.env.scenario.get_target_position()
        target_circle = patches.Circle(target, 3.0, color='green', alpha=0.3)
        self.ax_env.add_patch(target_circle)
        self.ax_env.plot(target[0], target[1], 'g*', markersize=15)
        
        # 羊群
        flock_state = self.env.scenario.get_flock_state()
        sheep_positions = np.array(flock_state['positions'])
        self.ax_env.scatter(sheep_positions[:, 0], sheep_positions[:, 1], 
                           c='gray', s=60, alpha=0.7, marker='o', edgecolors='black')
        
        flock_center = self.env.scenario.get_flock_center()
        self.ax_env.plot(flock_center[0], flock_center[1], 'k+', markersize=10, markeredgewidth=2)
        
        # 机械狗
        herder_positions = self.env.scenario.get_herder_positions()
        self.ax_env.scatter(herder_positions[:, 0], herder_positions[:, 1],
                           c='blue', s=100, alpha=0.9, marker='s', edgecolors='darkblue')
        
        self.ax_env.set_title(f'Environment - Step {self.current_episode_length}', fontsize=10, fontweight='bold')
        self.ax_env.grid(True, alpha=0.3)
        
    def update_plots(self, step):
        """更新训练曲线"""
        # 奖励曲线
        self.ax_reward.clear()
        if len(self.episode_rewards) > 0:
            self.ax_reward.plot(self.timesteps, self.episode_rewards, 'b-', linewidth=2, label='Episode Reward')
            if len(self.episode_rewards) > 10:
                # 移动平均
                window = min(10, len(self.episode_rewards))
                moving_avg = np.convolve(self.episode_rewards, np.ones(window)/window, mode='valid')
                self.ax_reward.plot(self.timesteps[window-1:], moving_avg, 'r--', linewidth=2, label='Moving Avg')
        self.ax_reward.set_xlabel('Timesteps')
        self.ax_reward.set_ylabel('Episode Reward')
        self.ax_reward.set_title('Training Progress', fontsize=10, fontweight='bold')
        self.ax_reward.legend()
        self.ax_reward.grid(True, alpha=0.3)
        
        # 损失曲线
        self.ax_loss.clear()
        if len(self.value_losses) > 0:
            self.ax_loss.plot(self.value_losses, 'g-', linewidth=2, label='Value Loss')
        if len(self.policy_losses) > 0:
            self.ax_loss.plot(self.policy_losses, 'r-', linewidth=2, label='Policy Loss')
        self.ax_loss.set_xlabel('Updates')
        self.ax_loss.set_ylabel('Loss')
        self.ax_loss.set_title('Training Losses', fontsize=10, fontweight='bold')
        self.ax_loss.legend()
        self.ax_loss.grid(True, alpha=0.3)
        
        # 训练信息
        self.ax_info.clear()
        self.ax_info.axis('off')
        
        info_text = f"Training Information\n"
        info_text += f"{'='*30}\n"
        info_text += f"Total Steps: {step}\n"
        info_text += f"Episodes: {len(self.episode_rewards)}\n"
        
        if len(self.episode_rewards) > 0:
            info_text += f"\nRecent Performance:\n"
            info_text += f"  Last Reward: {self.episode_rewards[-1]:.2f}\n"
            recent = self.episode_rewards[-min(10, len(self.episode_rewards)):]
            info_text += f"  Avg (last 10): {np.mean(recent):.2f}\n"
            info_text += f"  Best: {max(self.episode_rewards):.2f}\n"
        
        if len(self.value_losses) > 0:
            info_text += f"\nRecent Losses:\n"
            info_text += f"  Value: {self.value_losses[-1]:.4f}\n"
            info_text += f"  Policy: {self.policy_losses[-1]:.4f}\n"
        
        self.ax_info.text(0.1, 0.9, info_text, transform=self.ax_info.transAxes,
                         fontsize=10, verticalalignment='top', fontfamily='monospace',
                         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
    def collect_episode(self, buffer, device, render=False):
        """收集一个episode的数据"""
        obs = self.env.reset()
        self.current_episode_reward = 0
        self.current_episode_length = 0
        
        rnn_states_actor = np.zeros((1, self.args.recurrent_N, self.args.hidden_size), dtype=np.float32)
        rnn_states_critic = np.zeros((1, self.args.recurrent_N, self.args.hidden_size), dtype=np.float32)
        masks = np.ones((1, 1), dtype=np.float32)
        
        episode_data = []
        
        for step in range(self.env.episode_length):
            if render:
                self.render_env(obs)
                plt.pause(0.01)
            
            with torch.no_grad():
                obs_tensor = torch.from_numpy(obs).float().unsqueeze(0).to(device)
                rnn_states_actor_tensor = torch.from_numpy(rnn_states_actor).to(device)
                rnn_states_critic_tensor = torch.from_numpy(rnn_states_critic).to(device)
                masks_tensor = torch.from_numpy(masks).to(device)
                
                value, action, action_log_prob, rnn_states_actor_out, rnn_states_critic_out = \
                    self.policy.get_actions(
                        obs_tensor, rnn_states_actor_tensor, rnn_states_critic_tensor,
                        masks_tensor, deterministic=False
                    )
                
                rnn_states_actor = rnn_states_actor_out.cpu().numpy()
                rnn_states_critic = rnn_states_critic_out.cpu().numpy()
                
                action_np = action.cpu().numpy()[0]
                value_np = value.cpu().numpy()[0][0]
                action_log_prob_np = action_log_prob.cpu().numpy()[0][0]
                
                # 确保action形状正确
                if isinstance(action_np, np.ndarray) and action_np.ndim == 0:
                    action_np = np.array([action_np.item()])
                elif not isinstance(action_np, np.ndarray):
                    action_np = np.array([action_np])
            
            next_obs, reward, done, info = self.env.step(action_np)
            
            self.current_episode_reward += reward
            self.current_episode_length += 1
            
            # 存储到buffer
            buffer.insert(
                obs[np.newaxis, :],
                rnn_states_actor[0],
                rnn_states_critic[0],
                action_np[np.newaxis, :],
                np.array([[action_log_prob_np]]),
                np.array([[value_np]]),
                np.array([[reward]]),
                masks[0],
            )
            
            episode_data.append({
                'obs': obs.copy(),
                'action': action_np.copy(),
                'reward': reward,
                'done': done
            })
            
            obs = next_obs
            masks = np.ones((1, 1), dtype=np.float32) if not done else np.zeros((1, 1), dtype=np.float32)
            
            if done:
                break
        
        return episode_data
    
    def train_update(self, buffer, device, optimizer):
        """执行一次训练更新"""
        if buffer.step < 10:  # 需要足够的数据
            return None, None, None
        
        # 计算returns
        with torch.no_grad():
            last_obs = torch.from_numpy(buffer.obs[buffer.step]).float().unsqueeze(0).to(device)
            last_rnn_states = torch.from_numpy(buffer.rnn_states_critic[buffer.step]).float().to(device)
            last_masks = torch.from_numpy(buffer.masks[buffer.step]).float().to(device)
            next_value = self.policy.get_values(last_obs, last_rnn_states, last_masks)
            next_value = next_value.cpu().numpy()[0]
        
        buffer.compute_returns(next_value)
        
        # 计算advantages
        advantages = buffer.returns[:-1] - buffer.value_preds[:-1]
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy_loss = 0
        update_count = 0
        
        # PPO更新
        for _ in range(self.args.ppo_epoch):
            data_generator = buffer.feed_forward_generator(
                advantages, num_mini_batch=self.args.num_mini_batch
            )
            
            for batch in data_generator:
                obs_batch, rnn_states_batch, rnn_states_critic_batch, actions_batch, \
                    value_preds_batch, return_batch, masks_batch, active_masks_batch, \
                    old_action_log_probs_batch, adv_targ, available_actions_batch = batch
                
                obs_tensor = torch.from_numpy(obs_batch).float().to(device)
                actions_tensor = torch.from_numpy(actions_batch).float().to(device)
                rnn_states_actor_tensor = torch.from_numpy(rnn_states_batch).float().to(device)
                rnn_states_critic_tensor = torch.from_numpy(rnn_states_critic_batch).float().to(device)
                masks_tensor = torch.from_numpy(masks_batch).float().to(device)
                old_action_log_probs_tensor = torch.from_numpy(old_action_log_probs_batch).float().to(device)
                adv_targ_tensor = torch.from_numpy(adv_targ).float().to(device)
                return_tensor = torch.from_numpy(return_batch).float().to(device)
                
                # 评估动作
                values, action_log_probs, entropy = self.policy.evaluate_actions(
                    obs_tensor, rnn_states_actor_tensor, rnn_states_critic_tensor,
                    actions_tensor, masks_tensor
                )
                
                # Policy loss
                ratio = torch.exp(action_log_probs - old_action_log_probs_tensor)
                surr1 = ratio * adv_targ_tensor
                surr2 = torch.clamp(ratio, 1 - self.args.clip_param, 1 + self.args.clip_param) * adv_targ_tensor
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                value_loss = 0.5 * (return_tensor - values).pow(2).mean()
                
                # Entropy loss
                entropy_loss = -entropy.mean()
                
                # Total loss
                loss = policy_loss + self.args.value_loss_coef * value_loss + self.args.entropy_coef * entropy_loss
                
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.args.max_grad_norm)
                optimizer.step()
                
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy_loss += entropy_loss.item()
                update_count += 1
        
        buffer.after_update()
        
        if update_count > 0:
            return total_value_loss / update_count, total_policy_loss / update_count, total_entropy_loss / update_count
        return None, None, None


def main():
    parser = argparse.ArgumentParser(description='可视化训练过程')
    
    # 环境参数
    parser.add_argument('--num_sheep', type=int, default=2)
    parser.add_argument('--num_herders', type=int, default=3)
    parser.add_argument('--episode_length', type=int, default=30)
    parser.add_argument('--world_size', type=float, nargs=2, default=[50.0, 50.0])
    
    # 训练参数（简化）
    parser.add_argument('--num_env_steps', type=int, default=3000, help='总训练步数')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--hidden_size', type=int, default=128)
    parser.add_argument('--layer_N', type=int, default=2)
    
    # PPO参数
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--ppo_epoch', type=int, default=3)
    parser.add_argument('--num_mini_batch', type=int, default=2)
    parser.add_argument('--clip_param', type=float, default=0.2)
    parser.add_argument('--value_loss_coef', type=float, default=0.5)
    parser.add_argument('--entropy_coef', type=float, default=0.02)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--gae_lambda', type=float, default=0.95)
    parser.add_argument('--max_grad_norm', type=float, default=0.5)
    
    # 可视化参数
    parser.add_argument('--render_interval', type=int, default=5, help='每多少episode渲染一次')
    parser.add_argument('--update_interval', type=int, default=10, help='每多少episode更新一次')
    
    args = parser.parse_args()
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("="*60)
    print("可视化训练过程")
    print("="*60)
    print(f"总步数: {args.num_env_steps}")
    print(f"羊数量: {args.num_sheep}")
    print(f"机械狗数量: {args.num_herders}")
    print(f"Episode长度: {args.episode_length}")
    print("="*60)
    
    # 创建环境
    env = SheepFlockEnv(
        world_size=tuple(args.world_size),
        num_sheep=args.num_sheep,
        num_herders=args.num_herders,
        episode_length=args.episode_length,
        random_seed=args.seed,
    )
    
    # 设置PPO参数
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
    args.use_gae = True
    args.use_clipped_value_loss = True
    
    # 创建策略
    policy = PPOActorCritic(args, env.observation_space, env.action_space, device).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr)
    
    # 创建buffer
    args.n_rollout_threads = 1
    args.use_valuenorm = False
    buffer = PPOReplayBuffer(args, env.observation_space, env.action_space)
    
    # 创建可视化器
    visualizer = TrainingVisualizer(env, policy, args)
    
    # 训练循环
    total_steps = 0
    episode_count = 0
    
    while total_steps < args.num_env_steps:
        # 收集episode
        should_render = (episode_count + 1) % args.render_interval == 0
        episode_data = visualizer.collect_episode(buffer, device, render=should_render)
        
        episode_count += 1
        total_steps += visualizer.current_episode_length
        
        # 记录指标
        visualizer.episode_rewards.append(visualizer.current_episode_reward)
        visualizer.episode_lengths.append(visualizer.current_episode_length)
        visualizer.timesteps.append(total_steps)
        
        # 训练更新
        if episode_count % args.update_interval == 0:
            value_loss, policy_loss, entropy_loss = visualizer.train_update(buffer, device, optimizer)
            if value_loss is not None:
                visualizer.value_losses.append(value_loss)
                visualizer.policy_losses.append(policy_loss)
                visualizer.entropy_losses.append(entropy_loss)
        
        # 更新训练曲线（每个episode结束时）
        if should_render:
            visualizer.update_plots(total_steps)
        
        # 打印进度
        if episode_count % 5 == 0:
            print(f"Episode {episode_count} | Steps {total_steps}/{args.num_env_steps} | "
                  f"Reward: {visualizer.current_episode_reward:.2f}")
    
    print("\n" + "="*60)
    print("训练完成！")
    print("="*60)
    print(f"总Episodes: {episode_count}")
    print(f"总Steps: {total_steps}")
    print(f"平均奖励: {np.mean(visualizer.episode_rewards):.2f}")
    print(f"最佳奖励: {max(visualizer.episode_rewards):.2f}")
    
    # 保存最终图表
    save_dir = f"results/visualized_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(save_dir, exist_ok=True)
    visualizer.fig.savefig(os.path.join(save_dir, 'training_progress.png'), dpi=150, bbox_inches='tight')
    print(f"\n图表已保存到: {save_dir}")
    
    plt.show()
    env.close()


if __name__ == '__main__':
    main()
