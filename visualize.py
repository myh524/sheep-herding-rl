"""
模型可视化运行脚本
加载训练好的模型并实时渲染环境
"""

import argparse
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.widgets import Button
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

from envs import SheepFlockEnv
from onpolicy.algorithms.ppo_actor_critic import PPOActorCritic, ImprovedActorCritic

HERDER_EVASION_RADIUS = 8.0


@dataclass
class VisualizationState:
    current_step: int = 0
    total_reward: float = 0.0
    episode_done: bool = False
    success: bool = False
    action_history: List[np.ndarray] = None
    
    def __post_init__(self):
        if self.action_history is None:
            self.action_history = []


def parse_args():
    parser = argparse.ArgumentParser(description='可视化运行训练好的PPO模型')
    
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--num_episodes', type=int, default=5)
    parser.add_argument('--render_delay', type=int, default=30)
    
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
    parser.add_argument('--use_improved', action='store_true', default=False)
    parser.add_argument('--stacked_frames', type=int, default=1)
    
    parser.add_argument('--save_gif', type=str, default=None)
    parser.add_argument('--save_video', type=str, default=None)
    
    return parser.parse_args()


def create_args_from_checkpoint(checkpoint_path: str, use_improved: bool = False) -> argparse.Namespace:
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
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    model_type = detect_model_type(checkpoint)
    
    if args is None:
        args = create_args_from_checkpoint(model_path, model_type == 'improved')
    
    args.layer_N = detect_layer_N(checkpoint)
    args.hidden_size = detect_hidden_size(checkpoint)
    
    if model_type == 'improved':
        print(f"Detected ImprovedActorCritic")
        policy = ImprovedActorCritic(
            args, env.observation_space, env.action_space, device,
            hidden_size=args.hidden_size,
        ).to(device)
    else:
        print(f"Detected PPOActorCritic (layer_N={args.layer_N}, hidden_size={args.hidden_size})")
        policy = PPOActorCritic(
            args, env.observation_space, env.action_space, device,
        ).to(device)
    
    if 'policy_state_dict' in checkpoint:
        policy.load_state_dict(checkpoint['policy_state_dict'])
        print(f"Loaded model: step {checkpoint.get('total_num_steps', 'unknown')}")
    else:
        policy.load_state_dict(checkpoint)
    
    policy.eval()
    return policy, args


class Visualizer:
    def __init__(self, env: SheepFlockEnv, policy, device: torch.device, args: argparse.Namespace):
        self.env = env
        self.policy = policy
        self.device = device
        self.args = args
        
        self.state = VisualizationState()
        self.fig = None
        self.ax = None
        self.paused = False
        self.frames = []
        
        self.rnn_states_actor = None
        self.rnn_states_critic = None
        self.masks = None
        self.current_obs = None
        
        self.is_improved = isinstance(policy, ImprovedActorCritic)
        
    def setup_figure(self):
        self.fig, self.ax = plt.subplots(figsize=(10, 10))
        self.ax.set_xlim(0, self.env.world_size[0])
        self.ax.set_ylim(0, self.env.world_size[1])
        self.ax.set_aspect('equal')
        self.ax.set_facecolor('#f5f5f5')
        
        ax_pause = plt.axes([0.85, 0.01, 0.12, 0.04])
        self.btn_pause = Button(ax_pause, 'Pause/Resume')
        self.btn_pause.on_clicked(self.toggle_pause)
        
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        
        plt.tight_layout()
        
    def toggle_pause(self, event):
        self.paused = not self.paused
        
    def on_key_press(self, event):
        if event.key == ' ':
            self.paused = not self.paused
        
    def reset_episode(self):
        self.state = VisualizationState()
        self.current_obs = self.env.reset()
        
        if not self.is_improved:
            self.rnn_states_actor = np.zeros(
                (1, self.args.recurrent_N, self.args.hidden_size), dtype=np.float32
            )
            self.rnn_states_critic = np.zeros(
                (1, self.args.recurrent_N, self.args.hidden_size), dtype=np.float32
            )
        else:
            self.rnn_states_actor = np.zeros((1, 1), dtype=np.float32)
            self.rnn_states_critic = np.zeros((1, 1), dtype=np.float32)
        self.masks = np.ones((1, 1), dtype=np.float32)
        
    def get_action(self, obs: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            obs_tensor = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)
            rnn_states_actor_tensor = torch.from_numpy(self.rnn_states_actor).to(self.device)
            rnn_states_critic_tensor = torch.from_numpy(self.rnn_states_critic).to(self.device)
            masks_tensor = torch.from_numpy(self.masks).to(self.device)
            
            value, action, action_log_prob, rnn_states_actor_out, rnn_states_critic_out = \
                self.policy.get_actions(
                    obs_tensor, rnn_states_actor_tensor, rnn_states_critic_tensor,
                    masks_tensor, deterministic=True,
                )
            
            self.rnn_states_actor = rnn_states_actor_out.cpu().numpy()
            self.rnn_states_critic = rnn_states_critic_out.cpu().numpy()
            
            return action.cpu().numpy()[0]
    
    def render_env(self):
        self.ax.clear()
        
        self.ax.set_xlim(0, self.env.world_size[0])
        self.ax.set_ylim(0, self.env.world_size[1])
        self.ax.set_aspect('equal')
        self.ax.set_facecolor('#f5f5f5')
        
        target = self.env.scenario.get_target_position()
        target_circle = patches.Circle(target, 3.0, color='green', alpha=0.3)
        self.ax.add_patch(target_circle)
        self.ax.plot(target[0], target[1], 'g*', markersize=20)
        
        flock_state = self.env.scenario.get_flock_state()
        sheep_positions = flock_state['positions']
        
        if len(sheep_positions) > 0:
            sheep_positions = np.array(sheep_positions)
            self.ax.scatter(sheep_positions[:, 0], sheep_positions[:, 1], 
                           c='gray', s=80, alpha=0.7, marker='o', edgecolors='black')
            
            flock_center = self.env.scenario.get_flock_center()
            self.ax.plot(flock_center[0], flock_center[1], 'k+', markersize=12, markeredgewidth=2)
        
        herder_positions = self.env.scenario.herder_positions.copy()
        
        if len(self.state.action_history) > 0 and hasattr(self.env.scenario, 'herder_targets'):
            last_action = self.state.action_history[-1]
            herder_targets = self.env.scenario.herder_targets
            
            fc = self.env.scenario.get_flock_center()
            target_pos = self.env.scenario.get_target_position()
            target_direction = np.arctan2(
                target_pos[1] - fc[1],
                target_pos[0] - fc[0]
            )
            
            decoded = self.env.action_decoder.decode_action(last_action, target_direction)
            mu_r = decoded['mu_r']
            sigma_r = decoded['sigma_r']
            mu_theta = decoded['mu_theta']
            kappa = decoded['kappa']
            
            station_ring = patches.Circle(fc, mu_r, fill=False, 
                                         color='orange', linewidth=2, alpha=0.7)
            self.ax.add_patch(station_ring)
            
            main_target_x = fc[0] + mu_r * np.cos(mu_theta)
            main_target_y = fc[1] + mu_r * np.sin(mu_theta)
            self.ax.plot(main_target_x, main_target_y, 'o', color='red', 
                        markersize=12, markeredgecolor='darkred', markeredgewidth=2)
            
            if kappa < 0.1:
                wedge_angle = np.pi
            else:
                wedge_angle = 2.0 / np.sqrt(kappa)
            wedge_angle = min(wedge_angle, np.pi)
            
            wedge = patches.Wedge(
                fc, mu_r + sigma_r,
                np.degrees(mu_theta - wedge_angle),
                np.degrees(mu_theta + wedge_angle),
                width=2 * sigma_r if sigma_r > 0.1 else mu_r,
                facecolor='orange', alpha=0.2,
                edgecolor='darkorange', linewidth=1.5,
            )
            self.ax.add_patch(wedge)
            
            for i in range(self.env.num_herders):
                target_pos_i = herder_targets[i]
                
                self.ax.plot(target_pos_i[0], target_pos_i[1], 'o', color='darkorange', 
                            markersize=8, alpha=0.6, markeredgecolor='black', markeredgewidth=1)
                
                if i < len(herder_positions):
                    hpos = herder_positions[i]
                    direction = target_pos_i - hpos
                    dist = np.linalg.norm(direction)
                    
                    if dist > 0.5:
                        direction = direction / dist
                        arrow_len = min(dist * 0.4, 3.0)
                        self.ax.annotate('', 
                                        xy=(hpos[0] + direction[0] * arrow_len, 
                                            hpos[1] + direction[1] * arrow_len),
                                        xytext=hpos,
                                        arrowprops=dict(arrowstyle='->', color='green', 
                                                       lw=1.5, alpha=0.8))
            
            self.ax.text(fc[0], fc[1] - 2, f'r={mu_r:.1f} k={kappa:.1f}', 
                        ha='center', fontsize=9, color='darkorange', fontweight='bold')
        
        for i, hpos in enumerate(herder_positions):
            evasion_circle = patches.Circle(
                hpos, HERDER_EVASION_RADIUS, 
                fill=True, facecolor='blue', alpha=0.08,
                edgecolor='blue', linestyle='--', linewidth=1,
            )
            self.ax.add_patch(evasion_circle)
        
        self.ax.scatter(herder_positions[:, 0], herder_positions[:, 1],
                       c='blue', s=120, alpha=0.9, marker='s', edgecolors='darkblue')
        
        status = "PAUSED" if self.paused else "RUNNING"
        self.ax.set_title(f'Step {self.state.current_step} | Reward: {self.state.total_reward:.1f} | {status}', 
                         fontsize=12, fontweight='bold')
        self.ax.grid(True, alpha=0.3)
        
    def run_episode(self, episode_num: int, total_episodes: int):
        print(f"\n{'='*50}")
        print(f"Episode {episode_num}/{total_episodes}")
        print(f"{'='*50}")
        
        self.reset_episode()
        self.state.episode_done = False
        
        self.setup_figure()
        
        for step in range(self.env.episode_length):
            while self.paused:
                plt.pause(0.1)
            
            action = self.get_action(self.current_obs)
            
            actions_env = np.zeros((self.env.num_herders, len(action)), dtype=np.float32)
            for i in range(self.env.num_herders):
                actions_env[i] = action.copy()
            
            obs, reward, done, info = self.env.step(actions_env)
            
            self.state.current_step += 1
            self.state.total_reward += reward
            self.state.action_history.append(action.copy())
            
            self.render_env()
            
            if self.args.save_gif or self.args.save_video:
                self.fig.canvas.draw()
                self.frames.append(np.array(self.fig.canvas.renderer.buffer_rgba()))
            
            plt.pause(self.args.render_delay / 1000.0)
            
            self.current_obs = obs
            
            if done:
                break
        
        print(f"\nEpisode {episode_num} finished:")
        print(f"  Steps: {self.state.current_step}, Reward: {self.state.total_reward:.3f}")
        print(f"  Result: {'SUCCESS!' if self.state.success else 'FAILED'}")
        
        return self.state.success, self.state.total_reward, self.state.current_step
    
    def run(self):
        successes = []
        total_rewards = []
        total_steps = []
        
        for ep in range(self.args.num_episodes):
            success, reward, steps = self.run_episode(ep + 1, self.args.num_episodes)
            successes.append(success)
            total_rewards.append(reward)
            total_steps.append(steps)
        
        print(f"\n{'='*50}")
        print("Visualization completed!")
        print(f"Success rate: {sum(successes)}/{len(successes)} ({sum(successes)/len(successes)*100:.1f}%)")
        print(f"Average reward: {np.mean(total_rewards):.3f}")
        
        if self.args.save_gif:
            self.save_gif(self.args.save_gif)
        if self.args.save_video:
            self.save_video(self.args.save_video)
        
        plt.close('all')
    
    def save_gif(self, path: str):
        try:
            import imageio
            imageio.mimsave(path, self.frames, fps=20)
            print(f"GIF saved to: {path}")
        except ImportError:
            print("Warning: imageio required to save GIF")
    
    def save_video(self, path: str):
        try:
            import imageio
            imageio.mimsave(path, self.frames, fps=20)
            print(f"Video saved to: {path}")
        except ImportError:
            print("Warning: imageio required to save video")


def main():
    args = parse_args()
    
    print(f"Model: {args.model_path}")
    print(f"Config: {args.num_sheep} sheep, {args.num_herders} herders, world {args.world_size}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    env = SheepFlockEnv(
        world_size=tuple(args.world_size),
        num_sheep=args.num_sheep,
        num_herders=args.num_herders,
        episode_length=args.episode_length,
        random_seed=args.seed,
    )
    
    policy, args = load_model(args.model_path, env, device, args)
    
    visualizer = Visualizer(env, policy, device, args)
    visualizer.run()
    
    env.close()


if __name__ == '__main__':
    main()
