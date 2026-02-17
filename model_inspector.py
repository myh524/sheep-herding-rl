"""
模型状态检查器
可视化模型权重、梯度、激活值分布
"""

import argparse
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple
from pathlib import Path

from envs import SheepFlockEnv
from onpolicy.algorithms.ppo_actor_critic import PPOActorCritic, ImprovedActorCritic


class ModelInspector:
    def __init__(self, model_path: str, env: Optional[SheepFlockEnv] = None):
        self.model_path = model_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.policy, self.args = self._load_model(env)
        self.activations = {}
        
    def _load_model(self, env: Optional[SheepFlockEnv] = None) -> Tuple[torch.nn.Module, argparse.Namespace]:
        """加载模型"""
        checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
        
        args = self._create_args_from_checkpoint(checkpoint)
        
        if env is None:
            env = SheepFlockEnv(
                world_size=(50.0, 50.0),
                num_sheep=10,
                num_herders=3,
                episode_length=150,
                random_seed=42,
            )
        
        model_type = self._detect_model_type(checkpoint)
        
        if model_type == 'improved':
            print(f"Detected ImprovedActorCritic")
            policy = ImprovedActorCritic(
                args,
                env.observation_space,
                env.action_space,
                self.device,
                hidden_size=args.hidden_size,
            ).to(self.device)
        else:
            print(f"Detected PPOActorCritic")
            policy = PPOActorCritic(
                args,
                env.observation_space,
                env.action_space,
                self.device,
            ).to(self.device)
        
        if 'policy_state_dict' in checkpoint:
            policy.load_state_dict(checkpoint['policy_state_dict'])
            print(f"Loaded model: step {checkpoint.get('total_num_steps', 'unknown')}")
        else:
            policy.load_state_dict(checkpoint)
        
        policy.eval()
        return policy, args
    
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
    
    def _create_args_from_checkpoint(self, checkpoint: dict) -> argparse.Namespace:
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
    
    def _get_activations(self, layer, input, output):
        """获取激活值"""
        layer_name = f"{layer.__class__.__name__}_{id(layer)}"
        self.activations[layer_name] = output.detach().cpu().numpy()
    
    def _register_hooks(self):
        """注册钩子获取激活值"""
        self.activations.clear()
        
        def register_hook(module, name):
            def hook(module, input, output):
                self.activations[name] = output.detach().cpu().numpy()
            return hook
        
        # 为actor网络注册钩子
        if hasattr(self.policy, 'actor'):
            for i, layer in enumerate(self.policy.actor.modules()):
                if isinstance(layer, torch.nn.Linear):
                    layer.register_forward_hook(register_hook(layer, f'actor_linear_{i}'))
        
        # 为critic网络注册钩子
        if hasattr(self.policy, 'critic'):
            for i, layer in enumerate(self.policy.critic.modules()):
                if isinstance(layer, torch.nn.Linear):
                    layer.register_forward_hook(register_hook(layer, f'critic_linear_{i}'))
    
    def collect_activations(self, env: Optional[SheepFlockEnv] = None):
        """收集激活值"""
        if env is None:
            env = SheepFlockEnv(
                world_size=(50.0, 50.0),
                num_sheep=10,
                num_herders=3,
                episode_length=150,
                random_seed=42,
            )
        
        self._register_hooks()
        
        obs = env.reset()
        obs_tensor = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)
        
        # 初始化RNN状态
        if hasattr(self.args, 'recurrent_N'):
            rnn_states_actor = np.zeros(
                (1, self.args.recurrent_N, self.args.hidden_size),
                dtype=np.float32
            )
            rnn_states_critic = np.zeros(
                (1, self.args.recurrent_N, self.args.hidden_size),
                dtype=np.float32
            )
        else:
            rnn_states_actor = np.zeros((1, 1), dtype=np.float32)
            rnn_states_critic = np.zeros((1, 1), dtype=np.float32)
        
        masks = np.ones((1, 1), dtype=np.float32)
        
        rnn_states_actor_tensor = torch.from_numpy(rnn_states_actor).to(self.device)
        rnn_states_critic_tensor = torch.from_numpy(rnn_states_critic).to(self.device)
        masks_tensor = torch.from_numpy(masks).to(self.device)
        
        # 前向传播
        with torch.no_grad():
            value, action, action_log_prob, rnn_states_actor_out, rnn_states_critic_out = \
                self.policy.get_actions(
                    obs_tensor,
                    rnn_states_actor_tensor,
                    rnn_states_critic_tensor,
                    masks_tensor,
                    deterministic=True,
                )
    
    def visualize_weights(self, save_path: Optional[str] = None):
        """可视化权重分布"""
        state_dict = self.policy.state_dict()
        weight_layers = []
        
        for key, value in state_dict.items():
            if 'weight' in key and value.dim() == 2:
                weight_layers.append((key, value.cpu().numpy()))
        
        if not weight_layers:
            print("No weight layers found")
            return
        
        num_layers = len(weight_layers)
        rows = (num_layers + 2) // 3
        fig, axes = plt.subplots(rows, 3, figsize=(15, rows * 4))
        axes = axes.flatten()
        
        fig.suptitle('权重分布', fontsize=16, fontweight='bold')
        
        for i, (layer_name, weights) in enumerate(weight_layers):
            ax = axes[i]
            sns.histplot(weights.flatten(), bins=50, ax=ax)
            ax.set_title(f'{layer_name}\n均值: {weights.mean():.4f}, 标准差: {weights.std():.4f}')
            ax.set_xlabel('权重值')
            ax.set_ylabel('频次')
        
        for i in range(num_layers, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"权重可视化已保存到: {save_path}")
        else:
            plt.show()
    
    def visualize_gradients(self, env: Optional[SheepFlockEnv] = None, save_path: Optional[str] = None):
        """可视化梯度分布"""
        if env is None:
            env = SheepFlockEnv(
                world_size=(50.0, 50.0),
                num_sheep=10,
                num_herders=3,
                episode_length=150,
                random_seed=42,
            )
        
        # 准备数据
        obs = env.reset()
        obs_tensor = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)
        
        if hasattr(self.args, 'recurrent_N'):
            rnn_states_actor = np.zeros(
                (1, self.args.recurrent_N, self.args.hidden_size),
                dtype=np.float32
            )
            rnn_states_critic = np.zeros(
                (1, self.args.recurrent_N, self.args.hidden_size),
                dtype=np.float32
            )
        else:
            rnn_states_actor = np.zeros((1, 1), dtype=np.float32)
            rnn_states_critic = np.zeros((1, 1), dtype=np.float32)
        
        masks = np.ones((1, 1), dtype=np.float32)
        
        rnn_states_actor_tensor = torch.from_numpy(rnn_states_actor).to(self.device)
        rnn_states_critic_tensor = torch.from_numpy(rnn_states_critic).to(self.device)
        masks_tensor = torch.from_numpy(masks).to(self.device)
        
        # 前向传播
        value, action, action_log_prob, rnn_states_actor_out, rnn_states_critic_out = \
            self.policy.get_actions(
                obs_tensor,
                rnn_states_actor_tensor,
                rnn_states_critic_tensor,
                masks_tensor,
                deterministic=False,
            )
        
        # 反向传播
        action_log_prob.sum().backward()
        
        # 收集梯度
        grad_layers = []
        for name, param in self.policy.named_parameters():
            if param.grad is not None and param.dim() == 2:
                grad_layers.append((name, param.grad.cpu().numpy()))
        
        if not grad_layers:
            print("No gradients found")
            return
        
        num_layers = len(grad_layers)
        rows = (num_layers + 2) // 3
        fig, axes = plt.subplots(rows, 3, figsize=(15, rows * 4))
        axes = axes.flatten()
        
        fig.suptitle('梯度分布', fontsize=16, fontweight='bold')
        
        for i, (layer_name, grads) in enumerate(grad_layers):
            ax = axes[i]
            sns.histplot(grads.flatten(), bins=50, ax=ax)
            ax.set_title(f'{layer_name}\n均值: {grads.mean():.4f}, 标准差: {grads.std():.4f}')
            ax.set_xlabel('梯度值')
            ax.set_ylabel('频次')
        
        for i in range(num_layers, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"梯度可视化已保存到: {save_path}")
        else:
            plt.show()
        
        # 清除梯度
        self.policy.zero_grad()
    
    def visualize_activations(self, save_path: Optional[str] = None):
        """可视化激活值分布"""
        if not self.activations:
            print("No activations collected. Please run collect_activations() first.")
            return
        
        num_activations = len(self.activations)
        rows = (num_activations + 2) // 3
        fig, axes = plt.subplots(rows, 3, figsize=(15, rows * 4))
        axes = axes.flatten()
        
        fig.suptitle('激活值分布', fontsize=16, fontweight='bold')
        
        for i, (layer_name, activations) in enumerate(self.activations.items()):
            ax = axes[i]
            sns.histplot(activations.flatten(), bins=50, ax=ax)
            ax.set_title(f'{layer_name}\n均值: {activations.mean():.4f}, 标准差: {activations.std():.4f}')
            ax.set_xlabel('激活值')
            ax.set_ylabel('频次')
        
        for i in range(num_activations, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"激活值可视化已保存到: {save_path}")
        else:
            plt.show()
    
    def visualize_weight_distributions(self, save_path: Optional[str] = None):
        """可视化权重分布对比"""
        state_dict = self.policy.state_dict()
        weight_data = {}
        
        for key, value in state_dict.items():
            if 'weight' in key and value.dim() == 2:
                layer_name = key.split('.')[0] if '.' in key else key
                if layer_name not in weight_data:
                    weight_data[layer_name] = []
                weight_data[layer_name].extend(value.cpu().numpy().flatten())
        
        if not weight_data:
            print("No weight data found")
            return
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        for layer_name, weights in weight_data.items():
            sns.kdeplot(weights, label=layer_name, ax=ax)
        
        ax.set_title('权重分布对比', fontsize=16, fontweight='bold')
        ax.set_xlabel('权重值')
        ax.set_ylabel('密度')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"权重分布对比已保存到: {save_path}")
        else:
            plt.show()


def parse_args():
    parser = argparse.ArgumentParser(description='模型状态检查器')
    
    parser.add_argument('--model_path', type=str, required=True,
                        help='模型路径')
    parser.add_argument('--save_dir', type=str, default=None,
                        help='保存可视化结果的目录')
    parser.add_argument('--visualize_weights', action='store_true', default=False,
                        help='可视化权重分布')
    parser.add_argument('--visualize_gradients', action='store_true', default=False,
                        help='可视化梯度分布')
    parser.add_argument('--visualize_activations', action='store_true', default=False,
                        help='可视化激活值分布')
    parser.add_argument('--visualize_all', action='store_true', default=False,
                        help='可视化所有指标')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    print(f"加载模型: {args.model_path}")
    inspector = ModelInspector(args.model_path)
    
    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)
    
    if args.visualize_all or args.visualize_weights:
        save_path = os.path.join(args.save_dir, 'weights_visualization.png') if args.save_dir else None
        inspector.visualize_weights(save_path)
    
    if args.visualize_all or args.visualize_gradients:
        save_path = os.path.join(args.save_dir, 'gradients_visualization.png') if args.save_dir else None
        inspector.visualize_gradients(save_path=save_path)
    
    if args.visualize_all or args.visualize_activations:
        inspector.collect_activations()
        save_path = os.path.join(args.save_dir, 'activations_visualization.png') if args.save_dir else None
        inspector.visualize_activations(save_path)
    
    if args.visualize_all:
        save_path = os.path.join(args.save_dir, 'weight_distributions.png') if args.save_dir else None
        inspector.visualize_weight_distributions(save_path)


if __name__ == '__main__':
    main()
