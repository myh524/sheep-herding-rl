"""
训练过程分析脚本
解析训练日志并绘制训练曲线
"""

import argparse
import os
import re
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path


@dataclass
class TrainingLog:
    episode: List[int] = None
    total_steps: List[int] = None
    avg_reward: List[float] = None
    train_loss: List[float] = None
    stage: List[int] = None
    success_rate: List[float] = None
    learning_rate: List[float] = None
    grad_norm: List[float] = None
    entropy: List[float] = None
    value_loss: List[float] = None
    policy_loss: List[float] = None
    
    def __post_init__(self):
        if self.episode is None:
            self.episode = []
        if self.total_steps is None:
            self.total_steps = []
        if self.avg_reward is None:
            self.avg_reward = []
        if self.train_loss is None:
            self.train_loss = []
        if self.stage is None:
            self.stage = []
        if self.success_rate is None:
            self.success_rate = []
        if self.learning_rate is None:
            self.learning_rate = []
        if self.grad_norm is None:
            self.grad_norm = []
        if self.entropy is None:
            self.entropy = []
        if self.value_loss is None:
            self.value_loss = []
        if self.policy_loss is None:
            self.policy_loss = []


def parse_args():
    parser = argparse.ArgumentParser(description='分析训练日志')
    
    parser.add_argument('--log_dir', type=str, required=True,
                        help='训练日志目录路径')
    parser.add_argument('--compare_dirs', type=str, nargs='+', default=None,
                        help='用于对比的其他训练目录')
    parser.add_argument('--smooth', type=int, default=10,
                        help='平滑窗口大小，0表示不平滑')
    parser.add_argument('--output', type=str, default=None,
                        help='输出图片保存路径')
    parser.add_argument('--show', action='store_true', default=True,
                        help='显示图表')
    
    return parser.parse_args()


def parse_log_file(log_path: str) -> TrainingLog:
    """解析训练日志文件"""
    log = TrainingLog()
    
    if not os.path.exists(log_path):
        print(f"警告: 日志文件不存在: {log_path}")
        return log
    
    with open(log_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            metrics = {}
            parts = line.split('|')
            for part in parts:
                part = part.strip()
                if ':' in part:
                    key, value = part.split(':', 1)
                    key = key.strip()
                    value = value.strip()
                    metrics[key] = value
            
            if 'episode' in metrics:
                log.episode.append(int(metrics['episode']))
            if 'total_steps' in metrics:
                log.total_steps.append(int(metrics['total_steps']))
            if 'avg_reward' in metrics:
                log.avg_reward.append(float(metrics['avg_reward']))
            if 'train_loss' in metrics:
                log.train_loss.append(float(metrics['train_loss']))
            if 'stage' in metrics:
                log.stage.append(int(metrics['stage']))
            if 'success_rate' in metrics:
                rate_str = metrics['success_rate'].rstrip('%')
                log.success_rate.append(float(rate_str) / 100.0)
            if 'learning_rate' in metrics:
                log.learning_rate.append(float(metrics['learning_rate']))
            if 'grad_norm' in metrics:
                log.grad_norm.append(float(metrics['grad_norm']))
            if 'entropy' in metrics:
                log.entropy.append(float(metrics['entropy']))
            if 'value_loss' in metrics:
                log.value_loss.append(float(metrics['value_loss']))
            if 'policy_loss' in metrics:
                log.policy_loss.append(float(metrics['policy_loss']))
    
    return log


def smooth_data(data: List[float], window: int) -> np.ndarray:
    """使用移动平均平滑数据"""
    if window <= 1 or len(data) < window:
        return np.array(data)
    
    data = np.array(data)
    smoothed = np.convolve(data, np.ones(window)/window, mode='valid')
    
    padding = np.full(window - 1, data[0])
    for i in range(window - 1):
        padding[i] = np.mean(data[:i+1])
    
    return np.concatenate([padding, smoothed])


def plot_single_run(log: TrainingLog, args: argparse.Namespace, 
                    title: str = "训练曲线"):
    """绘制单个训练run的曲线"""
    fig, axes = plt.subplots(3, 2, figsize=(14, 15))
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    if len(log.episode) == 0:
        print("警告: 没有有效的训练数据")
        return fig
    
    ax1, ax2, ax3, ax4, ax5, ax6 = axes.flatten()
    
    if log.avg_reward:
        rewards = smooth_data(log.avg_reward, args.smooth) if args.smooth else np.array(log.avg_reward)
        ax1.plot(log.episode, log.avg_reward, alpha=0.3, color='blue', label='原始')
        if args.smooth:
            ax1.plot(log.episode, rewards, color='blue', linewidth=2, label=f'平滑(w={args.smooth})')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('平均奖励')
        ax1.set_title('奖励曲线')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
    
    if log.train_loss:
        losses = smooth_data(log.train_loss, args.smooth) if args.smooth else np.array(log.train_loss)
        ax2.plot(log.episode, log.train_loss, alpha=0.3, color='red', label='原始')
        if args.smooth:
            ax2.plot(log.episode, losses, color='red', linewidth=2, label=f'平滑(w={args.smooth})')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('训练损失')
        ax2.set_title('损失曲线')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    if log.success_rate:
        rates = [r * 100 for r in log.success_rate]
        ax3.plot(log.episode[:len(rates)], rates, color='green', linewidth=2)
        ax3.set_xlabel('Episode')
        ax3.set_ylabel('成功率 (%)')
        ax3.set_title('成功率曲线')
        ax3.set_ylim(0, 100)
        ax3.grid(True, alpha=0.3)
    
    if log.stage:
        ax4.plot(log.episode[:len(log.stage)], log.stage, color='purple', linewidth=2)
        ax4.set_xlabel('Episode')
        ax4.set_ylabel('课程阶段')
        ax4.set_title('课程学习进度')
        ax4.grid(True, alpha=0.3)
    
    if log.learning_rate:
        lr = smooth_data(log.learning_rate, args.smooth) if args.smooth else np.array(log.learning_rate)
        ax5.plot(log.episode[:len(log.learning_rate)], log.learning_rate, alpha=0.3, color='orange', label='原始')
        if args.smooth:
            ax5.plot(log.episode[:len(log.learning_rate)], lr, color='orange', linewidth=2, label=f'平滑(w={args.smooth})')
        ax5.set_xlabel('Episode')
        ax5.set_ylabel('学习率')
        ax5.set_title('学习率曲线')
        ax5.set_yscale('log')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
    
    if log.grad_norm:
        grad = smooth_data(log.grad_norm, args.smooth) if args.smooth else np.array(log.grad_norm)
        ax6.plot(log.episode[:len(log.grad_norm)], log.grad_norm, alpha=0.3, color='brown', label='原始')
        if args.smooth:
            ax6.plot(log.episode[:len(log.grad_norm)], grad, color='brown', linewidth=2, label=f'平滑(w={args.smooth})')
        ax6.set_xlabel('Episode')
        ax6.set_ylabel('梯度范数')
        ax6.set_title('梯度范数曲线')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_comparison(logs: List[Tuple[str, TrainingLog]], args: argparse.Namespace):
    """绘制多个训练run的对比图"""
    fig, axes = plt.subplots(3, 2, figsize=(14, 15))
    fig.suptitle('训练对比', fontsize=16, fontweight='bold')
    
    ax1, ax2, ax3, ax4, ax5, ax6 = axes.flatten()
    colors = plt.cm.tab10(np.linspace(0, 1, len(logs)))
    
    for (name, log), color in zip(logs, colors):
        if len(log.episode) == 0:
            continue
        
        label = Path(name).name if name else 'default'
        
        if log.avg_reward:
            rewards = smooth_data(log.avg_reward, args.smooth) if args.smooth else np.array(log.avg_reward)
            ax1.plot(log.episode, rewards, color=color, linewidth=2, label=label)
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('平均奖励')
        ax1.set_title('奖励对比')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        if log.train_loss:
            losses = smooth_data(log.train_loss, args.smooth) if args.smooth else np.array(log.train_loss)
            ax2.plot(log.episode, losses, color=color, linewidth=2, label=label)
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('训练损失')
        ax2.set_title('损失对比')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        if log.success_rate:
            rates = [r * 100 for r in log.success_rate]
            ax3.plot(log.episode[:len(rates)], rates, color=color, linewidth=2, label=label)
        ax3.set_xlabel('Episode')
        ax3.set_ylabel('成功率 (%)')
        ax3.set_title('成功率对比')
        ax3.set_ylim(0, 100)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        if log.stage:
            ax4.plot(log.episode[:len(log.stage)], log.stage, color=color, linewidth=2, label=label)
        ax4.set_xlabel('Episode')
        ax4.set_ylabel('课程阶段')
        ax4.set_title('课程学习进度')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        if log.learning_rate:
            lr = smooth_data(log.learning_rate, args.smooth) if args.smooth else np.array(log.learning_rate)
            ax5.plot(log.episode[:len(log.learning_rate)], lr, color=color, linewidth=2, label=label)
        ax5.set_xlabel('Episode')
        ax5.set_ylabel('学习率')
        ax5.set_title('学习率对比')
        ax5.set_yscale('log')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        if log.grad_norm:
            grad = smooth_data(log.grad_norm, args.smooth) if args.smooth else np.array(log.grad_norm)
            ax6.plot(log.episode[:len(log.grad_norm)], grad, color=color, linewidth=2, label=label)
        ax6.set_xlabel('Episode')
        ax6.set_ylabel('梯度范数')
        ax6.set_title('梯度范数对比')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def find_log_file(log_dir: str) -> Optional[str]:
    """查找日志文件"""
    possible_names = ['training_log.txt', 'log.txt', 'train.log']
    for name in possible_names:
        path = os.path.join(log_dir, name)
        if os.path.exists(path):
            return path
    
    for f in os.listdir(log_dir):
        if f.endswith('.txt') and 'log' in f.lower():
            return os.path.join(log_dir, f)
    
    return None


def print_statistics(log: TrainingLog, name: str = ""):
    """打印统计信息"""
    if len(log.episode) == 0:
        return
    
    print(f"\n{'='*50}")
    print(f"统计信息: {name}")
    print(f"{'='*50}")
    print(f"总Episode数: {len(log.episode)}")
    
    if log.avg_reward:
        print(f"平均奖励: {np.mean(log.avg_reward):.3f} ± {np.std(log.avg_reward):.3f}")
        print(f"最大奖励: {np.max(log.avg_reward):.3f}")
        print(f"最小奖励: {np.min(log.avg_reward):.3f}")
    
    if log.train_loss:
        print(f"平均损失: {np.mean(log.train_loss):.4f}")
        print(f"最终损失: {log.train_loss[-1]:.4f}")
    
    if log.success_rate:
        print(f"最终成功率: {log.success_rate[-1]*100:.1f}%")


def main():
    args = parse_args()
    
    log_file = find_log_file(args.log_dir)
    if log_file is None:
        print(f"错误: 在 {args.log_dir} 中找不到日志文件")
        return
    
    print(f"解析日志: {log_file}")
    main_log = parse_log_file(log_file)
    print_statistics(main_log, args.log_dir)
    
    if args.compare_dirs:
        logs = [(args.log_dir, main_log)]
        for dir_path in args.compare_dirs:
            log_file = find_log_file(dir_path)
            if log_file:
                log = parse_log_file(log_file)
                logs.append((dir_path, log))
                print_statistics(log, dir_path)
        
        fig = plot_comparison(logs, args)
    else:
        fig = plot_single_run(main_log, args, title=f"训练曲线 - {args.log_dir}")
    
    if args.output:
        fig.savefig(args.output, dpi=150, bbox_inches='tight')
        print(f"\n图表已保存到: {args.output}")
    
    if args.show:
        plt.show()
    else:
        plt.close()


if __name__ == '__main__':
    main()
