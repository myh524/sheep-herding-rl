#!/usr/bin/env python3
"""
训练分析报告生成器
生成详细的训练分析报告，包含统计信息和可视化图表
"""

import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'WenQuanYi Zen Hei', 'DejaVu Sans', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False


@dataclass
class TrainingStatistics:
    """训练统计数据结构"""
    # 奖励统计
    reward_mean: float = 0.0
    reward_std: float = 0.0
    reward_max: float = 0.0
    reward_min: float = 0.0
    reward_final: float = 0.0
    reward_best_episode: int = 0
    
    # 损失统计
    loss_mean: float = 0.0
    loss_std: float = 0.0
    loss_final: float = 0.0
    loss_trend: str = "stable"  # increasing, decreasing, stable
    
    # 价值损失统计
    value_loss_mean: float = 0.0
    value_loss_final: float = 0.0
    
    # 策略损失统计
    policy_loss_mean: float = 0.0
    policy_loss_final: float = 0.0
    
    # 梯度统计
    grad_norm_mean: float = 0.0
    grad_norm_max: float = 0.0
    grad_norm_final: float = 0.0
    
    # 熵统计
    entropy_mean: float = 0.0
    entropy_final: float = 0.0
    entropy_trend: str = "stable"
    
    # 学习率统计
    lr_initial: float = 0.0
    lr_final: float = 0.0
    lr_decay_ratio: float = 1.0
    
    # KL散度统计
    kl_mean: float = 0.0
    kl_max: float = 0.0
    
    # 训练效率
    total_episodes: int = 0
    total_steps: int = 0
    convergence_episode: Optional[int] = None
    
    # 稳定性指标
    reward_stability: float = 0.0  # 奖励稳定性分数 (0-1)
    loss_stability: float = 0.0    # 损失稳定性分数 (0-1)


class TrainingAnalyzer:
    """训练分析器"""
    
    def __init__(self):
        self.data: Dict[str, List[float]] = {}
        self.statistics = TrainingStatistics()
        
    def load_from_json(self, json_path: str) -> bool:
        """从JSON文件加载训练数据"""
        try:
            with open(json_path, 'r') as f:
                raw_data = json.load(f)
                
            if isinstance(raw_data, list):
                # 转换为字典格式
                for key in ['episode', 'total_steps', 'avg_reward', 'train_loss', 
                           'value_loss', 'policy_loss', 'entropy', 'grad_norm', 
                           'lr', 'entropy_coef', 'kl_divergence', 'kl_coef', 
                           'gae_lambda', 'value_pred_error']:
                    self.data[key] = [d.get(key, 0.0) for d in raw_data]
            else:
                self.data = raw_data
                
            return True
        except Exception as e:
            print(f"加载JSON文件失败: {e}")
            return False
            
    def load_from_log(self, log_path: str) -> bool:
        """从日志文件加载训练数据"""
        try:
            with open(log_path, 'r') as f:
                lines = f.readlines()
                
            # 初始化数据字典
            keys = ['episode', 'total_steps', 'avg_reward', 'train_loss', 
                   'value_loss', 'policy_loss', 'entropy', 'grad_norm', 
                   'lr', 'entropy_coef', 'kl_divergence', 'kl_coef', 
                   'gae_lambda', 'value_pred_error']
            for key in keys:
                self.data[key] = []
                
            for line in lines:
                metrics = {}
                parts = line.strip().split('|')
                
                for part in parts:
                    part = part.strip()
                    if ':' in part:
                        key, value = part.split(':', 1)
                        key = key.strip()
                        value = value.strip()
                        
                        try:
                            if '%' in value:
                                metrics[key] = float(value.rstrip('%')) / 100.0
                            else:
                                metrics[key] = float(value)
                        except ValueError:
                            pass
                            
                for key in keys:
                    self.data[key].append(metrics.get(key, 0.0))
                    
            return True
        except Exception as e:
            print(f"加载日志文件失败: {e}")
            return False
            
    def compute_statistics(self):
        """计算统计数据"""
        if not self.data.get('episode'):
            return
            
        # 奖励统计
        if self.data.get('avg_reward'):
            rewards = np.array(self.data['avg_reward'])
            self.statistics.reward_mean = float(np.mean(rewards))
            self.statistics.reward_std = float(np.std(rewards))
            self.statistics.reward_max = float(np.max(rewards))
            self.statistics.reward_min = float(np.min(rewards))
            self.statistics.reward_final = float(rewards[-1])
            self.statistics.reward_best_episode = int(np.argmax(rewards))
            
        # 损失统计
        if self.data.get('train_loss'):
            losses = np.array(self.data['train_loss'])
            self.statistics.loss_mean = float(np.mean(losses))
            self.statistics.loss_std = float(np.std(losses))
            self.statistics.loss_final = float(losses[-1])
            
            # 计算损失趋势
            if len(losses) > 10:
                first_half = np.mean(losses[:len(losses)//2])
                second_half = np.mean(losses[len(losses)//2:])
                if second_half > first_half * 1.2:
                    self.statistics.loss_trend = "increasing"
                elif second_half < first_half * 0.8:
                    self.statistics.loss_trend = "decreasing"
                else:
                    self.statistics.loss_trend = "stable"
                    
        # 价值损失统计
        if self.data.get('value_loss'):
            value_losses = np.array(self.data['value_loss'])
            self.statistics.value_loss_mean = float(np.mean(value_losses))
            self.statistics.value_loss_final = float(value_losses[-1])
            
        # 策略损失统计
        if self.data.get('policy_loss'):
            policy_losses = np.array(self.data['policy_loss'])
            self.statistics.policy_loss_mean = float(np.mean(policy_losses))
            self.statistics.policy_loss_final = float(policy_losses[-1])
            
        # 梯度统计
        if self.data.get('grad_norm'):
            grad_norms = np.array(self.data['grad_norm'])
            self.statistics.grad_norm_mean = float(np.mean(grad_norms))
            self.statistics.grad_norm_max = float(np.max(grad_norms))
            self.statistics.grad_norm_final = float(grad_norms[-1])
            
        # 熵统计
        if self.data.get('entropy'):
            entropies = np.array(self.data['entropy'])
            self.statistics.entropy_mean = float(np.mean(entropies))
            self.statistics.entropy_final = float(entropies[-1])
            
            # 计算熵趋势
            if len(entropies) > 10:
                first_half = np.mean(entropies[:len(entropies)//2])
                second_half = np.mean(entropies[len(entropies)//2:])
                if second_half < first_half * 0.5:
                    self.statistics.entropy_trend = "decreasing"
                elif second_half > first_half * 1.5:
                    self.statistics.entropy_trend = "increasing"
                else:
                    self.statistics.entropy_trend = "stable"
                    
        # 学习率统计
        if self.data.get('lr'):
            lrs = np.array(self.data['lr'])
            self.statistics.lr_initial = float(lrs[0])
            self.statistics.lr_final = float(lrs[-1])
            if self.statistics.lr_initial > 0:
                self.statistics.lr_decay_ratio = self.statistics.lr_final / self.statistics.lr_initial
                
        # KL散度统计
        if self.data.get('kl_divergence'):
            kl_divs = np.array(self.data['kl_divergence'])
            self.statistics.kl_mean = float(np.mean(np.abs(kl_divs)))
            self.statistics.kl_max = float(np.max(np.abs(kl_divs)))
            
        # 训练效率
        if self.data.get('episode'):
            self.statistics.total_episodes = len(self.data['episode'])
        if self.data.get('total_steps'):
            self.statistics.total_steps = int(self.data['total_steps'][-1]) if self.data['total_steps'] else 0
            
        # 计算收敛episode
        if self.data.get('avg_reward') and len(self.data['avg_reward']) > 20:
            rewards = np.array(self.data['avg_reward'])
            # 找到奖励开始稳定的点
            window = 20
            for i in range(len(rewards) - window):
                recent = rewards[i:i+window]
                if np.std(recent) < 0.1 * abs(np.mean(recent) + 1e-6):
                    self.statistics.convergence_episode = i
                    break
                    
        # 计算稳定性分数
        if self.data.get('avg_reward'):
            rewards = np.array(self.data['avg_reward'])
            if len(rewards) > 10:
                # 使用变异系数的倒数作为稳定性分数
                cv = self.statistics.reward_std / (abs(self.statistics.reward_mean) + 1e-6)
                self.statistics.reward_stability = float(min(1.0, 1.0 / (cv + 0.1)))
                
        if self.data.get('train_loss'):
            losses = np.array(self.data['train_loss'])
            if len(losses) > 10:
                cv = self.statistics.loss_std / (self.statistics.loss_mean + 1e-6)
                self.statistics.loss_stability = float(min(1.0, 1.0 / (cv + 0.1)))
                
    def generate_report(self, output_dir: str) -> str:
        """生成分析报告"""
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 生成图表
        self._generate_plots(output_dir, timestamp)
        
        # 生成文本报告
        report_path = self._generate_text_report(output_dir, timestamp)
        
        # 生成JSON报告
        self._generate_json_report(output_dir, timestamp)
        
        return report_path
        
    def _generate_plots(self, output_dir: str, timestamp: str):
        """生成可视化图表"""
        fig = plt.figure(figsize=(20, 16))
        gs = gridspec.GridSpec(4, 3, figure=fig, hspace=0.35, wspace=0.3)
        
        fig.suptitle('训练分析报告', fontsize=18, fontweight='bold')
        
        episodes = self.data.get('episode', list(range(len(self.data.get('avg_reward', [])))))
        
        # 1. 奖励曲线
        ax1 = fig.add_subplot(gs[0, 0])
        if self.data.get('avg_reward'):
            rewards = self.data['avg_reward']
            ax1.plot(episodes, rewards, 'b-', alpha=0.5, label='原始奖励')
            # 添加移动平均
            if len(rewards) > 10:
                window = min(20, len(rewards) // 5)
                smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
                ax1.plot(episodes[window-1:], smoothed, 'b-', linewidth=2, label=f'移动平均(w={window})')
            ax1.axhline(y=self.statistics.reward_mean, color='g', linestyle='--', alpha=0.5, label=f'平均值: {self.statistics.reward_mean:.2f}')
            ax1.axhline(y=self.statistics.reward_max, color='r', linestyle=':', alpha=0.5, label=f'最大值: {self.statistics.reward_max:.2f}')
        ax1.set_title('奖励曲线')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('奖励值')
        ax1.legend(loc='lower right', fontsize=8)
        ax1.grid(True, alpha=0.3)
        
        # 2. 损失曲线
        ax2 = fig.add_subplot(gs[0, 1])
        if self.data.get('train_loss'):
            ax2.plot(episodes, self.data['train_loss'], 'r-', alpha=0.5, label='总损失')
            if len(self.data['train_loss']) > 10:
                window = min(20, len(self.data['train_loss']) // 5)
                smoothed = np.convolve(self.data['train_loss'], np.ones(window)/window, mode='valid')
                ax2.plot(episodes[window-1:], smoothed, 'r-', linewidth=2, label='移动平均')
        ax2.set_title(f'损失曲线 (趋势: {self.statistics.loss_trend})')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('损失值')
        ax2.legend(loc='upper right', fontsize=8)
        ax2.grid(True, alpha=0.3)
        
        # 3. 损失分解
        ax3 = fig.add_subplot(gs[0, 2])
        if self.data.get('value_loss') and self.data.get('policy_loss'):
            ax3.plot(episodes, self.data['value_loss'], 'b-', alpha=0.7, label='价值损失')
            ax3.plot(episodes, self.data['policy_loss'], 'r-', alpha=0.7, label='策略损失')
        ax3.set_title('损失分解')
        ax3.set_xlabel('Episode')
        ax3.set_ylabel('损失值')
        ax3.legend(loc='upper right', fontsize=8)
        ax3.grid(True, alpha=0.3)
        
        # 4. 梯度范数
        ax4 = fig.add_subplot(gs[1, 0])
        if self.data.get('grad_norm'):
            ax4.plot(episodes, self.data['grad_norm'], 'm-', alpha=0.5)
            ax4.axhline(y=self.statistics.grad_norm_mean, color='g', linestyle='--', alpha=0.5, 
                       label=f'平均值: {self.statistics.grad_norm_mean:.2f}')
            ax4.axhline(y=self.statistics.grad_norm_max, color='r', linestyle=':', alpha=0.5,
                       label=f'最大值: {self.statistics.grad_norm_max:.2f}')
        ax4.set_title('梯度范数')
        ax4.set_xlabel('Episode')
        ax4.set_ylabel('范数值')
        ax4.legend(loc='upper right', fontsize=8)
        ax4.grid(True, alpha=0.3)
        
        # 5. 策略熵
        ax5 = fig.add_subplot(gs[1, 1])
        if self.data.get('entropy'):
            ax5.plot(episodes, self.data['entropy'], 'c-', alpha=0.5)
            ax5.axhline(y=self.statistics.entropy_mean, color='g', linestyle='--', alpha=0.5,
                       label=f'平均值: {self.statistics.entropy_mean:.4f}')
        ax5.set_title(f'策略熵 (趋势: {self.statistics.entropy_trend})')
        ax5.set_xlabel('Episode')
        ax5.set_ylabel('熵值')
        ax5.legend(loc='upper right', fontsize=8)
        ax5.grid(True, alpha=0.3)
        
        # 6. 学习率
        ax6 = fig.add_subplot(gs[1, 2])
        if self.data.get('lr'):
            ax6.plot(episodes, self.data['lr'], 'g-', linewidth=2)
            ax6.set_yscale('log')
        ax6.set_title(f'学习率 (衰减比: {self.statistics.lr_decay_ratio:.2%})')
        ax6.set_xlabel('Episode')
        ax6.set_ylabel('学习率')
        ax6.grid(True, alpha=0.3)
        
        # 7. KL散度
        ax7 = fig.add_subplot(gs[2, 0])
        if self.data.get('kl_divergence'):
            ax7.plot(episodes, self.data['kl_divergence'], 'orange', alpha=0.5)
            ax7.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax7.set_title(f'KL散度 (最大: {self.statistics.kl_max:.4f})')
        ax7.set_xlabel('Episode')
        ax7.set_ylabel('KL散度')
        ax7.grid(True, alpha=0.3)
        
        # 8. GAE Lambda
        ax8 = fig.add_subplot(gs[2, 1])
        if self.data.get('gae_lambda'):
            ax8.plot(episodes, self.data['gae_lambda'], 'purple', linewidth=2)
        ax8.set_title('GAE Lambda')
        ax8.set_xlabel('Episode')
        ax8.set_ylabel('Lambda值')
        ax8.grid(True, alpha=0.3)
        
        # 9. 熵系数
        ax9 = fig.add_subplot(gs[2, 2])
        if self.data.get('entropy_coef'):
            ax9.plot(episodes, self.data['entropy_coef'], 'brown', linewidth=2)
        ax9.set_title('熵系数')
        ax9.set_xlabel('Episode')
        ax9.set_ylabel('系数值')
        ax9.grid(True, alpha=0.3)
        
        # 10. 统计摘要面板
        ax10 = fig.add_subplot(gs[3, :])
        ax10.axis('off')
        
        stats_text = f"""
        ╔══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
        ║                                           训 统 计 摘 要                                                                    ║
        ╠══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╣
        ║  训练效率: 总Episodes: {self.statistics.total_episodes} | 总步数: {self.statistics.total_steps} | 收敛Episode: {self.statistics.convergence_episode or '未检测到'}          ║
        ╠══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╣
        ║  奖励统计: 平均: {self.statistics.reward_mean:.4f} ± {self.statistics.reward_std:.4f} | 最大: {self.statistics.reward_max:.4f} | 最小: {self.statistics.reward_min:.4f} | 最终: {self.statistics.reward_final:.4f}     ║
        ║  损失统计: 平均: {self.statistics.loss_mean:.4f} ± {self.statistics.loss_std:.4f} | 最终: {self.statistics.loss_final:.4f} | 趋势: {self.statistics.loss_trend}                              ║
        ║  梯度统计: 平均: {self.statistics.grad_norm_mean:.4f} | 最大: {self.statistics.grad_norm_max:.4f} | 最终: {self.statistics.grad_norm_final:.4f}                                      ║
        ║  熵统计:   平均: {self.statistics.entropy_mean:.4f} | 最终: {self.statistics.entropy_final:.4f} | 趋势: {self.statistics.entropy_trend}                                                ║
        ║  学习率:   初始: {self.statistics.lr_initial:.2e} | 最终: {self.statistics.lr_final:.2e} | 衰减比: {self.statistics.lr_decay_ratio:.2%}                                        ║
        ╠══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╣
        ║  稳定性评估: 奖励稳定性: {self.statistics.reward_stability:.2%} | 损失稳定性: {self.statistics.loss_stability:.2%}                                                  ║
        ╚══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝
        """
        
        ax10.text(0.5, 0.5, stats_text, transform=ax10.transAxes, fontsize=10,
                 verticalalignment='center', horizontalalignment='center',
                 fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        # 保存图表
        plot_path = os.path.join(output_dir, f'training_analysis_{timestamp}.png')
        fig.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        print(f"图表已保存: {plot_path}")
        
    def _generate_text_report(self, output_dir: str, timestamp: str) -> str:
        """生成文本报告"""
        report_path = os.path.join(output_dir, f'training_report_{timestamp}.txt')
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("                      训练分析报告\n")
            f.write(f"                    生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n\n")
            
            f.write("一、训练概览\n")
            f.write("-" * 40 + "\n")
            f.write(f"  总训练Episodes: {self.statistics.total_episodes}\n")
            f.write(f"  总训练步数: {self.statistics.total_steps}\n")
            f.write(f"  收敛Episode: {self.statistics.convergence_episode or '未检测到'}\n\n")
            
            f.write("二、奖励分析\n")
            f.write("-" * 40 + "\n")
            f.write(f"  平均奖励: {self.statistics.reward_mean:.4f} ± {self.statistics.reward_std:.4f}\n")
            f.write(f"  最大奖励: {self.statistics.reward_max:.4f} (Episode {self.statistics.reward_best_episode})\n")
            f.write(f"  最小奖励: {self.statistics.reward_min:.4f}\n")
            f.write(f"  最终奖励: {self.statistics.reward_final:.4f}\n")
            f.write(f"  奖励稳定性: {self.statistics.reward_stability:.2%}\n\n")
            
            f.write("三、损失分析\n")
            f.write("-" * 40 + "\n")
            f.write(f"  平均损失: {self.statistics.loss_mean:.4f} ± {self.statistics.loss_std:.4f}\n")
            f.write(f"  最终损失: {self.statistics.loss_final:.4f}\n")
            f.write(f"  损失趋势: {self.statistics.loss_trend}\n")
            f.write(f"  价值损失: {self.statistics.value_loss_mean:.4f} (平均), {self.statistics.value_loss_final:.4f} (最终)\n")
            f.write(f"  策略损失: {self.statistics.policy_loss_mean:.4f} (平均), {self.statistics.policy_loss_final:.4f} (最终)\n")
            f.write(f"  损失稳定性: {self.statistics.loss_stability:.2%}\n\n")
            
            f.write("四、梯度分析\n")
            f.write("-" * 40 + "\n")
            f.write(f"  平均梯度范数: {self.statistics.grad_norm_mean:.4f}\n")
            f.write(f"  最大梯度范数: {self.statistics.grad_norm_max:.4f}\n")
            f.write(f"  最终梯度范数: {self.statistics.grad_norm_final:.4f}\n\n")
            
            f.write("五、策略分析\n")
            f.write("-" * 40 + "\n")
            f.write(f"  平均熵: {self.statistics.entropy_mean:.4f}\n")
            f.write(f"  最终熵: {self.statistics.entropy_final:.4f}\n")
            f.write(f"  熵趋势: {self.statistics.entropy_trend}\n\n")
            
            f.write("六、优化器分析\n")
            f.write("-" * 40 + "\n")
            f.write(f"  初始学习率: {self.statistics.lr_initial:.2e}\n")
            f.write(f"  最终学习率: {self.statistics.lr_final:.2e}\n")
            f.write(f"  学习率衰减比: {self.statistics.lr_decay_ratio:.2%}\n")
            f.write(f"  平均KL散度: {self.statistics.kl_mean:.4f}\n")
            f.write(f"  最大KL散度: {self.statistics.kl_max:.4f}\n\n")
            
            f.write("七、训练质量评估\n")
            f.write("-" * 40 + "\n")
            
            # 综合评估
            score = 0
            comments = []
            
            # 奖励评估
            if self.statistics.reward_final > self.statistics.reward_mean:
                score += 20
                comments.append("奖励呈上升趋势")
            else:
                comments.append("奖励呈下降趋势")
                
            # 稳定性评估
            if self.statistics.reward_stability > 0.5:
                score += 20
                comments.append("奖励稳定性良好")
            else:
                comments.append("奖励波动较大")
                
            # 损失评估
            if self.statistics.loss_trend == "decreasing":
                score += 20
                comments.append("损失持续下降")
            elif self.statistics.loss_trend == "stable":
                score += 10
                comments.append("损失趋于稳定")
                
            # 梯度评估
            if self.statistics.grad_norm_max < 100:
                score += 20
                comments.append("梯度范数正常")
            else:
                comments.append("存在梯度爆炸风险")
                
            # 熵评估
            if self.statistics.entropy_final > 0.01:
                score += 20
                comments.append("策略保持足够探索")
            else:
                comments.append("策略可能过早收敛")
                
            f.write(f"  综合评分: {score}/100\n")
            f.write(f"  评估意见:\n")
            for comment in comments:
                f.write(f"    - {comment}\n")
                
            f.write("\n" + "=" * 80 + "\n")
            f.write("                          报告结束\n")
            f.write("=" * 80 + "\n")
            
        print(f"文本报告已保存: {report_path}")
        return report_path
        
    def _generate_json_report(self, output_dir: str, timestamp: str):
        """生成JSON报告"""
        report_path = os.path.join(output_dir, f'training_report_{timestamp}.json')
        
        report = {
            'timestamp': timestamp,
            'statistics': {
                'total_episodes': self.statistics.total_episodes,
                'total_steps': self.statistics.total_steps,
                'convergence_episode': self.statistics.convergence_episode,
                'reward': {
                    'mean': self.statistics.reward_mean,
                    'std': self.statistics.reward_std,
                    'max': self.statistics.reward_max,
                    'min': self.statistics.reward_min,
                    'final': self.statistics.reward_final,
                    'best_episode': self.statistics.reward_best_episode,
                    'stability': self.statistics.reward_stability,
                },
                'loss': {
                    'mean': self.statistics.loss_mean,
                    'std': self.statistics.loss_std,
                    'final': self.statistics.loss_final,
                    'trend': self.statistics.loss_trend,
                    'stability': self.statistics.loss_stability,
                },
                'value_loss': {
                    'mean': self.statistics.value_loss_mean,
                    'final': self.statistics.value_loss_final,
                },
                'policy_loss': {
                    'mean': self.statistics.policy_loss_mean,
                    'final': self.statistics.policy_loss_final,
                },
                'gradient': {
                    'mean': self.statistics.grad_norm_mean,
                    'max': self.statistics.grad_norm_max,
                    'final': self.statistics.grad_norm_final,
                },
                'entropy': {
                    'mean': self.statistics.entropy_mean,
                    'final': self.statistics.entropy_final,
                    'trend': self.statistics.entropy_trend,
                },
                'learning_rate': {
                    'initial': self.statistics.lr_initial,
                    'final': self.statistics.lr_final,
                    'decay_ratio': self.statistics.lr_decay_ratio,
                },
                'kl_divergence': {
                    'mean': self.statistics.kl_mean,
                    'max': self.statistics.kl_max,
                },
            }
        }
        
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
            
        print(f"JSON报告已保存: {report_path}")


def main():
    parser = argparse.ArgumentParser(description='训练分析报告生成器')
    parser.add_argument('--log_file', type=str, default=None, help='训练日志文件路径')
    parser.add_argument('--json_file', type=str, default=None, help='训练JSON文件路径')
    parser.add_argument('--output_dir', type=str, default='./analysis_reports', help='输出目录')
    
    args = parser.parse_args()
    
    analyzer = TrainingAnalyzer()
    
    if args.json_file:
        if not analyzer.load_from_json(args.json_file):
            return
    elif args.log_file:
        if not analyzer.load_from_log(args.log_file):
            return
    else:
        print("请指定 --log_file 或 --json_file 参数")
        return
        
    analyzer.compute_statistics()
    report_path = analyzer.generate_report(args.output_dir)
    
    print(f"\n分析报告已生成: {report_path}")


if __name__ == '__main__':
    main()
