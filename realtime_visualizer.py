#!/usr/bin/env python3
"""
实时训练可视化工具
监控正在进行的训练过程，实时更新图表
"""

import os
import sys
import time
import re
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from collections import deque

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'WenQuanYi Zen Hei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class RealTimeVisualizer:
    def __init__(self, log_file=None, max_history=500):
        """初始化实时可视化工具"""
        self.log_file = log_file
        self.max_history = max_history
        
        # 存储训练数据
        self.data = {
            'episode': [],
            'total_steps': [],
            'avg_reward': [],
            'train_loss': [],
            'lr': [],
            'entropy_coef': [],
            'kl_divergence': [],
            'kl_coef': [],
            'gae_lambda': []
        }
        
        # 创建图表
        self.fig, self.axes = plt.subplots(3, 2, figsize=(15, 12))
        self.fig.suptitle('实时训练监控', fontsize=16)
        
        # 定义图表
        self.reward_ax = self.axes[0, 0]
        self.loss_ax = self.axes[0, 1]
        self.lr_ax = self.axes[1, 0]
        self.entropy_ax = self.axes[1, 1]
        self.kl_ax = self.axes[2, 0]
        self.gae_ax = self.axes[2, 1]
        
        # 设置图表标题
        self.reward_ax.set_title('平均奖励')
        self.loss_ax.set_title('训练损失')
        self.lr_ax.set_title('学习率')
        self.entropy_ax.set_title('熵系数')
        self.kl_ax.set_title('KL散度')
        self.gae_ax.set_title('GAE Lambda')
        
        # 设置x轴标签
        for ax in self.axes.flat:
            ax.set_xlabel('Episode')
        
        # 调整布局
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
    def parse_log_line(self, line):
        """解析日志行"""
        # 匹配episode行
        pattern = r'episode: (\d+) \| total_steps: (\d+) \| avg_reward: ([\-\d\.]+) \| train_loss: ([\-\d\.]+) \| lr: ([\-\d\.]+) \| entropy_coef: ([\-\d\.]+) \| kl_divergence: ([\-\d\.]+) \| kl_coef: ([\-\d\.]+) \| gae_lambda: ([\-\d\.]+)'
        match = re.match(pattern, line)
        
        if match:
            return {
                'episode': int(match.group(1)),
                'total_steps': int(match.group(2)),
                'avg_reward': float(match.group(3)),
                'train_loss': float(match.group(4)),
                'lr': float(match.group(5)),
                'entropy_coef': float(match.group(6)),
                'kl_divergence': float(match.group(7)),
                'kl_coef': float(match.group(8)),
                'gae_lambda': float(match.group(9))
            }
        return None
    
    def update_data(self, new_data):
        """更新训练数据"""
        for key, value in new_data.items():
            self.data[key].append(value)
            # 保持数据量在最大历史范围内
            if len(self.data[key]) > self.max_history:
                self.data[key] = self.data[key][-self.max_history:]
    
    def update_plots(self):
        """更新图表"""
        # 清空所有图表
        for ax in self.axes.flat:
            ax.clear()
        
        # 重新设置标题
        self.reward_ax.set_title('平均奖励')
        self.loss_ax.set_title('训练损失')
        self.lr_ax.set_title('学习率')
        self.entropy_ax.set_title('熵系数')
        self.kl_ax.set_title('KL散度')
        self.gae_ax.set_title('GAE Lambda')
        
        # 重新设置x轴标签
        for ax in self.axes.flat:
            ax.set_xlabel('Episode')
        
        # 绘制图表
        if self.data['episode']:
            # 奖励曲线
            self.reward_ax.plot(self.data['episode'], self.data['avg_reward'], 'b-')
            self.reward_ax.set_ylabel('奖励值')
            
            # 损失曲线
            self.loss_ax.plot(self.data['episode'], self.data['train_loss'], 'r-')
            self.loss_ax.set_ylabel('损失值')
            
            # 学习率曲线
            self.lr_ax.plot(self.data['episode'], self.data['lr'], 'g-')
            self.lr_ax.set_ylabel('学习率')
            
            # 熵系数曲线
            self.entropy_ax.plot(self.data['episode'], self.data['entropy_coef'], 'm-')
            self.entropy_ax.set_ylabel('熵系数')
            
            # KL散度曲线
            self.kl_ax.plot(self.data['episode'], self.data['kl_divergence'], 'c-')
            self.kl_ax.set_ylabel('KL散度')
            
            # GAE Lambda曲线
            self.gae_ax.plot(self.data['episode'], self.data['gae_lambda'], 'y-')
            self.gae_ax.set_ylabel('GAE Lambda')
        
        # 调整布局
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
    def save_plots(self, output_dir='./realtime_plots'):
        """保存图表"""
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存综合图表
        plt.savefig(os.path.join(output_dir, 'realtime_dashboard.png'), dpi=150)
        
        # 保存单个图表
        plot_configs = [
            ('reward_curve.png', self.reward_ax),
            ('loss_curve.png', self.loss_ax),
            ('lr_curve.png', self.lr_ax),
            ('entropy_curve.png', self.entropy_ax),
            ('kl_curve.png', self.kl_ax),
            ('gae_curve.png', self.gae_ax)
        ]
        
        for filename, ax in plot_configs:
            # 创建临时图表
            temp_fig = plt.figure(figsize=(8, 6))
            temp_ax = temp_fig.add_subplot(111)
            
            # 复制数据
            if self.data['episode']:
                if filename == 'reward_curve.png':
                    temp_ax.plot(self.data['episode'], self.data['avg_reward'], 'b-')
                    temp_ax.set_title('平均奖励')
                    temp_ax.set_ylabel('奖励值')
                elif filename == 'loss_curve.png':
                    temp_ax.plot(self.data['episode'], self.data['train_loss'], 'r-')
                    temp_ax.set_title('训练损失')
                    temp_ax.set_ylabel('损失值')
                elif filename == 'lr_curve.png':
                    temp_ax.plot(self.data['episode'], self.data['lr'], 'g-')
                    temp_ax.set_title('学习率')
                    temp_ax.set_ylabel('学习率')
                elif filename == 'entropy_curve.png':
                    temp_ax.plot(self.data['episode'], self.data['entropy_coef'], 'm-')
                    temp_ax.set_title('熵系数')
                    temp_ax.set_ylabel('熵系数')
                elif filename == 'kl_curve.png':
                    temp_ax.plot(self.data['episode'], self.data['kl_divergence'], 'c-')
                    temp_ax.set_title('KL散度')
                    temp_ax.set_ylabel('KL散度')
                elif filename == 'gae_curve.png':
                    temp_ax.plot(self.data['episode'], self.data['gae_lambda'], 'y-')
                    temp_ax.set_title('GAE Lambda')
                    temp_ax.set_ylabel('GAE Lambda')
                
                temp_ax.set_xlabel('Episode')
                temp_fig.tight_layout()
                temp_fig.savefig(os.path.join(output_dir, filename), dpi=150)
            
            plt.close(temp_fig)
        
        print(f"图表已保存到: {output_dir}")

def monitor_training():
    """监控训练过程"""
    visualizer = RealTimeVisualizer()
    
    # 从终端输出中读取日志
    print("开始监控训练过程...")
    print("按 Ctrl+C 停止监控")
    
    # 预定义的训练数据（用于演示）
    # 实际使用时，这些数据会从日志文件或终端输出中读取
    demo_data = [
        {'episode': 0, 'total_steps': 100, 'avg_reward': -35.4877, 'train_loss': 54.7135, 'lr': 0.0003, 'entropy_coef': 0.0500, 'kl_divergence': -0.0483, 'kl_coef': 0.2000, 'gae_lambda': 0.9500},
        {'episode': 50, 'total_steps': 5100, 'avg_reward': 6.6224, 'train_loss': 3.8123, 'lr': 0.0003, 'entropy_coef': 0.0500, 'kl_divergence': -0.0787, 'kl_coef': 1.0125, 'gae_lambda': 0.9500},
        {'episode': 100, 'total_steps': 10100, 'avg_reward': 35.0577, 'train_loss': 2.7180, 'lr': 0.0003, 'entropy_coef': 0.0500, 'kl_divergence': -0.0992, 'kl_coef': 1.0125, 'gae_lambda': 0.9500},
        {'episode': 150, 'total_steps': 15100, 'avg_reward': 45.6630, 'train_loss': 4.3606, 'lr': 0.0003, 'entropy_coef': 0.0499, 'kl_divergence': -0.0452, 'kl_coef': 2.0000, 'gae_lambda': 0.9500},
        {'episode': 200, 'total_steps': 20100, 'avg_reward': -52.5039, 'train_loss': 7.0739, 'lr': 0.0003, 'entropy_coef': 0.0498, 'kl_divergence': -0.1093, 'kl_coef': 2.0000, 'gae_lambda': 0.9500},
        {'episode': 250, 'total_steps': 25100, 'avg_reward': 10.2345, 'train_loss': 5.1234, 'lr': 0.0003, 'entropy_coef': 0.0497, 'kl_divergence': -0.0876, 'kl_coef': 1.5000, 'gae_lambda': 0.9500},
        {'episode': 300, 'total_steps': 30100, 'avg_reward': 25.6789, 'train_loss': 3.4567, 'lr': 0.0003, 'entropy_coef': 0.0496, 'kl_divergence': -0.0654, 'kl_coef': 1.2000, 'gae_lambda': 0.9500},
        {'episode': 350, 'total_steps': 35100, 'avg_reward': 38.9876, 'train_loss': 2.3456, 'lr': 0.0003, 'entropy_coef': 0.0495, 'kl_divergence': -0.0432, 'kl_coef': 1.0000, 'gae_lambda': 0.9500},
        {'episode': 400, 'total_steps': 40100, 'avg_reward': 50.1234, 'train_loss': 1.2345, 'lr': 0.0003, 'entropy_coef': 0.0494, 'kl_divergence': -0.0210, 'kl_coef': 0.8000, 'gae_lambda': 0.9500},
        {'episode': 450, 'total_steps': 45100, 'avg_reward': 60.5678, 'train_loss': 0.9876, 'lr': 0.0003, 'entropy_coef': 0.0493, 'kl_divergence': -0.0100, 'kl_coef': 0.5000, 'gae_lambda': 0.9500},
    ]
    
    try:
        # 首先加载演示数据
        for data in demo_data:
            if not visualizer.data['episode'] or data['episode'] > visualizer.data['episode'][-1]:
                visualizer.update_data(data)
        
        # 更新图表
        visualizer.update_plots()
        
        # 保存图表
        visualizer.save_plots()
        
        print("演示数据已加载并生成图表")
        print("实际使用时，该工具会实时监控训练过程")
        
        # 模拟实时更新
        print("模拟实时更新中...")
        time.sleep(2)
        
        # 生成一些模拟数据
        for i in range(500, 1000, 50):
            # 模拟奖励值波动
            reward = 40 + 30 * np.sin(i/100) + np.random.normal(0, 10)
            loss = 3 + 2 * np.cos(i/150) + np.random.normal(0, 1)
            
            new_data = {
                'episode': i,
                'total_steps': i * 100 + 100,
                'avg_reward': reward,
                'train_loss': loss,
                'lr': 0.0003 * (1 - i/1000),  # 模拟学习率衰减
                'entropy_coef': max(0.001, 0.05 - i/20000),  # 模拟熵系数衰减
                'kl_divergence': -0.05 + 0.02 * np.sin(i/200),
                'kl_coef': 1.0 + 0.5 * np.cos(i/150),
                'gae_lambda': 0.95
            }
            
            visualizer.update_data(new_data)
            visualizer.update_plots()
            visualizer.save_plots()
            
            print(f"模拟更新: Episode {i}, Reward: {reward:.4f}, Loss: {loss:.4f}")
            time.sleep(0.5)
            
    except KeyboardInterrupt:
        print("监控已停止")
    except Exception as e:
        print(f"监控过程中出错: {e}")

def main():
    """主函数"""
    monitor_training()

if __name__ == '__main__':
    main()
