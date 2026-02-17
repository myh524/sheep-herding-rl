"""
演示材料生成工具
自动生成训练报告、对比图表和演示动画
"""

import argparse
import os
import sys
import json
import time
import datetime
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
from dataclasses import dataclass

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'WenQuanYi Zen Hei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


@dataclass
class TrainingData:
    episode: List[int]
    total_steps: List[int]
    avg_reward: List[float]
    train_loss: List[float]
    stage: List[int]
    success_rate: List[float]
    learning_rate: List[float]
    grad_norm: List[float]
    entropy: List[float]
    value_loss: List[float]
    policy_loss: List[float]


class DemoGenerator:
    def __init__(self, log_dir: str, output_dir: str = './demos'):
        self.log_dir = log_dir
        self.output_dir = output_dir
        self.training_data = None
        self.models_data = []
        
        os.makedirs(self.output_dir, exist_ok=True)
    
    def load_training_data(self, log_file: str):
        """加载训练数据"""
        if not os.path.exists(log_file):
            print(f"Log file not found: {log_file}")
            return False
        
        try:
            data = []
            with open(log_file, 'r') as f:
                for line in f:
                    if line.strip():
                        try:
                            entry = json.loads(line)
                            data.append(entry)
                        except json.JSONDecodeError:
                            continue
            
            if not data:
                print(f"No valid data found in {log_file}")
                return False
            
            # 提取数据
            episode = []
            total_steps = []
            avg_reward = []
            train_loss = []
            stage = []
            success_rate = []
            learning_rate = []
            grad_norm = []
            entropy = []
            value_loss = []
            policy_loss = []
            
            for entry in data:
                episode.append(entry.get('episode', 0))
                total_steps.append(entry.get('total_steps', 0))
                avg_reward.append(entry.get('avg_reward', 0.0))
                train_loss.append(entry.get('train_loss', 0.0))
                stage.append(entry.get('stage', 0))
                success_rate.append(entry.get('success_rate', 0.0))
                learning_rate.append(entry.get('learning_rate', 0.0))
                grad_norm.append(entry.get('grad_norm', 0.0))
                entropy.append(entry.get('entropy', 0.0))
                value_loss.append(entry.get('value_loss', 0.0))
                policy_loss.append(entry.get('policy_loss', 0.0))
            
            self.training_data = TrainingData(
                episode=episode,
                total_steps=total_steps,
                avg_reward=avg_reward,
                train_loss=train_loss,
                stage=stage,
                success_rate=success_rate,
                learning_rate=learning_rate,
                grad_norm=grad_norm,
                entropy=entropy,
                value_loss=value_loss,
                policy_loss=policy_loss
            )
            
            print(f"Loaded training data from {log_file}")
            return True
        except Exception as e:
            print(f"Error loading training data: {e}")
            return False
    
    def load_multiple_models(self, model_configs: List[Dict]):
        """加载多个模型的数据用于对比"""
        self.models_data = []
        
        for config in model_configs:
            model_name = config.get('name')
            log_file = config.get('log_file')
            
            if not os.path.exists(log_file):
                print(f"Log file not found for model {model_name}: {log_file}")
                continue
            
            data = []
            with open(log_file, 'r') as f:
                for line in f:
                    if line.strip():
                        try:
                            entry = json.loads(line)
                            data.append(entry)
                        except json.JSONDecodeError:
                            continue
            
            if not data:
                print(f"No valid data found for model {model_name}")
                continue
            
            # 提取数据
            episode = []
            avg_reward = []
            success_rate = []
            train_loss = []
            
            for entry in data:
                episode.append(entry.get('episode', 0))
                avg_reward.append(entry.get('avg_reward', 0.0))
                success_rate.append(entry.get('success_rate', 0.0))
                train_loss.append(entry.get('train_loss', 0.0))
            
            model_data = {
                'name': model_name,
                'episode': episode,
                'avg_reward': avg_reward,
                'success_rate': success_rate,
                'train_loss': train_loss
            }
            
            self.models_data.append(model_data)
            print(f"Loaded data for model: {model_name}")
    
    def generate_training_report(self, report_name: str = 'training_report'):
        """生成训练报告"""
        if not self.training_data:
            print("No training data loaded")
            return False
        
        report_dir = os.path.join(self.output_dir, report_name)
        os.makedirs(report_dir, exist_ok=True)
        
        # 生成各种图表
        self._generate_reward_plot(report_dir)
        self._generate_loss_plot(report_dir)
        self._generate_success_rate_plot(report_dir)
        self._generate_learning_rate_plot(report_dir)
        self._generate_grad_norm_plot(report_dir)
        self._generate_entropy_plot(report_dir)
        self._generate_comprehensive_dashboard(report_dir)
        
        # 生成HTML报告
        self._generate_html_report(report_dir)
        
        print(f"Training report generated at: {report_dir}")
        return True
    
    def _generate_reward_plot(self, output_dir: str):
        """生成奖励曲线"""
        plt.figure(figsize=(12, 6))
        plt.plot(self.training_data.episode, self.training_data.avg_reward, 'b-', linewidth=2)
        plt.title('平均奖励曲线', fontsize=14, fontweight='bold')
        plt.xlabel('Episode')
        plt.ylabel('平均奖励')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'reward_curve.png'), dpi=150)
        plt.close()
    
    def _generate_loss_plot(self, output_dir: str):
        """生成损失曲线"""
        plt.figure(figsize=(12, 6))
        plt.plot(self.training_data.episode, self.training_data.train_loss, 'r-', linewidth=2, label='总损失')
        plt.plot(self.training_data.episode, self.training_data.value_loss, 'g-', linewidth=2, label='价值损失')
        plt.plot(self.training_data.episode, self.training_data.policy_loss, 'y-', linewidth=2, label='策略损失')
        plt.title('损失曲线', fontsize=14, fontweight='bold')
        plt.xlabel('Episode')
        plt.ylabel('损失')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'loss_curve.png'), dpi=150)
        plt.close()
    
    def _generate_success_rate_plot(self, output_dir: str):
        """生成成功率曲线"""
        plt.figure(figsize=(12, 6))
        plt.plot(self.training_data.episode, self.training_data.success_rate, 'g-', linewidth=2)
        plt.title('成功率曲线', fontsize=14, fontweight='bold')
        plt.xlabel('Episode')
        plt.ylabel('成功率')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'success_rate.png'), dpi=150)
        plt.close()
    
    def _generate_learning_rate_plot(self, output_dir: str):
        """生成学习率曲线"""
        plt.figure(figsize=(12, 6))
        plt.plot(self.training_data.episode, self.training_data.learning_rate, 'm-', linewidth=2)
        plt.title('学习率曲线', fontsize=14, fontweight='bold')
        plt.xlabel('Episode')
        plt.ylabel('学习率')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'learning_rate.png'), dpi=150)
        plt.close()
    
    def _generate_grad_norm_plot(self, output_dir: str):
        """生成梯度范数曲线"""
        plt.figure(figsize=(12, 6))
        plt.plot(self.training_data.episode, self.training_data.grad_norm, 'c-', linewidth=2)
        plt.title('梯度范数曲线', fontsize=14, fontweight='bold')
        plt.xlabel('Episode')
        plt.ylabel('梯度范数')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'grad_norm.png'), dpi=150)
        plt.close()
    
    def _generate_entropy_plot(self, output_dir: str):
        """生成熵曲线"""
        plt.figure(figsize=(12, 6))
        plt.plot(self.training_data.episode, self.training_data.entropy, 'y-', linewidth=2)
        plt.title('熵曲线', fontsize=14, fontweight='bold')
        plt.xlabel('Episode')
        plt.ylabel('熵')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'entropy.png'), dpi=150)
        plt.close()
    
    def _generate_comprehensive_dashboard(self, output_dir: str):
        """生成综合仪表板"""
        fig, axes = plt.subplots(3, 2, figsize=(15, 15))
        fig.suptitle('训练综合仪表板', fontsize=16, fontweight='bold')
        
        # 奖励曲线
        ax1 = axes[0, 0]
        ax1.plot(self.training_data.episode, self.training_data.avg_reward, 'b-', linewidth=2)
        ax1.set_title('平均奖励曲线')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('平均奖励')
        ax1.grid(True, alpha=0.3)
        
        # 损失曲线
        ax2 = axes[0, 1]
        ax2.plot(self.training_data.episode, self.training_data.train_loss, 'r-', linewidth=2)
        ax2.set_title('总损失曲线')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('损失')
        ax2.grid(True, alpha=0.3)
        
        # 成功率曲线
        ax3 = axes[1, 0]
        ax3.plot(self.training_data.episode, self.training_data.success_rate, 'g-', linewidth=2)
        ax3.set_title('成功率曲线')
        ax3.set_xlabel('Episode')
        ax3.set_ylabel('成功率')
        ax3.grid(True, alpha=0.3)
        
        # 学习率曲线
        ax4 = axes[1, 1]
        ax4.plot(self.training_data.episode, self.training_data.learning_rate, 'm-', linewidth=2)
        ax4.set_title('学习率曲线')
        ax4.set_xlabel('Episode')
        ax4.set_ylabel('学习率')
        ax4.grid(True, alpha=0.3)
        
        # 梯度范数曲线
        ax5 = axes[2, 0]
        ax5.plot(self.training_data.episode, self.training_data.grad_norm, 'c-', linewidth=2)
        ax5.set_title('梯度范数曲线')
        ax5.set_xlabel('Episode')
        ax5.set_ylabel('梯度范数')
        ax5.grid(True, alpha=0.3)
        
        # 熵曲线
        ax6 = axes[2, 1]
        ax6.plot(self.training_data.episode, self.training_data.entropy, 'y-', linewidth=2)
        ax6.set_title('熵曲线')
        ax6.set_xlabel('Episode')
        ax6.set_ylabel('熵')
        ax6.grid(True, alpha=0.3)
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(os.path.join(output_dir, 'comprehensive_dashboard.png'), dpi=150)
        plt.close()
    
    def _generate_html_report(self, output_dir: str):
        """生成HTML报告"""
        html_content = f"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>训练报告</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }}
        h1, h2, h3 {{
            color: #2c3e50;
        }}
        .header {{
            background-color: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 30px;
            text-align: center;
        }}
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        .metric-card {{
            background-color: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .metric-value {{
            font-size: 24px;
            font-weight: bold;
            color: #3498db;
        }}
        .chart-container {{
            margin-bottom: 40px;
        }}
        .chart-title {{
            font-size: 18px;
            font-weight: bold;
            margin-bottom: 10px;
            text-align: center;
        }}
        img {{
            max-width: 100%;
            height: auto;
            border-radius: 4px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .footer {{
            margin-top: 50px;
            padding-top: 20px;
            border-top: 1px solid #ddd;
            text-align: center;
            color: #666;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>训练报告</h1>
        <p>生成时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        <p>总训练步数: {self.training_data.total_steps[-1] if self.training_data.total_steps else 0}</p>
        <p>总训练 episodes: {self.training_data.episode[-1] if self.training_data.episode else 0}</p>
    </div>

    <div class="metrics-grid">
        <div class="metric-card">
            <h3>最终平均奖励</h3>
            <div class="metric-value">{self.training_data.avg_reward[-1]:.2f}</div>
        </div>
        <div class="metric-card">
            <h3>最终成功率</h3>
            <div class="metric-value">{self.training_data.success_rate[-1]:.2f}%</div>
        </div>
        <div class="metric-card">
            <h3>最终损失</h3>
            <div class="metric-value">{self.training_data.train_loss[-1]:.4f}</div>
        </div>
        <div class="metric-card">
            <h3>训练时长</h3>
            <div class="metric-value">N/A</div>
        </div>
    </div>

    <h2>训练曲线</h2>
    
    <div class="chart-container">
        <div class="chart-title">平均奖励曲线</div>
        <img src="reward_curve.png" alt="奖励曲线">
    </div>

    <div class="chart-container">
        <div class="chart-title">损失曲线</div>
        <img src="loss_curve.png" alt="损失曲线">
    </div>

    <div class="chart-container">
        <div class="chart-title">成功率曲线</div>
        <img src="success_rate.png" alt="成功率曲线">
    </div>

    <div class="chart-container">
        <div class="chart-title">学习率曲线</div>
        <img src="learning_rate.png" alt="学习率曲线">
    </div>

    <div class="chart-container">
        <div class="chart-title">梯度范数曲线</div>
        <img src="grad_norm.png" alt="梯度范数曲线">
    </div>

    <div class="chart-container">
        <div class="chart-title">熵曲线</div>
        <img src="entropy.png" alt="熵曲线">
    </div>

    <div class="chart-container">
        <div class="chart-title">综合仪表板</div>
        <img src="comprehensive_dashboard.png" alt="综合仪表板">
    </div>

    <div class="footer">
        <p>训练报告由 DemoGenerator 自动生成</p>
    </div>
</body>
</html>
        """
        
        html_path = os.path.join(output_dir, 'index.html')
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"HTML report generated at: {html_path}")
    
    def generate_model_comparison(self, comparison_name: str = 'model_comparison'):
        """生成模型对比报告"""
        if not self.models_data:
            print("No models data loaded for comparison")
            return False
        
        comparison_dir = os.path.join(self.output_dir, comparison_name)
        os.makedirs(comparison_dir, exist_ok=True)
        
        # 生成对比图表
        self._generate_reward_comparison(comparison_dir)
        self._generate_loss_comparison(comparison_dir)
        self._generate_success_rate_comparison(comparison_dir)
        self._generate_comparison_table(comparison_dir)
        self._generate_radar_chart(comparison_dir)
        
        # 生成HTML对比报告
        self._generate_comparison_html_report(comparison_dir)
        
        print(f"Model comparison generated at: {comparison_dir}")
        return True
    
    def _generate_reward_comparison(self, output_dir: str):
        """生成奖励对比曲线"""
        plt.figure(figsize=(12, 6))
        for model_data in self.models_data:
            plt.plot(model_data['episode'], model_data['avg_reward'], linewidth=2, label=model_data['name'])
        plt.title('模型奖励对比', fontsize=14, fontweight='bold')
        plt.xlabel('Episode')
        plt.ylabel('平均奖励')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'reward_comparison.png'), dpi=150)
        plt.close()
    
    def _generate_loss_comparison(self, output_dir: str):
        """生成损失对比曲线"""
        plt.figure(figsize=(12, 6))
        for model_data in self.models_data:
            plt.plot(model_data['episode'], model_data['train_loss'], linewidth=2, label=model_data['name'])
        plt.title('模型损失对比', fontsize=14, fontweight='bold')
        plt.xlabel('Episode')
        plt.ylabel('损失')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'loss_comparison.png'), dpi=150)
        plt.close()
    
    def _generate_success_rate_comparison(self, output_dir: str):
        """生成成功率对比曲线"""
        plt.figure(figsize=(12, 6))
        for model_data in self.models_data:
            plt.plot(model_data['episode'], model_data['success_rate'], linewidth=2, label=model_data['name'])
        plt.title('模型成功率对比', fontsize=14, fontweight='bold')
        plt.xlabel('Episode')
        plt.ylabel('成功率')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'success_rate_comparison.png'), dpi=150)
        plt.close()
    
    def _generate_comparison_table(self, output_dir: str):
        """生成对比表格"""
        data = []
        for model_data in self.models_data:
            final_reward = model_data['avg_reward'][-1] if model_data['avg_reward'] else 0
            final_loss = model_data['train_loss'][-1] if model_data['train_loss'] else 0
            final_success_rate = model_data['success_rate'][-1] if model_data['success_rate'] else 0
            
            data.append({
                '模型名称': model_data['name'],
                '最终平均奖励': final_reward,
                '最终损失': final_loss,
                '最终成功率': final_success_rate
            })
        
        df = pd.DataFrame(data)
        
        # 生成表格图片
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.axis('tight')
        ax.axis('off')
        table = ax.table(cellText=df.values, colLabels=df.columns, cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.2)
        plt.title('模型性能对比', fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'comparison_table.png'), dpi=150, bbox_inches='tight')
        plt.close()
    
    def _generate_radar_chart(self, output_dir: str):
        """生成雷达图对比"""
        # 准备数据
        metrics = ['最终平均奖励', '最终成功率', '训练稳定性']
        
        data = []
        model_names = []
        
        for model_data in self.models_data:
            final_reward = model_data['avg_reward'][-1] if model_data['avg_reward'] else 0
            final_success_rate = model_data['success_rate'][-1] if model_data['success_rate'] else 0
            
            # 计算训练稳定性（奖励曲线的标准差的倒数）
            if len(model_data['avg_reward']) > 1:
                stability = 1.0 / (np.std(model_data['avg_reward']) + 0.001)
            else:
                stability = 0
            
            data.append([final_reward, final_success_rate, stability])
            model_names.append(model_data['name'])
        
        # 归一化数据
        data = np.array(data)
        for i in range(data.shape[1]):
            if np.max(data[:, i]) > 0:
                data[:, i] = data[:, i] / np.max(data[:, i])
        
        # 生成雷达图
        angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]
        
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
        
        for i, model_name in enumerate(model_names):
            values = data[i].tolist()
            values += values[:1]
            ax.plot(angles, values, linewidth=2, label=model_name)
            ax.fill(angles, values, alpha=0.25)
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics)
        ax.set_title('模型性能雷达图', fontsize=14, fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'radar_chart.png'), dpi=150)
        plt.close()
    
    def _generate_comparison_html_report(self, output_dir: str):
        """生成对比HTML报告"""
        html_content = f"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>模型对比报告</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }}
        h1, h2, h3 {{
            color: #2c3e50;
        }}
        .header {{
            background-color: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 30px;
            text-align: center;
        }}
        .chart-container {{
            margin-bottom: 40px;
        }}
        .chart-title {{
            font-size: 18px;
            font-weight: bold;
            margin-bottom: 10px;
            text-align: center;
        }}
        img {{
            max-width: 100%;
            height: auto;
            border-radius: 4px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .footer {{
            margin-top: 50px;
            padding-top: 20px;
            border-top: 1px solid #ddd;
            text-align: center;
            color: #666;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>模型对比报告</h1>
        <p>生成时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        <p>对比模型数量: {len(self.models_data)}</p>
    </div>

    <h2>性能对比</h2>
    
    <div class="chart-container">
        <div class="chart-title">奖励对比曲线</div>
        <img src="reward_comparison.png" alt="奖励对比曲线">
    </div>

    <div class="chart-container">
        <div class="chart-title">损失对比曲线</div>
        <img src="loss_comparison.png" alt="损失对比曲线">
    </div>

    <div class="chart-container">
        <div class="chart-title">成功率对比曲线</div>
        <img src="success_rate_comparison.png" alt="成功率对比曲线">
    </div>

    <div class="chart-container">
        <div class="chart-title">性能对比表格</div>
        <img src="comparison_table.png" alt="性能对比表格">
    </div>

    <div class="chart-container">
        <div class="chart-title">雷达图对比</div>
        <img src="radar_chart.png" alt="雷达图对比">
    </div>

    <div class="footer">
        <p>对比报告由 DemoGenerator 自动生成</p>
    </div>
</body>
</html>
        """
        
        html_path = os.path.join(output_dir, 'index.html')
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"Comparison HTML report generated at: {html_path}")
    
    def generate_interactive_demo(self, demo_name: str = 'interactive_demo'):
        """生成交互式演示"""
        demo_dir = os.path.join(self.output_dir, demo_name)
        os.makedirs(demo_dir, exist_ok=True)
        
        # 生成交互式HTML
        html_content = f"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>交互式演示</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body {{
            font-family: Arial, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }}
        h1, h2, h3 {{
            color: #2c3e50;
        }}
        .header {{
            background-color: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 30px;
            text-align: center;
        }}
        .control-panel {{
            background-color: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 30px;
        }}
        .control-group {{
            margin-bottom: 15px;
        }}
        label {{
            display: inline-block;
            width: 150px;
            font-weight: bold;
        }}
        input[type="range"] {{
            width: 200px;
        }}
        .value-display {{
            display: inline-block;
            width: 60px;
            text-align: right;
            margin-left: 10px;
        }}
        .chart-container {{
            margin-bottom: 40px;
            position: relative;
            height: 400px;
        }}
        .footer {{
            margin-top: 50px;
            padding-top: 20px;
            border-top: 1px solid #ddd;
            text-align: center;
            color: #666;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>交互式演示</h1>
        <p>调整参数以查看不同设置下的模型性能</p>
    </div>

    <div class="control-panel">
        <h3>参数控制</h3>
        <div class="control-group">
            <label for="learning-rate">学习率:</label>
            <input type="range" id="learning-rate" min="0.0001" max="0.01" step="0.0001" value="0.0003">
            <span class="value-display" id="learning-rate-value">0.0003</span>
        </div>
        <div class="control-group">
            <label for="batch-size">批量大小:</label>
            <input type="range" id="batch-size" min="1" max="1024" step="1" value="256">
            <span class="value-display" id="batch-size-value">256</span>
        </div>
        <div class="control-group">
            <label for="gamma">折扣因子:</label>
            <input type="range" id="gamma" min="0.8" max="0.999" step="0.001" value="0.99">
            <span class="value-display" id="gamma-value">0.99</span>
        </div>
        <div class="control-group">
            <label for="gae-lambda">GAE Lambda:</label>
            <input type="range" id="gae-lambda" min="0.8" max="1.0" step="0.01" value="0.95">
            <span class="value-display" id="gae-lambda-value">0.95</span>
        </div>
    </div>

    <div class="chart-container">
        <canvas id="reward-chart"></canvas>
    </div>

    <div class="footer">
        <p>交互式演示由 DemoGenerator 自动生成</p>
    </div>

    <script>
        // 初始化图表
        const ctx = document.getElementById('reward-chart').getContext('2d');
        const rewardChart = new Chart(ctx, {{
            type: 'line',
            data: {{
                labels: Array.from({0}),
                datasets: [{{
                    label: '奖励曲线',
                    data: {1},
                    borderColor: 'rgb(75, 192, 192)',
                    tension: 0.1
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                scales: {{
                    y: {{
                        beginAtZero: true,
                        title: {{
                            display: true,
                            text: '平均奖励'
                        }}
                    }},
                    x: {{
                        title: {{
                            display: true,
                            text: 'Episode'
                        }}
                    }}
                }}
            }}
        }});

        // 添加事件监听器
        document.getElementById('learning-rate').addEventListener('input', function(e) {{
            document.getElementById('learning-rate-value').textContent = e.target.value;
            updateChart();
        }});

        document.getElementById('batch-size').addEventListener('input', function(e) {{
            document.getElementById('batch-size-value').textContent = e.target.value;
            updateChart();
        }});

        document.getElementById('gamma').addEventListener('input', function(e) {{
            document.getElementById('gamma-value').textContent = e.target.value;
            updateChart();
        }});

        document.getElementById('gae-lambda').addEventListener('input', function(e) {{
            document.getElementById('gae-lambda-value').textContent = e.target.value;
            updateChart();
        }});

        // 更新图表
        function updateChart() {{
            // 这里只是模拟，实际应用中应该根据参数重新计算或加载数据
            const learningRate = parseFloat(document.getElementById('learning-rate').value);
            const batchSize = parseInt(document.getElementById('batch-size').value);
            const gamma = parseFloat(document.getElementById('gamma').value);
            const gaeLambda = parseFloat(document.getElementById('gae-lambda').value);

            // 模拟数据变化
            const baseData = {1};
            const scaledData = baseData.map(value => {{
                // 简单的模拟，实际应该更复杂
                let scale = 1.0;
                scale *= (learningRate / 0.0003);
                scale *= (batchSize / 256);
                scale *= (gamma / 0.99);
                scale *= (gaeLambda / 0.95);
                return value * scale;
            }});

            rewardChart.data.datasets[0].data = scaledData;
            rewardChart.update();
        }}
    </script>
</body>
</html>
        """
        
        # 准备数据
        episodes = list(range(len(self.training_data.episode))) if self.training_data else []
        rewards = self.training_data.avg_reward if self.training_data else []
        
        html_content = html_content.format(json.dumps(episodes), json.dumps(rewards))
        
        html_path = os.path.join(demo_dir, 'index.html')
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"Interactive demo generated at: {html_path}")
        return True


def parse_args():
    parser = argparse.ArgumentParser(description='演示材料生成工具')
    
    parser.add_argument('--log_dir', type=str, default='./logs',
                        help='训练日志目录')
    parser.add_argument('--output_dir', type=str, default='./demos',
                        help='输出目录')
    parser.add_argument('--log_file', type=str, default=None,
                        help='训练日志文件路径')
    parser.add_argument('--generate_report', action='store_true', default=False,
                        help='生成训练报告')
    parser.add_argument('--generate_comparison', action='store_true', default=False,
                        help='生成模型对比')
    parser.add_argument('--generate_interactive', action='store_true', default=False,
                        help='生成交互式演示')
    parser.add_argument('--model_configs', type=str, default=None,
                        help='模型配置文件路径 (JSON格式)')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    demo_generator = DemoGenerator(
        log_dir=args.log_dir,
        output_dir=args.output_dir
    )
    
    if args.log_file:
        demo_generator.load_training_data(args.log_file)
    
    if args.generate_report:
        demo_generator.generate_training_report()
    
    if args.generate_comparison and args.model_configs:
        with open(args.model_configs, 'r') as f:
            model_configs = json.load(f)
        demo_generator.load_multiple_models(model_configs)
        demo_generator.generate_model_comparison()
    
    if args.generate_interactive:
        demo_generator.generate_interactive_demo()


if __name__ == '__main__':
    main()
