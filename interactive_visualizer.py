#!/usr/bin/env python3
"""
交互式训练可视化界面
支持参数调节、实时效果展示和多维度分析
"""

import os
import sys
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.widgets import Button, Slider, RadioButtons, CheckButtons
from matplotlib.patches import Rectangle
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from collections import deque
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'WenQuanYi Zen Hei', 'DejaVu Sans', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False


class InteractiveTrainingVisualizer:
    """交互式训练可视化器"""
    
    def __init__(self, max_history: int = 2000):
        self.max_history = max_history
        
        # 数据存储
        self.data: Dict[str, List[float]] = {
            'episode': [],
            'total_steps': [],
            'avg_reward': [],
            'train_loss': [],
            'value_loss': [],
            'policy_loss': [],
            'entropy': [],
            'grad_norm': [],
            'lr': [],
            'entropy_coef': [],
            'kl_divergence': [],
            'kl_coef': [],
            'gae_lambda': [],
            'value_pred_error': [],
        }
        
        # 可视化参数
        self.smooth_window = 10
        self.show_raw_data = True
        self.show_smoothed = True
        self.show_thresholds = True
        self.selected_metrics = ['avg_reward', 'train_loss', 'grad_norm', 'entropy']
        
        # 阈值设置
        self.thresholds = {
            'grad_norm_max': 100.0,
            'entropy_min': 0.001,
            'kl_divergence_max': 0.1,
        }
        
        # 视图范围
        self.view_start = 0
        self.view_end = -1  # -1 表示显示全部
        
        # 图表对象
        self.fig = None
        self.axes = {}
        self.lines = {}
        
        # 状态
        self.paused = False
        self.auto_scroll = True
        
    def load_data(self, data_source: str) -> bool:
        """加载数据"""
        if data_source.endswith('.json'):
            return self._load_from_json(data_source)
        elif data_source.endswith('.txt'):
            return self._load_from_log(data_source)
        return False
        
    def _load_from_json(self, json_path: str) -> bool:
        """从JSON文件加载"""
        try:
            with open(json_path, 'r') as f:
                raw_data = json.load(f)
                
            if isinstance(raw_data, list):
                for key in self.data.keys():
                    self.data[key] = [d.get(key, 0.0) for d in raw_data]
            else:
                self.data = raw_data
                
            return True
        except Exception as e:
            print(f"加载JSON失败: {e}")
            return False
            
    def _load_from_log(self, log_path: str) -> bool:
        """从日志文件加载"""
        try:
            with open(log_path, 'r') as f:
                lines = f.readlines()
                
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
                            
                for key in self.data.keys():
                    self.data[key].append(metrics.get(key, 0.0))
                    
            return True
        except Exception as e:
            print(f"加载日志失败: {e}")
            return False
            
    def smooth_data(self, data: List[float], window: int) -> np.ndarray:
        """平滑数据"""
        if len(data) < window:
            return np.array(data)
        data_array = np.array(data)
        kernel = np.ones(window) / window
        smoothed = np.convolve(data_array, kernel, mode='valid')
        padding = np.full(window - 1, data_array[0])
        for i in range(window - 1):
            padding[i] = np.mean(data_array[:i+1])
        return np.concatenate([padding, smoothed])
        
    def get_view_data(self, data: List[float]) -> Tuple[List[float], List[float]]:
        """获取视图范围内的数据"""
        episodes = self.data['episode']
        if not episodes:
            return [], []
            
        start = self.view_start if self.view_start >= 0 else 0
        end = self.view_end if self.view_end > 0 else len(episodes)
        
        return episodes[start:end], data[start:end]
        
    def setup_interface(self):
        """设置交互界面"""
        self.fig = plt.figure(figsize=(20, 14))
        
        # 创建网格布局
        gs = gridspec.GridSpec(5, 4, figure=self.fig, 
                              height_ratios=[1, 1, 1, 1, 0.3],
                              width_ratios=[1, 1, 1, 0.3],
                              hspace=0.35, wspace=0.3)
        
        self.fig.suptitle('交互式训练可视化界面', fontsize=16, fontweight='bold')
        
        # 创建主图表区域
        self._create_main_plots(gs)
        
        # 创建控制面板
        self._create_control_panel(gs)
        
        # 创建统计面板
        self._create_stats_panel(gs)
        
    def _create_main_plots(self, gs):
        """创建主图表区域"""
        # 奖励曲线
        self.axes['reward'] = self.fig.add_subplot(gs[0, 0])
        self.axes['reward'].set_title('奖励曲线')
        self.axes['reward'].set_xlabel('Episode')
        self.axes['reward'].set_ylabel('奖励值')
        self.axes['reward'].grid(True, alpha=0.3)
        
        # 损失曲线
        self.axes['loss'] = self.fig.add_subplot(gs[0, 1])
        self.axes['loss'].set_title('损失曲线')
        self.axes['loss'].set_xlabel('Episode')
        self.axes['loss'].set_ylabel('损失值')
        self.axes['loss'].grid(True, alpha=0.3)
        
        # 梯度范数
        self.axes['grad_norm'] = self.fig.add_subplot(gs[0, 2])
        self.axes['grad_norm'].set_title('梯度范数')
        self.axes['grad_norm'].set_xlabel('Episode')
        self.axes['grad_norm'].set_ylabel('范数值')
        self.axes['grad_norm'].grid(True, alpha=0.3)
        
        # 学习率
        self.axes['lr'] = self.fig.add_subplot(gs[1, 0])
        self.axes['lr'].set_title('学习率')
        self.axes['lr'].set_xlabel('Episode')
        self.axes['lr'].set_ylabel('学习率')
        self.axes['lr'].set_yscale('log')
        self.axes['lr'].grid(True, alpha=0.3)
        
        # 策略熵
        self.axes['entropy'] = self.fig.add_subplot(gs[1, 1])
        self.axes['entropy'].set_title('策略熵')
        self.axes['entropy'].set_xlabel('Episode')
        self.axes['entropy'].set_ylabel('熵值')
        self.axes['entropy'].grid(True, alpha=0.3)
        
        # KL散度
        self.axes['kl'] = self.fig.add_subplot(gs[1, 2])
        self.axes['kl'].set_title('KL散度')
        self.axes['kl'].set_xlabel('Episode')
        self.axes['kl'].set_ylabel('KL散度')
        self.axes['kl'].grid(True, alpha=0.3)
        
        # 损失分解
        self.axes['loss_breakdown'] = self.fig.add_subplot(gs[2, 0])
        self.axes['loss_breakdown'].set_title('损失分解')
        self.axes['loss_breakdown'].set_xlabel('Episode')
        self.axes['loss_breakdown'].set_ylabel('损失值')
        self.axes['loss_breakdown'].grid(True, alpha=0.3)
        
        # GAE Lambda
        self.axes['gae'] = self.fig.add_subplot(gs[2, 1])
        self.axes['gae'].set_title('GAE Lambda')
        self.axes['gae'].set_xlabel('Episode')
        self.axes['gae'].set_ylabel('Lambda值')
        self.axes['gae'].grid(True, alpha=0.3)
        
        # 熵系数
        self.axes['entropy_coef'] = self.fig.add_subplot(gs[2, 2])
        self.axes['entropy_coef'].set_title('熵系数')
        self.axes['entropy_coef'].set_xlabel('Episode')
        self.axes['entropy_coef'].set_ylabel('系数值')
        self.axes['entropy_coef'].grid(True, alpha=0.3)
        
        # 价值预测误差
        self.axes['value_error'] = self.fig.add_subplot(gs[3, 0])
        self.axes['value_error'].set_title('价值预测误差')
        self.axes['value_error'].set_xlabel('Episode')
        self.axes['value_error'].set_ylabel('误差值')
        self.axes['value_error'].grid(True, alpha=0.3)
        
        # 奖励分布直方图
        self.axes['reward_hist'] = self.fig.add_subplot(gs[3, 1])
        self.axes['reward_hist'].set_title('奖励分布')
        self.axes['reward_hist'].set_xlabel('奖励值')
        self.axes['reward_hist'].set_ylabel('频次')
        self.axes['reward_hist'].grid(True, alpha=0.3)
        
        # 梯度分布直方图
        self.axes['grad_hist'] = self.fig.add_subplot(gs[3, 2])
        self.axes['grad_hist'].set_title('梯度分布')
        self.axes['grad_hist'].set_xlabel('梯度范数')
        self.axes['grad_hist'].set_ylabel('频次')
        self.axes['grad_hist'].grid(True, alpha=0.3)
        
    def _create_control_panel(self, gs):
        """创建控制面板"""
        # 控制面板区域
        ax_control = self.fig.add_subplot(gs[0:2, 3])
        ax_control.axis('off')
        ax_control.set_title('控制面板', fontsize=12, fontweight='bold')
        
        # 平滑窗口滑块
        ax_smooth = plt.axes([0.78, 0.85, 0.18, 0.03])
        self.slider_smooth = Slider(ax_smooth, '平滑窗口', 1, 50, 
                                    valinit=self.smooth_window, valstep=1)
        self.slider_smooth.on_changed(self._on_smooth_change)
        
        # 视图范围滑块
        ax_view_start = plt.axes([0.78, 0.80, 0.18, 0.03])
        self.slider_view_start = Slider(ax_view_start, '起始Episode', 0, 
                                        len(self.data['episode']) - 1 if self.data['episode'] else 0,
                                        valinit=0, valstep=1)
        self.slider_view_start.on_changed(self._on_view_change)
        
        ax_view_end = plt.axes([0.78, 0.75, 0.18, 0.03])
        self.slider_view_end = Slider(ax_view_end, '结束Episode', 0,
                                      len(self.data['episode']) - 1 if self.data['episode'] else 0,
                                      valinit=len(self.data['episode']) - 1 if self.data['episode'] else 0,
                                      valstep=1)
        self.slider_view_end.on_changed(self._on_view_change)
        
        # 梯度阈值滑块
        ax_grad_thresh = plt.axes([0.78, 0.70, 0.18, 0.03])
        self.slider_grad_thresh = Slider(ax_grad_thresh, '梯度阈值', 1, 500,
                                         valinit=self.thresholds['grad_norm_max'])
        self.slider_grad_thresh.on_changed(self._on_threshold_change)
        
        # 熵阈值滑块
        ax_entropy_thresh = plt.axes([0.78, 0.65, 0.18, 0.03])
        self.slider_entropy_thresh = Slider(ax_entropy_thresh, '熵阈值', 0.0001, 0.1,
                                            valinit=self.thresholds['entropy_min'])
        self.slider_entropy_thresh.on_changed(self._on_threshold_change)
        
        # 显示选项复选框
        ax_check = plt.axes([0.78, 0.50, 0.18, 0.12])
        self.check_buttons = CheckButtons(ax_check, 
                                          ['显示原始数据', '显示平滑曲线', '显示阈值线'],
                                          [True, True, True])
        self.check_buttons.on_clicked(self._on_check_change)
        
        # 按钮区域
        ax_reset = plt.axes([0.78, 0.42, 0.08, 0.04])
        self.btn_reset = Button(ax_reset, '重置视图')
        self.btn_reset.on_clicked(self._on_reset)
        
        ax_auto = plt.axes([0.88, 0.42, 0.08, 0.04])
        self.btn_auto = Button(ax_auto, '自动滚动')
        self.btn_auto.on_clicked(self._on_auto_scroll)
        
        ax_save = plt.axes([0.78, 0.36, 0.18, 0.04])
        self.btn_save = Button(ax_save, '保存图表')
        self.btn_save.on_clicked(self._on_save)
        
        # 指标选择区域
        ax_radio = plt.axes([0.78, 0.15, 0.18, 0.18])
        ax_radio.set_title('主要指标', fontsize=10)
        self.radio_metrics = RadioButtons(ax_radio, 
                                          ['奖励', '损失', '梯度', '熵'],
                                          active=0)
        self.radio_metrics.on_clicked(self._on_metric_select)
        
    def _create_stats_panel(self, gs):
        """创建统计面板"""
        ax_stats = self.fig.add_subplot(gs[3, 3])
        ax_stats.axis('off')
        ax_stats.set_title('统计摘要', fontsize=10, fontweight='bold')
        
        self.stats_text = ax_stats.text(0.1, 0.9, '', transform=ax_stats.transAxes,
                                        fontsize=9, verticalalignment='top',
                                        fontfamily='monospace')
        
    def _on_smooth_change(self, val):
        """平滑窗口变化回调"""
        self.smooth_window = int(val)
        self.update_plots()
        
    def _on_view_change(self, val):
        """视图范围变化回调"""
        self.view_start = int(self.slider_view_start.val)
        self.view_end = int(self.slider_view_end.val)
        self.auto_scroll = False
        self.update_plots()
        
    def _on_threshold_change(self, val):
        """阈值变化回调"""
        self.thresholds['grad_norm_max'] = self.slider_grad_thresh.val
        self.thresholds['entropy_min'] = self.slider_entropy_thresh.val
        self.update_plots()
        
    def _on_check_change(self, label):
        """复选框变化回调"""
        if label == '显示原始数据':
            self.show_raw_data = not self.show_raw_data
        elif label == '显示平滑曲线':
            self.show_smoothed = not self.show_smoothed
        elif label == '显示阈值线':
            self.show_thresholds = not self.show_thresholds
        self.update_plots()
        
    def _on_reset(self, event):
        """重置视图"""
        self.view_start = 0
        self.view_end = -1
        self.smooth_window = 10
        self.slider_smooth.set_val(10)
        if self.data['episode']:
            self.slider_view_start.set_val(0)
            self.slider_view_end.set_val(len(self.data['episode']) - 1)
        self.update_plots()
        
    def _on_auto_scroll(self, event):
        """切换自动滚动"""
        self.auto_scroll = not self.auto_scroll
        if self.auto_scroll:
            self.view_end = -1
            
    def _on_save(self, event):
        """保存图表"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = './interactive_plots'
        os.makedirs(output_dir, exist_ok=True)
        
        self.fig.savefig(os.path.join(output_dir, f'interactive_visualization_{timestamp}.png'),
                        dpi=150, bbox_inches='tight')
        print(f"图表已保存到: {output_dir}")
        
    def _on_metric_select(self, label):
        """指标选择回调"""
        metric_map = {
            '奖励': 'avg_reward',
            '损失': 'train_loss',
            '梯度': 'grad_norm',
            '熵': 'entropy',
        }
        self.selected_metrics[0] = metric_map.get(label, 'avg_reward')
        self.update_plots()
        
    def update_plots(self):
        """更新所有图表"""
        if not self.data['episode']:
            return
            
        episodes = self.data['episode']
        start = self.view_start if self.view_start >= 0 else 0
        end = self.view_end if self.view_end > 0 else len(episodes)
        view_episodes = episodes[start:end]
        
        # 更新奖励曲线
        self._update_plot('reward', view_episodes, self.data['avg_reward'][start:end],
                         '奖励曲线', '奖励值', 'b')
        
        # 更新损失曲线
        self._update_plot('loss', view_episodes, self.data['train_loss'][start:end],
                         '损失曲线', '损失值', 'r')
        
        # 更新梯度范数
        self._update_plot('grad_norm', view_episodes, self.data['grad_norm'][start:end],
                         '梯度范数', '范数值', 'm', 
                         threshold=self.thresholds['grad_norm_max'] if self.show_thresholds else None)
        
        # 更新学习率
        self._update_plot('lr', view_episodes, self.data['lr'][start:end],
                         '学习率', '学习率', 'g')
        
        # 更新策略熵
        self._update_plot('entropy', view_episodes, self.data['entropy'][start:end],
                         '策略熵', '熵值', 'c',
                         threshold=self.thresholds['entropy_min'] if self.show_thresholds else None,
                         threshold_position='min')
        
        # 更新KL散度
        self._update_plot('kl', view_episodes, self.data['kl_divergence'][start:end],
                         'KL散度', 'KL散度', 'orange')
        
        # 更新损失分解
        self.axes['loss_breakdown'].clear()
        self.axes['loss_breakdown'].set_title('损失分解')
        self.axes['loss_breakdown'].set_xlabel('Episode')
        self.axes['loss_breakdown'].set_ylabel('损失值')
        self.axes['loss_breakdown'].grid(True, alpha=0.3)
        if self.data['value_loss']:
            self.axes['loss_breakdown'].plot(view_episodes, 
                                            self.data['value_loss'][start:end], 
                                            'b-', alpha=0.7, label='价值损失')
        if self.data['policy_loss']:
            self.axes['loss_breakdown'].plot(view_episodes,
                                            self.data['policy_loss'][start:end],
                                            'r-', alpha=0.7, label='策略损失')
        self.axes['loss_breakdown'].legend(loc='upper right', fontsize=8)
        
        # 更新GAE Lambda
        self._update_plot('gae', view_episodes, self.data['gae_lambda'][start:end],
                         'GAE Lambda', 'Lambda值', 'purple')
        
        # 更新熵系数
        self._update_plot('entropy_coef', view_episodes, self.data['entropy_coef'][start:end],
                         '熵系数', '系数值', 'brown')
        
        # 更新价值预测误差
        self._update_plot('value_error', view_episodes, self.data['value_pred_error'][start:end],
                         '价值预测误差', '误差值', 'teal')
        
        # 更新奖励分布直方图
        self.axes['reward_hist'].clear()
        self.axes['reward_hist'].set_title('奖励分布')
        self.axes['reward_hist'].set_xlabel('奖励值')
        self.axes['reward_hist'].set_ylabel('频次')
        self.axes['reward_hist'].grid(True, alpha=0.3)
        if self.data['avg_reward']:
            self.axes['reward_hist'].hist(self.data['avg_reward'][start:end], bins=30, 
                                          color='blue', alpha=0.7, edgecolor='black')
        
        # 更新梯度分布直方图
        self.axes['grad_hist'].clear()
        self.axes['grad_hist'].set_title('梯度分布')
        self.axes['grad_hist'].set_xlabel('梯度范数')
        self.axes['grad_hist'].set_ylabel('频次')
        self.axes['grad_hist'].grid(True, alpha=0.3)
        if self.data['grad_norm']:
            self.axes['grad_hist'].hist(self.data['grad_norm'][start:end], bins=30,
                                        color='magenta', alpha=0.7, edgecolor='black')
        
        # 更新统计面板
        self._update_stats_panel()
        
        self.fig.canvas.draw_idle()
        
    def _update_plot(self, ax_name: str, episodes: List[float], data: List[float],
                    title: str, ylabel: str, color: str,
                    threshold: Optional[float] = None, threshold_position: str = 'max'):
        """更新单个图表"""
        ax = self.axes[ax_name]
        ax.clear()
        ax.set_title(title)
        ax.set_xlabel('Episode')
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        
        if not data:
            return
            
        # 绘制原始数据
        if self.show_raw_data:
            ax.plot(episodes, data, color=color, alpha=0.3, label='原始')
            
        # 绘制平滑曲线
        if self.show_smoothed and len(data) > self.smooth_window:
            smoothed = self.smooth_data(data, self.smooth_window)
            ax.plot(episodes, smoothed, color=color, linewidth=2, 
                   label=f'平滑(w={self.smooth_window})')
            
        # 绘制阈值线
        if threshold is not None and self.show_thresholds:
            if threshold_position == 'max':
                ax.axhline(y=threshold, color='r', linestyle='--', alpha=0.5, label=f'阈值: {threshold:.2f}')
            else:
                ax.axhline(y=threshold, color='r', linestyle='--', alpha=0.5, label=f'阈值: {threshold:.4f}')
                
        ax.legend(loc='best', fontsize=8)
        
    def _update_stats_panel(self):
        """更新统计面板"""
        if not self.data['episode']:
            return
            
        start = self.view_start if self.view_start >= 0 else 0
        end = self.view_end if self.view_end > 0 else len(self.data['episode'])
        
        rewards = self.data['avg_reward'][start:end]
        losses = self.data['train_loss'][start:end]
        grads = self.data['grad_norm'][start:end]
        
        stats_text = f"""
Episodes: {end - start}
        
奖励:
  均值: {np.mean(rewards):.2f}
  标准差: {np.std(rewards):.2f}
  最大: {np.max(rewards):.2f}
  最小: {np.min(rewards):.2f}
  
损失:
  均值: {np.mean(losses):.4f}
  最终: {losses[-1]:.4f}
  
梯度:
  均值: {np.mean(grads):.4f}
  最大: {np.max(grads):.4f}
        """
        
        self.stats_text.set_text(stats_text)
        
    def run(self):
        """运行交互界面"""
        self.setup_interface()
        self.update_plots()
        plt.show()


def main():
    parser = argparse.ArgumentParser(description='交互式训练可视化界面')
    parser.add_argument('--data_file', type=str, required=True,
                       help='训练数据文件路径 (JSON或TXT)')
    
    args = parser.parse_args()
    
    visualizer = InteractiveTrainingVisualizer()
    
    if not visualizer.load_data(args.data_file):
        print("加载数据失败")
        return
        
    print(f"已加载 {len(visualizer.data['episode'])} 条训练记录")
    print("\n交互说明:")
    print("  - 使用滑块调整平滑窗口和视图范围")
    print("  - 使用复选框切换显示选项")
    print("  - 点击'重置视图'恢复默认设置")
    print("  - 点击'保存图表'保存当前视图")
    print("\n启动交互界面...")
    
    visualizer.run()


if __name__ == '__main__':
    main()
