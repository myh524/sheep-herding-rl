#!/usr/bin/env python3
"""
增强版实时训练监控工具
支持多指标同步监控、异常检测和丰富的可视化界面
"""

import os
import sys
import time
import re
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.widgets import Button, Slider
from collections import deque
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'WenQuanYi Zen Hei', 'DejaVu Sans', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False


class AnomalyType(Enum):
    """异常类型枚举"""
    GRADIENT_EXPLOSION = "梯度爆炸"
    GRADIENT_VANISHING = "梯度消失"
    LOSS_SPIKE = "损失尖峰"
    LOSS_NAN = "损失NaN"
    REWARD_COLLAPSE = "奖励崩溃"
    ENTROPY_COLLAPSE = "熵崩溃"
    KL_DIVERGENCE_HIGH = "KL散度过高"
    VALUE_PREDICTION_ERROR = "价值预测误差过大"
    LEARNING_RATE_TOO_LOW = "学习率过低"


@dataclass
class AnomalyEvent:
    """异常事件"""
    anomaly_type: AnomalyType
    episode: int
    value: float
    threshold: float
    message: str
    timestamp: str = field(default_factory=lambda: datetime.now().strftime('%H:%M:%S'))
    severity: str = "warning"  # warning, error, critical


@dataclass
class TrainingMetrics:
    """训练指标数据结构"""
    episode: int = 0
    total_steps: int = 0
    avg_reward: float = 0.0
    train_loss: float = 0.0
    value_loss: float = 0.0
    policy_loss: float = 0.0
    entropy: float = 0.0
    grad_norm: float = 0.0
    lr: float = 0.0
    entropy_coef: float = 0.0
    kl_divergence: float = 0.0
    kl_coef: float = 0.0
    gae_lambda: float = 0.0
    value_pred_error: float = 0.0
    stage: int = 0
    success_rate: float = 0.0


class AnomalyDetector:
    """训练异常检测器"""
    
    def __init__(self):
        self.history: List[TrainingMetrics] = []
        self.anomalies: List[AnomalyEvent] = []
        
        # 阈值配置
        self.thresholds = {
            'grad_norm_max': 100.0,          # 梯度范数最大值
            'grad_norm_min': 1e-6,           # 梯度范数最小值
            'loss_spike_ratio': 5.0,         # 损失尖峰比率
            'reward_min': -200.0,            # 奖励最小值
            'entropy_min': 0.001,            # 熵最小值
            'kl_divergence_max': 0.1,        # KL散度最大值
            'value_pred_error_max': 10.0,    # 价值预测误差最大值
            'lr_min': 1e-6,                  # 学习率最小值
        }
        
        # 历史窗口大小
        self.window_size = 10
        
    def update(self, metrics: TrainingMetrics) -> List[AnomalyEvent]:
        """更新指标并检测异常"""
        self.history.append(metrics)
        new_anomalies = []
        
        # 检测梯度爆炸
        if metrics.grad_norm > self.thresholds['grad_norm_max']:
            anomaly = AnomalyEvent(
                anomaly_type=AnomalyType.GRADIENT_EXPLOSION,
                episode=metrics.episode,
                value=metrics.grad_norm,
                threshold=self.thresholds['grad_norm_max'],
                message=f"梯度范数 {metrics.grad_norm:.4f} 超过阈值 {self.thresholds['grad_norm_max']}",
                severity="error"
            )
            new_anomalies.append(anomaly)
            
        # 检测梯度消失
        if metrics.grad_norm < self.thresholds['grad_norm_min'] and len(self.history) > 5:
            anomaly = AnomalyEvent(
                anomaly_type=AnomalyType.GRADIENT_VANISHING,
                episode=metrics.episode,
                value=metrics.grad_norm,
                threshold=self.thresholds['grad_norm_min'],
                message=f"梯度范数 {metrics.grad_norm:.6f} 过小，可能发生梯度消失",
                severity="warning"
            )
            new_anomalies.append(anomaly)
            
        # 检测损失尖峰
        if len(self.history) >= self.window_size:
            recent_losses = [m.train_loss for m in self.history[-self.window_size:-1]]
            if recent_losses:
                avg_loss = np.mean(recent_losses)
                if avg_loss > 0 and metrics.train_loss > avg_loss * self.thresholds['loss_spike_ratio']:
                    anomaly = AnomalyEvent(
                        anomaly_type=AnomalyType.LOSS_SPIKE,
                        episode=metrics.episode,
                        value=metrics.train_loss,
                        threshold=avg_loss * self.thresholds['loss_spike_ratio'],
                        message=f"损失 {metrics.train_loss:.4f} 相比平均值 {avg_loss:.4f} 出现尖峰",
                        severity="warning"
                    )
                    new_anomalies.append(anomaly)
                    
        # 检测损失NaN
        if np.isnan(metrics.train_loss) or np.isinf(metrics.train_loss):
            anomaly = AnomalyEvent(
                anomaly_type=AnomalyType.LOSS_NAN,
                episode=metrics.episode,
                value=metrics.train_loss,
                threshold=0.0,
                message="损失值为NaN或Inf",
                severity="critical"
            )
            new_anomalies.append(anomaly)
            
        # 检测奖励崩溃
        if metrics.avg_reward < self.thresholds['reward_min']:
            anomaly = AnomalyEvent(
                anomaly_type=AnomalyType.REWARD_COLLAPSE,
                episode=metrics.episode,
                value=metrics.avg_reward,
                threshold=self.thresholds['reward_min'],
                message=f"奖励 {metrics.avg_reward:.4f} 低于阈值 {self.thresholds['reward_min']}",
                severity="warning"
            )
            new_anomalies.append(anomaly)
            
        # 检测熵崩溃
        if metrics.entropy < self.thresholds['entropy_min']:
            anomaly = AnomalyEvent(
                anomaly_type=AnomalyType.ENTROPY_COLLAPSE,
                episode=metrics.episode,
                value=metrics.entropy,
                threshold=self.thresholds['entropy_min'],
                message=f"熵 {metrics.entropy:.6f} 过低，策略可能过早收敛",
                severity="warning"
            )
            new_anomalies.append(anomaly)
            
        # 检测KL散度过高
        if abs(metrics.kl_divergence) > self.thresholds['kl_divergence_max']:
            anomaly = AnomalyEvent(
                anomaly_type=AnomalyType.KL_DIVERGENCE_HIGH,
                episode=metrics.episode,
                value=abs(metrics.kl_divergence),
                threshold=self.thresholds['kl_divergence_max'],
                message=f"KL散度 {abs(metrics.kl_divergence):.6f} 过高",
                severity="warning"
            )
            new_anomalies.append(anomaly)
            
        # 检测价值预测误差过大
        if metrics.value_pred_error > self.thresholds['value_pred_error_max']:
            anomaly = AnomalyEvent(
                anomaly_type=AnomalyType.VALUE_PREDICTION_ERROR,
                episode=metrics.episode,
                value=metrics.value_pred_error,
                threshold=self.thresholds['value_pred_error_max'],
                message=f"价值预测误差 {metrics.value_pred_error:.4f} 过大",
                severity="warning"
            )
            new_anomalies.append(anomaly)
            
        # 检测学习率过低
        if metrics.lr < self.thresholds['lr_min']:
            anomaly = AnomalyEvent(
                anomaly_type=AnomalyType.LEARNING_RATE_TOO_LOW,
                episode=metrics.episode,
                value=metrics.lr,
                threshold=self.thresholds['lr_min'],
                message=f"学习率 {metrics.lr:.2e} 过低",
                severity="info"
            )
            new_anomalies.append(anomaly)
            
        self.anomalies.extend(new_anomalies)
        return new_anomalies
    
    def get_recent_anomalies(self, n: int = 10) -> List[AnomalyEvent]:
        """获取最近的异常事件"""
        return self.anomalies[-n:]
    
    def get_anomaly_summary(self) -> Dict[str, int]:
        """获取异常统计摘要"""
        summary = {}
        for anomaly in self.anomalies:
            key = anomaly.anomaly_type.value
            summary[key] = summary.get(key, 0) + 1
        return summary


class EnhancedTrainingMonitor:
    """增强版训练监控器"""
    
    def __init__(self, log_file: Optional[str] = None, max_history: int = 1000):
        self.log_file = log_file
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
        
        # 异常检测器
        self.anomaly_detector = AnomalyDetector()
        
        # 可视化设置
        self.fig = None
        self.axes = {}
        self.lines = {}
        self.paused = False
        self.smooth_window = 10
        
        # 统计信息
        self.best_reward = float('-inf')
        self.best_episode = 0
        
    def parse_log_line(self, line: str) -> Optional[Dict[str, Any]]:
        """解析日志行"""
        metrics = {}
        parts = line.strip().split('|')
        
        for part in parts:
            part = part.strip()
            if ':' in part:
                key, value = part.split(':', 1)
                key = key.strip()
                value = value.strip()
                
                # 尝试转换为数值
                try:
                    if '%' in value:
                        metrics[key] = float(value.rstrip('%')) / 100.0
                    else:
                        metrics[key] = float(value)
                except ValueError:
                    metrics[key] = value
                    
        if 'episode' in metrics:
            return metrics
        return None
    
    def update_data(self, new_data: Dict[str, Any]):
        """更新训练数据"""
        for key, value in new_data.items():
            if key in self.data:
                self.data[key].append(float(value) if isinstance(value, (int, float)) else 0.0)
                # 保持数据量在最大历史范围内
                if len(self.data[key]) > self.max_history:
                    self.data[key] = self.data[key][-self.max_history:]
                    
        # 更新最佳奖励
        if 'avg_reward' in new_data:
            reward = float(new_data['avg_reward'])
            if reward > self.best_reward:
                self.best_reward = reward
                self.best_episode = int(new_data.get('episode', 0))
                
        # 创建指标对象并检测异常
        metrics = TrainingMetrics(
            episode=int(new_data.get('episode', 0)),
            total_steps=int(new_data.get('total_steps', 0)),
            avg_reward=float(new_data.get('avg_reward', 0.0)),
            train_loss=float(new_data.get('train_loss', 0.0)),
            value_loss=float(new_data.get('value_loss', 0.0)),
            policy_loss=float(new_data.get('policy_loss', 0.0)),
            entropy=float(new_data.get('entropy', 0.0)),
            grad_norm=float(new_data.get('grad_norm', 0.0)),
            lr=float(new_data.get('lr', 0.0)),
            entropy_coef=float(new_data.get('entropy_coef', 0.0)),
            kl_divergence=float(new_data.get('kl_divergence', 0.0)),
            kl_coef=float(new_data.get('kl_coef', 0.0)),
            gae_lambda=float(new_data.get('gae_lambda', 0.0)),
            value_pred_error=float(new_data.get('value_pred_error', 0.0)),
        )
        
        anomalies = self.anomaly_detector.update(metrics)
        for anomaly in anomalies:
            print(f"\n[异常检测] {anomaly.timestamp} - {anomaly.anomaly_type.value}: {anomaly.message}")
            
    def smooth_data(self, data: List[float], window: int) -> np.ndarray:
        """平滑数据"""
        if len(data) < window:
            return np.array(data)
        data_array = np.array(data)
        kernel = np.ones(window) / window
        smoothed = np.convolve(data_array, kernel, mode='valid')
        # 填充前面的数据
        padding = np.full(window - 1, data_array[0])
        for i in range(window - 1):
            padding[i] = np.mean(data_array[:i+1])
        return np.concatenate([padding, smoothed])
    
    def setup_plots(self):
        """设置图表"""
        self.fig = plt.figure(figsize=(18, 12))
        gs = gridspec.GridSpec(4, 3, figure=self.fig, hspace=0.35, wspace=0.3)
        
        # 标题
        self.fig.suptitle('增强版训练监控面板', fontsize=16, fontweight='bold')
        
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
        
        # 熵
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
        
        # 价值损失 vs 策略损失
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
        
        # 异常检测面板
        self.axes['anomaly'] = self.fig.add_subplot(gs[2, 2])
        self.axes['anomaly'].set_title('异常检测')
        self.axes['anomaly'].axis('off')
        
        # 统计信息面板
        self.axes['stats'] = self.fig.add_subplot(gs[3, :])
        self.axes['stats'].set_title('训练统计信息')
        self.axes['stats'].axis('off')
        
        # 添加暂停按钮
        ax_pause = plt.axes([0.92, 0.02, 0.07, 0.03])
        self.btn_pause = Button(ax_pause, '暂停/继续')
        self.btn_pause.on_clicked(self.toggle_pause)
        
        # 添加平滑窗口滑块
        ax_slider = plt.axes([0.15, 0.02, 0.3, 0.02])
        self.slider_smooth = Slider(ax_slider, '平滑窗口', 1, 50, valinit=self.smooth_window, valstep=1)
        self.slider_smooth.on_changed(self.update_smooth_window)
        
    def toggle_pause(self, event):
        """切换暂停状态"""
        self.paused = not self.paused
        
    def update_smooth_window(self, val):
        """更新平滑窗口大小"""
        self.smooth_window = int(val)
        
    def update_plots(self):
        """更新所有图表"""
        if not self.data['episode']:
            return
            
        episodes = self.data['episode']
        
        # 更新奖励曲线
        self.axes['reward'].clear()
        self.axes['reward'].set_title('奖励曲线')
        self.axes['reward'].set_xlabel('Episode')
        self.axes['reward'].set_ylabel('奖励值')
        self.axes['reward'].grid(True, alpha=0.3)
        if self.data['avg_reward']:
            self.axes['reward'].plot(episodes, self.data['avg_reward'], 'b-', alpha=0.3, label='原始')
            smoothed = self.smooth_data(self.data['avg_reward'], self.smooth_window)
            self.axes['reward'].plot(episodes, smoothed, 'b-', linewidth=2, label=f'平滑(w={self.smooth_window})')
            self.axes['reward'].axhline(y=self.best_reward, color='g', linestyle='--', alpha=0.5, label=f'最佳: {self.best_reward:.2f}')
            self.axes['reward'].legend(loc='upper left', fontsize=8)
            
        # 更新损失曲线
        self.axes['loss'].clear()
        self.axes['loss'].set_title('损失曲线')
        self.axes['loss'].set_xlabel('Episode')
        self.axes['loss'].set_ylabel('损失值')
        self.axes['loss'].grid(True, alpha=0.3)
        if self.data['train_loss']:
            self.axes['loss'].plot(episodes, self.data['train_loss'], 'r-', alpha=0.3, label='总损失')
            smoothed = self.smooth_data(self.data['train_loss'], self.smooth_window)
            self.axes['loss'].plot(episodes, smoothed, 'r-', linewidth=2, label='平滑')
            self.axes['loss'].legend(loc='upper right', fontsize=8)
            
        # 更新梯度范数
        self.axes['grad_norm'].clear()
        self.axes['grad_norm'].set_title('梯度范数')
        self.axes['grad_norm'].set_xlabel('Episode')
        self.axes['grad_norm'].set_ylabel('范数值')
        self.axes['grad_norm'].grid(True, alpha=0.3)
        if self.data['grad_norm']:
            self.axes['grad_norm'].plot(episodes, self.data['grad_norm'], 'm-', alpha=0.3)
            smoothed = self.smooth_data(self.data['grad_norm'], self.smooth_window)
            self.axes['grad_norm'].plot(episodes, smoothed, 'm-', linewidth=2)
            # 标记异常阈值
            self.axes['grad_norm'].axhline(y=self.anomaly_detector.thresholds['grad_norm_max'], 
                                           color='r', linestyle='--', alpha=0.5, label='爆炸阈值')
            self.axes['grad_norm'].legend(loc='upper right', fontsize=8)
            
        # 更新学习率
        self.axes['lr'].clear()
        self.axes['lr'].set_title('学习率')
        self.axes['lr'].set_xlabel('Episode')
        self.axes['lr'].set_ylabel('学习率')
        self.axes['lr'].set_yscale('log')
        self.axes['lr'].grid(True, alpha=0.3)
        if self.data['lr']:
            self.axes['lr'].plot(episodes, self.data['lr'], 'g-', linewidth=2)
            
        # 更新熵
        self.axes['entropy'].clear()
        self.axes['entropy'].set_title('策略熵')
        self.axes['entropy'].set_xlabel('Episode')
        self.axes['entropy'].set_ylabel('熵值')
        self.axes['entropy'].grid(True, alpha=0.3)
        if self.data['entropy']:
            self.axes['entropy'].plot(episodes, self.data['entropy'], 'c-', alpha=0.3)
            smoothed = self.smooth_data(self.data['entropy'], self.smooth_window)
            self.axes['entropy'].plot(episodes, smoothed, 'c-', linewidth=2)
            # 标记崩溃阈值
            self.axes['entropy'].axhline(y=self.anomaly_detector.thresholds['entropy_min'], 
                                         color='r', linestyle='--', alpha=0.5, label='崩溃阈值')
            self.axes['entropy'].legend(loc='upper right', fontsize=8)
            
        # 更新KL散度
        self.axes['kl'].clear()
        self.axes['kl'].set_title('KL散度')
        self.axes['kl'].set_xlabel('Episode')
        self.axes['kl'].set_ylabel('KL散度')
        self.axes['kl'].grid(True, alpha=0.3)
        if self.data['kl_divergence']:
            self.axes['kl'].plot(episodes, self.data['kl_divergence'], 'orange', alpha=0.3)
            smoothed = self.smooth_data(self.data['kl_divergence'], self.smooth_window)
            self.axes['kl'].plot(episodes, smoothed, 'orange', linewidth=2)
            # 标记阈值
            self.axes['kl'].axhline(y=self.anomaly_detector.thresholds['kl_divergence_max'], 
                                    color='r', linestyle='--', alpha=0.5, label='警告阈值')
            self.axes['kl'].axhline(y=-self.anomaly_detector.thresholds['kl_divergence_max'], 
                                    color='r', linestyle='--', alpha=0.5)
            self.axes['kl'].legend(loc='upper right', fontsize=8)
            
        # 更新损失分解
        self.axes['loss_breakdown'].clear()
        self.axes['loss_breakdown'].set_title('损失分解')
        self.axes['loss_breakdown'].set_xlabel('Episode')
        self.axes['loss_breakdown'].set_ylabel('损失值')
        self.axes['loss_breakdown'].grid(True, alpha=0.3)
        if self.data['value_loss'] and self.data['policy_loss']:
            self.axes['loss_breakdown'].plot(episodes, self.data['value_loss'], 'b-', alpha=0.7, label='价值损失')
            self.axes['loss_breakdown'].plot(episodes, self.data['policy_loss'], 'r-', alpha=0.7, label='策略损失')
            self.axes['loss_breakdown'].legend(loc='upper right', fontsize=8)
            
        # 更新GAE Lambda
        self.axes['gae'].clear()
        self.axes['gae'].set_title('GAE Lambda')
        self.axes['gae'].set_xlabel('Episode')
        self.axes['gae'].set_ylabel('Lambda值')
        self.axes['gae'].grid(True, alpha=0.3)
        if self.data['gae_lambda']:
            self.axes['gae'].plot(episodes, self.data['gae_lambda'], 'purple', linewidth=2)
            self.axes['gae'].set_ylim(0.8, 1.0)
            
        # 更新异常检测面板
        self.axes['anomaly'].clear()
        self.axes['anomaly'].set_title('异常检测')
        self.axes['anomaly'].axis('off')
        
        anomaly_summary = self.anomaly_detector.get_anomaly_summary()
        recent_anomalies = self.anomaly_detector.get_recent_anomalies(5)
        
        if anomaly_summary:
            text = "异常统计:\n"
            for anomaly_type, count in anomaly_summary.items():
                text += f"  {anomaly_type}: {count}次\n"
            text += "\n最近异常:\n"
            for anomaly in recent_anomalies[-3:]:
                text += f"  [{anomaly.timestamp}] {anomaly.anomaly_type.value}\n"
        else:
            text = "暂无异常检测"
            
        self.axes['anomaly'].text(0.1, 0.9, text, transform=self.axes['anomaly'].transAxes,
                                  fontsize=9, verticalalignment='top', fontfamily='monospace',
                                  bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # 更新统计信息面板
        self.axes['stats'].clear()
        self.axes['stats'].set_title('训练统计信息')
        self.axes['stats'].axis('off')
        
        if self.data['episode']:
            stats_text = f"""
            当前Episode: {self.data['episode'][-1]} | 总步数: {self.data['total_steps'][-1] if self.data['total_steps'] else 0}
            最佳奖励: {self.best_reward:.4f} (Episode {self.best_episode})
            平均奖励: {np.mean(self.data['avg_reward']):.4f} | 最近奖励: {self.data['avg_reward'][-1]:.4f}
            平均损失: {np.mean(self.data['train_loss']):.4f} | 最近损失: {self.data['train_loss'][-1]:.4f}
            平均梯度范数: {np.mean(self.data['grad_norm']):.4f} | 最近梯度范数: {self.data['grad_norm'][-1]:.4f}
            当前学习率: {self.data['lr'][-1]:.2e} | 当前熵系数: {self.data['entropy_coef'][-1]:.4f}
            """
            self.axes['stats'].text(0.1, 0.5, stats_text, transform=self.axes['stats'].transAxes,
                                    fontsize=10, verticalalignment='center', fontfamily='monospace')
        
        self.fig.canvas.draw_idle()
        
    def save_plots(self, output_dir: str = './monitor_output'):
        """保存图表"""
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 保存主图表
        self.fig.savefig(os.path.join(output_dir, f'training_monitor_{timestamp}.png'), 
                        dpi=150, bbox_inches='tight')
        
        # 保存异常报告
        anomaly_report = {
            'timestamp': timestamp,
            'total_anomalies': len(self.anomaly_detector.anomalies),
            'anomaly_summary': self.anomaly_detector.get_anomaly_summary(),
            'recent_anomalies': [
                {
                    'type': a.anomaly_type.value,
                    'episode': a.episode,
                    'value': a.value,
                    'threshold': a.threshold,
                    'message': a.message,
                    'severity': a.severity,
                }
                for a in self.anomaly_detector.get_recent_anomalies(20)
            ]
        }
        
        with open(os.path.join(output_dir, f'anomaly_report_{timestamp}.json'), 'w') as f:
            json.dump(anomaly_report, f, indent=2, ensure_ascii=False)
            
        print(f"图表和报告已保存到: {output_dir}")
        
    def monitor_from_file(self, log_path: str, refresh_interval: float = 2.0):
        """从文件监控训练"""
        self.setup_plots()
        
        print(f"开始监控日志文件: {log_path}")
        print("按 Ctrl+C 停止监控")
        
        last_size = 0
        
        try:
            while True:
                if self.paused:
                    time.sleep(0.1)
                    continue
                    
                # 检查文件大小变化
                try:
                    current_size = os.path.getsize(log_path)
                except FileNotFoundError:
                    print(f"日志文件不存在: {log_path}")
                    break
                    
                if current_size > last_size:
                    with open(log_path, 'r') as f:
                        f.seek(last_size)
                        new_lines = f.readlines()
                        
                    for line in new_lines:
                        metrics = self.parse_log_line(line)
                        if metrics:
                            self.update_data(metrics)
                            
                    last_size = current_size
                    self.update_plots()
                    
                time.sleep(refresh_interval)
                
        except KeyboardInterrupt:
            print("\n监控已停止")
            self.save_plots()
            
    def monitor_from_json(self, json_path: str, refresh_interval: float = 2.0):
        """从JSON文件监控训练"""
        self.setup_plots()
        
        print(f"开始监控JSON文件: {json_path}")
        print("按 Ctrl+C 停止监控")
        
        last_count = 0
        
        try:
            while True:
                if self.paused:
                    time.sleep(0.1)
                    continue
                    
                try:
                    with open(json_path, 'r') as f:
                        data = json.load(f)
                except (FileNotFoundError, json.JSONDecodeError):
                    time.sleep(refresh_interval)
                    continue
                    
                if isinstance(data, list) and len(data) > last_count:
                    for metrics in data[last_count:]:
                        self.update_data(metrics)
                    last_count = len(data)
                    self.update_plots()
                    
                time.sleep(refresh_interval)
                
        except KeyboardInterrupt:
            print("\n监控已停止")
            self.save_plots()


def main():
    parser = argparse.ArgumentParser(description='增强版训练监控工具')
    parser.add_argument('--log_file', type=str, default=None, help='训练日志文件路径')
    parser.add_argument('--json_file', type=str, default=None, help='训练JSON文件路径')
    parser.add_argument('--refresh_interval', type=float, default=2.0, help='刷新间隔(秒)')
    parser.add_argument('--output_dir', type=str, default='./monitor_output', help='输出目录')
    
    args = parser.parse_args()
    
    monitor = EnhancedTrainingMonitor()
    
    if args.json_file:
        monitor.monitor_from_json(args.json_file, args.refresh_interval)
    elif args.log_file:
        monitor.monitor_from_file(args.log_file, args.refresh_interval)
    else:
        print("请指定 --log_file 或 --json_file 参数")
        print("\n示例用法:")
        print("  python enhanced_training_monitor.py --log_file results/training_log.txt")
        print("  python enhanced_training_monitor.py --json_file results/training_metrics.json")


if __name__ == '__main__':
    main()
