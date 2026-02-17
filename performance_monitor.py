"""
性能监控工具
监控CPU、GPU、内存占用情况
"""

import argparse
import os
import time
import psutil
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from datetime import datetime
from typing import Dict, List, Optional, Tuple


try:
    import torch
    has_cuda = torch.cuda.is_available()
except ImportError:
    has_cuda = False

try:
    import pynvml
    has_pynvml = True
except ImportError:
    has_pynvml = False


class PerformanceMonitor:
    def __init__(self, interval: int = 1000, log_dir: Optional[str] = None):
        self.interval = interval
        self.log_dir = log_dir
        self.data = {
            'timestamp': [],
            'cpu_percent': [],
            'memory_percent': [],
            'memory_used': [],
            'memory_total': [],
            'disk_percent': [],
        }
        
        # GPU数据
        if has_cuda or has_pynvml:
            self.data['gpu_usage'] = []
            self.data['gpu_memory_used'] = []
            self.data['gpu_memory_total'] = []
        
        self.start_time = time.time()
        self.fig = None
        self.axes = None
        self.lines = {}
        
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            self.log_file = os.path.join(log_dir, f'performance_log_{timestamp}.txt')
        else:
            self.log_file = None
    
    def collect_data(self):
        """收集性能数据"""
        timestamp = time.time() - self.start_time
        
        # CPU使用率
        cpu_percent = psutil.cpu_percent(interval=0.1)
        
        # 内存使用情况
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        memory_used = memory.used / (1024 ** 3)  # GB
        memory_total = memory.total / (1024 ** 3)  # GB
        
        # 磁盘使用情况
        disk = psutil.disk_usage('/')
        disk_percent = disk.percent
        
        # GPU使用情况
        gpu_usage = 0.0
        gpu_memory_used = 0.0
        gpu_memory_total = 0.0
        
        if has_cuda:
            try:
                gpu_usage = torch.cuda.utilization()
                gpu_memory = torch.cuda.memory_allocated() / (1024 ** 3)  # GB
                gpu_memory_total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)  # GB
            except Exception as e:
                pass
        elif has_pynvml:
            try:
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                info = pynvml.nvmlDeviceGetUtilizationRates(handle)
                gpu_usage = info.gpu
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                gpu_memory_used = mem_info.used / (1024 ** 3)  # GB
                gpu_memory_total = mem_info.total / (1024 ** 3)  # GB
                pynvml.nvmlShutdown()
            except Exception as e:
                pass
        
        # 存储数据
        self.data['timestamp'].append(timestamp)
        self.data['cpu_percent'].append(cpu_percent)
        self.data['memory_percent'].append(memory_percent)
        self.data['memory_used'].append(memory_used)
        self.data['memory_total'].append(memory_total)
        self.data['disk_percent'].append(disk_percent)
        
        if has_cuda or has_pynvml:
            self.data['gpu_usage'].append(gpu_usage)
            self.data['gpu_memory_used'].append(gpu_memory_used)
            self.data['gpu_memory_total'].append(gpu_memory_total)
        
        # 写入日志文件
        if self.log_file:
            with open(self.log_file, 'a') as f:
                log_line = f"{timestamp:.2f}|cpu:{cpu_percent:.1f}%|mem:{memory_percent:.1f}%|disk:{disk_percent:.1f}%"
                if has_cuda or has_pynvml:
                    log_line += f"|gpu:{gpu_usage:.1f}%|gpu_mem:{gpu_memory_used:.2f}/{gpu_memory_total:.2f}GB"
                f.write(log_line + '\n')
    
    def setup_plots(self):
        """设置图表"""
        num_plots = 3
        if has_cuda or has_pynvml:
            num_plots = 4
        
        rows = (num_plots + 1) // 2
        self.fig, self.axes = plt.subplots(rows, 2, figsize=(14, rows * 4))
        self.fig.suptitle('系统性能监控', fontsize=16, fontweight='bold')
        
        axes = self.axes.flatten() if rows > 1 else [self.axes]
        
        # CPU使用率图表
        ax1 = axes[0]
        line1, = ax1.plot([], [], 'b-', linewidth=2, label='CPU使用率')
        self.lines['cpu'] = line1
        ax1.set_xlabel('时间 (秒)')
        ax1.set_ylabel('使用率 (%)')
        ax1.set_title('CPU使用率')
        ax1.set_ylim(0, 100)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 内存使用率图表
        ax2 = axes[1]
        line2, = ax2.plot([], [], 'g-', linewidth=2, label='内存使用率')
        self.lines['memory'] = line2
        ax2.set_xlabel('时间 (秒)')
        ax2.set_ylabel('使用率 (%)')
        ax2.set_title('内存使用率')
        ax2.set_ylim(0, 100)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 磁盘使用率图表
        ax3 = axes[2]
        line3, = ax3.plot([], [], 'y-', linewidth=2, label='磁盘使用率')
        self.lines['disk'] = line3
        ax3.set_xlabel('时间 (秒)')
        ax3.set_ylabel('使用率 (%)')
        ax3.set_title('磁盘使用率')
        ax3.set_ylim(0, 100)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # GPU使用率图表
        if has_cuda or has_pynvml:
            ax4 = axes[3]
            line4, = ax4.plot([], [], 'r-', linewidth=2, label='GPU使用率')
            line5, = ax4.plot([], [], 'm--', linewidth=2, label='GPU内存使用率')
            self.lines['gpu'] = line4
            self.lines['gpu_mem'] = line5
            ax4.set_xlabel('时间 (秒)')
            ax4.set_ylabel('使用率 (%)')
            ax4.set_title('GPU使用率')
            ax4.set_ylim(0, 100)
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
    
    def update_plot(self, frame):
        """更新图表"""
        self.collect_data()
        
        axes = self.axes.flatten() if len(self.axes.shape) > 1 else [self.axes]
        
        # 更新CPU图表
        if 'cpu' in self.lines:
            self.lines['cpu'].set_data(self.data['timestamp'], self.data['cpu_percent'])
            axes[0].relim()
            axes[0].autoscale_view()
        
        # 更新内存图表
        if 'memory' in self.lines:
            self.lines['memory'].set_data(self.data['timestamp'], self.data['memory_percent'])
            axes[1].relim()
            axes[1].autoscale_view()
        
        # 更新磁盘图表
        if 'disk' in self.lines:
            self.lines['disk'].set_data(self.data['timestamp'], self.data['disk_percent'])
            axes[2].relim()
            axes[2].autoscale_view()
        
        # 更新GPU图表
        if has_cuda or has_pynvml:
            if 'gpu' in self.lines:
                self.lines['gpu'].set_data(self.data['timestamp'], self.data['gpu_usage'])
            if 'gpu_mem' in self.lines and self.data['gpu_memory_total'] and self.data['gpu_memory_used']:
                gpu_mem_percent = [(used / total * 100) for used, total in 
                                 zip(self.data['gpu_memory_used'], self.data['gpu_memory_total'])]
                self.lines['gpu_mem'].set_data(self.data['timestamp'], gpu_mem_percent)
            axes[3].relim()
            axes[3].autoscale_view()
        
        return list(self.lines.values())
    
    def run(self, duration: Optional[int] = None):
        """运行监控"""
        self.setup_plots()
        
        ani = animation.FuncAnimation(
            self.fig, self.update_plot,
            interval=self.interval,
            blit=False
        )
        
        if duration:
            # 运行指定时间后停止
            self.fig.canvas.manager.window.after(duration * 1000, plt.close)
        
        plt.show()
    
    def save_data(self, save_path: Optional[str] = None):
        """保存监控数据"""
        if not save_path:
            if not self.log_dir:
                print("请指定保存路径")
                return
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            save_path = os.path.join(self.log_dir, f'performance_data_{timestamp}.npy')
        
        np.save(save_path, self.data)
        print(f"监控数据已保存到: {save_path}")
    
    def generate_report(self, save_path: Optional[str] = None):
        """生成性能报告"""
        if not self.data['timestamp']:
            print("没有监控数据")
            return
        
        if not save_path:
            if not self.log_dir:
                print("请指定保存路径")
                return
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            save_path = os.path.join(self.log_dir, f'performance_report_{timestamp}.png')
        
        num_plots = 3
        if has_cuda or has_pynvml:
            num_plots = 4
        
        rows = (num_plots + 1) // 2
        fig, axes = plt.subplots(rows, 2, figsize=(14, rows * 4))
        fig.suptitle('系统性能报告', fontsize=16, fontweight='bold')
        
        axes = axes.flatten() if rows > 1 else [axes]
        
        # CPU使用率图表
        ax1 = axes[0]
        ax1.plot(self.data['timestamp'], self.data['cpu_percent'], 'b-', linewidth=2)
        ax1.set_xlabel('时间 (秒)')
        ax1.set_ylabel('使用率 (%)')
        ax1.set_title('CPU使用率')
        ax1.set_ylim(0, 100)
        ax1.grid(True, alpha=0.3)
        avg_cpu = np.mean(self.data['cpu_percent'])
        max_cpu = np.max(self.data['cpu_percent'])
        ax1.text(0.05, 0.95, f'平均: {avg_cpu:.1f}%\n最大: {max_cpu:.1f}%', 
                 transform=ax1.transAxes, verticalalignment='top', 
                 bbox=dict(boxstyle='round', alpha=0.1))
        
        # 内存使用率图表
        ax2 = axes[1]
        ax2.plot(self.data['timestamp'], self.data['memory_percent'], 'g-', linewidth=2)
        ax2.set_xlabel('时间 (秒)')
        ax2.set_ylabel('使用率 (%)')
        ax2.set_title('内存使用率')
        ax2.set_ylim(0, 100)
        ax2.grid(True, alpha=0.3)
        avg_mem = np.mean(self.data['memory_percent'])
        max_mem = np.max(self.data['memory_percent'])
        ax2.text(0.05, 0.95, f'平均: {avg_mem:.1f}%\n最大: {max_mem:.1f}%', 
                 transform=ax2.transAxes, verticalalignment='top', 
                 bbox=dict(boxstyle='round', alpha=0.1))
        
        # 磁盘使用率图表
        ax3 = axes[2]
        ax3.plot(self.data['timestamp'], self.data['disk_percent'], 'y-', linewidth=2)
        ax3.set_xlabel('时间 (秒)')
        ax3.set_ylabel('使用率 (%)')
        ax3.set_title('磁盘使用率')
        ax3.set_ylim(0, 100)
        ax3.grid(True, alpha=0.3)
        avg_disk = np.mean(self.data['disk_percent'])
        max_disk = np.max(self.data['disk_percent'])
        ax3.text(0.05, 0.95, f'平均: {avg_disk:.1f}%\n最大: {max_disk:.1f}%', 
                 transform=ax3.transAxes, verticalalignment='top', 
                 bbox=dict(boxstyle='round', alpha=0.1))
        
        # GPU使用率图表
        if has_cuda or has_pynvml:
            ax4 = axes[3]
            ax4.plot(self.data['timestamp'], self.data['gpu_usage'], 'r-', linewidth=2, label='GPU使用率')
            if self.data['gpu_memory_total'] and self.data['gpu_memory_used']:
                gpu_mem_percent = [(used / total * 100) for used, total in 
                                 zip(self.data['gpu_memory_used'], self.data['gpu_memory_total'])]
                ax4.plot(self.data['timestamp'], gpu_mem_percent, 'm--', linewidth=2, label='GPU内存使用率')
            ax4.set_xlabel('时间 (秒)')
            ax4.set_ylabel('使用率 (%)')
            ax4.set_title('GPU使用率')
            ax4.set_ylim(0, 100)
            ax4.legend()
            ax4.grid(True, alpha=0.3)
            avg_gpu = np.mean(self.data['gpu_usage'])
            max_gpu = np.max(self.data['gpu_usage'])
            ax4.text(0.05, 0.95, f'平均: {avg_gpu:.1f}%\n最大: {max_gpu:.1f}%', 
                     transform=ax4.transAxes, verticalalignment='top', 
                     bbox=dict(boxstyle='round', alpha=0.1))
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"性能报告已保存到: {save_path}")


class TrainingPerformanceMonitor:
    """训练过程性能监控器"""
    def __init__(self, log_dir: Optional[str] = None):
        self.monitor = PerformanceMonitor(log_dir=log_dir)
        self.is_running = False
    
    def start(self):
        """开始监控"""
        self.is_running = True
        self.monitor.start_time = time.time()
        print("性能监控已启动")
    
    def stop(self):
        """停止监控"""
        self.is_running = False
        print("性能监控已停止")
    
    def collect(self):
        """收集数据"""
        if self.is_running:
            self.monitor.collect_data()
    
    def save_report(self, save_path: Optional[str] = None):
        """保存报告"""
        self.monitor.generate_report(save_path)
    
    def save_data(self, save_path: Optional[str] = None):
        """保存数据"""
        self.monitor.save_data(save_path)


def parse_args():
    parser = argparse.ArgumentParser(description='性能监控工具')
    
    parser.add_argument('--interval', type=int, default=1000,
                        help='监控间隔 (毫秒)')
    parser.add_argument('--duration', type=int, default=None,
                        help='监控持续时间 (秒)')
    parser.add_argument('--log_dir', type=str, default=None,
                        help='日志保存目录')
    parser.add_argument('--save_data', type=str, default=None,
                        help='保存监控数据的路径')
    parser.add_argument('--generate_report', type=str, default=None,
                        help='生成性能报告的路径')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    monitor = PerformanceMonitor(
        interval=args.interval,
        log_dir=args.log_dir
    )
    
    print("性能监控工具启动")
    print(f"监控间隔: {args.interval}ms")
    if args.duration:
        print(f"监控持续时间: {args.duration}s")
    if args.log_dir:
        print(f"日志保存目录: {args.log_dir}")
    
    # 运行监控
    monitor.run(duration=args.duration)
    
    # 保存数据
    if args.save_data:
        monitor.save_data(args.save_data)
    
    # 生成报告
    if args.generate_report:
        monitor.generate_report(args.generate_report)
    
    print("性能监控工具退出")


if __name__ == '__main__':
    main()
