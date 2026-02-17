#!/usr/bin/env python3
"""
将训练日志转换为JSON格式
"""

import json
import os

def convert_log_to_json(log_file, output_file):
    """将训练日志转换为JSON格式"""
    data = []
    
    with open(log_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            # 解析日志行
            parts = line.split(' | ')
            entry = {}
            
            for part in parts:
                key, value = part.split(': ')
                key = key.strip()
                value = value.strip()
                
                # 转换数值类型
                if key in ['episode', 'total_steps']:
                    entry[key] = int(value)
                elif key in ['avg_reward', 'train_loss']:
                    entry[key] = float(value)
                else:
                    entry[key] = value
            
            # 添加一些默认值
            entry['success_rate'] = 0.0  # 假设默认成功率为0
            entry['learning_rate'] = 0.0003  # 假设学习率为0.0003
            entry['grad_norm'] = 0.0  # 假设默认梯度范数为0
            entry['entropy'] = 0.0  # 假设默认熵为0
            entry['value_loss'] = 0.0  # 假设默认价值损失为0
            entry['policy_loss'] = 0.0  # 假设默认策略损失为0
            entry['stage'] = 0  # 假设默认阶段为0
            
            data.append(entry)
    
    # 写入JSON文件
    with open(output_file, 'w') as f:
        for entry in data:
            json.dump(entry, f)
            f.write('\n')
    
    print(f"Conversion completed: {output_file}")

if __name__ == '__main__':
    log_file = '/home/hmy524/github_project/high_layer/results/sheep_herding/fast_train/ppo/seed1/training_log.txt'
    output_file = '/home/hmy524/github_project/high_layer/results/sheep_herding/fast_train/ppo/seed1/training_log.jsonl'
    
    convert_log_to_json(log_file, output_file)
