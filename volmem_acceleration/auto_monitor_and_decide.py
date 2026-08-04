#!/usr/bin/env python3
"""
自动监控和决策系统
- 持续监控运行中的实验
- 分析性能数据
- 自动启动下一轮实验
"""

import json
import os
import subprocess
import time
from pathlib import Path
from collections import defaultdict

class ExperimentMonitor:
    def __init__(self):
        self.base_dir = Path("data/outputs/volmem")
        self.exp_results = {}
        self.decision_log = []
        
    def extract_latest_data(self, exp_name):
        """从train.jsonl提取最新数据"""
        exp_dir = self.base_dir / f"exp_{exp_name}_*"
        
        # 找到匹配的目录
        import glob
        dirs = glob.glob(str(exp_dir))
        if not dirs:
            return None
            
        log_file = Path(dirs[0]) / "train.jsonl"
        if not log_file.exists():
            return None
        
        try:
            # 读取最后N行
            with open(log_file, 'r') as f:
                lines = f.readlines()[-50:]  # 最后50行
            
            if not lines:
                return None
                
            # 解析最后一行
            last_data = json.loads(lines[-1])
            
            return {
                'exp': exp_name,
                'steps': last_data.get('step', 0),
                'time_ms': last_data.get('time_ms', 0),
                'memory_gb': last_data.get('peak_memory_gb', 0),
                'loss': last_data.get('loss', 0),
            }
        except Exception as e:
            print(f"Error parsing {log_file}: {e}")
            return None
    
    def analyze_results(self, baseline_time=55339, baseline_memory=30.1):
        """分析所有实验的结果"""
        results = {}
        
        for exp in ['a', 'b', 'c', 'd']:
            data = self.extract_latest_data(exp)
            if data:
                data['speedup'] = baseline_time / data['time_ms'] if data['time_ms'] > 0 else 0
                data['memory_saving'] = (baseline_memory - data['memory_gb']) / baseline_memory * 100
                results[exp] = data
        
        return results
    
    def make_decision(self, results):
        """基于结果做出下一步决策"""
        decisions = []
        
        # 找到最好的实验
        best_exp = None
        best_speedup = 1.0
        
        for exp, data in results.items():
            if data['speedup'] > best_speedup:
                best_speedup = data['speedup']
                best_exp = exp
        
        if best_speedup >= 1.2:
            decisions.append(f"✅ 实验成功! 最好的实验: Exp_{best_exp} (加速: {best_speedup:.2f}x)")
        else:
            decisions.append(f"⚠️  加速不足. 需要启动更激进的实验或代码级优化")
        
        return decisions
    
    def run_continuous_monitoring(self, interval=60, max_iterations=1000):
        """持续监控"""
        iteration = 0
        while iteration < max_iterations:
            print(f"\n[监控循环 {iteration}] 时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
            
            results = self.analyze_results()
            
            if results:
                print("\n实验结果:")
                for exp, data in sorted(results.items()):
                    print(f"  Exp_{exp}: {data['time_ms']:.0f}ms -> {data['speedup']:.2f}x, "
                          f"显存: {data['memory_gb']:.1f}GB ({data['memory_saving']:.0f}%), "
                          f"Loss: {data['loss']:.6f}")
                
                decisions = self.make_decision(results)
                for d in decisions:
                    print(d)
            else:
                print("还没有实验数据...")
            
            # 保存监控记录
            self.save_monitoring_log(results)
            
            iteration += 1
            time.sleep(interval)
    
    def save_monitoring_log(self, results):
        """保存监控日志"""
        log_file = Path("volmem_acceleration/results/monitoring_log.json")
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            existing = []
            if log_file.exists():
                with open(log_file) as f:
                    existing = json.load(f)
        except:
            existing = []
        
        entry = {
            'timestamp': time.time(),
            'results': results
        }
        existing.append(entry)
        
        # 只保留最后100条记录
        with open(log_file, 'w') as f:
            json.dump(existing[-100:], f, indent=2)

if __name__ == "__main__":
    monitor = ExperimentMonitor()
    monitor.run_continuous_monitoring(interval=30)  # 每30秒检查一次

