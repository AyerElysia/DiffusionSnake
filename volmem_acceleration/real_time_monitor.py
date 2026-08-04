#!/usr/bin/env python3
import json
import os
import time
from pathlib import Path
from collections import defaultdict

def monitor_experiments():
    """实时监控所有实验进度"""
    base_dir = Path("data/outputs/volmem")
    
    while True:
        print("\n" + "="*60)
        print(f"【实验进度监控】{time.strftime('%H:%M:%S')}")
        print("="*60)
        
        experiments = {
            'Baseline': 'verse_memflowdit_v0_5_minimal_gpu6',
            'Safe_A': 'safe_exp_a_chunks16',
            'Safe_B': 'safe_exp_b_chunks20',
            'Safe_C': 'safe_exp_c_chunks24',
            'Safe_D': 'safe_exp_d_memory_halve',
        }
        
        results = {}
        for name, dir_name in experiments.items():
            log_file = base_dir / dir_name / "train.jsonl"
            if log_file.exists():
                try:
                    with open(log_file) as f:
                        last_line = f.readlines()[-1]
                    data = json.loads(last_line)
                    step = data.get('step', 0)
                    time_ms = data.get('time_ms', 0)
                    mem = data.get('peak_memory_gb', 0)
                    loss = data.get('loss', 0)
                    
                    results[name] = {
                        'step': step,
                        'time': time_ms/1000,
                        'memory': mem,
                        'loss': loss
                    }
                    
                    accel = 55.3 / (time_ms/1000) if time_ms > 0 else 0
                    print(f"✅ {name:12} | step={step:5d} | time={time_ms/1000:6.1f}s | mem={mem:5.1f}GB | accel={accel:4.2f}x")
                except:
                    print(f"❌ {name:12} | 数据解析错误")
            else:
                print(f"⏳ {name:12} | 未启动")
        
        # 每 30 秒检查一次
        time.sleep(30)

if __name__ == "__main__":
    monitor_experiments()
