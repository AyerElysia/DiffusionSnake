#!/usr/bin/env python3
"""
自动触发第二阶段
当第一阶段完成时自动启动第二阶段
"""

import json
import subprocess
import time
from pathlib import Path

def check_phase1_ready():
    """检查第一阶段是否完成（50+ steps）"""
    results = {}
    base_dir = Path("data/outputs/volmem")
    
    for exp in ['safe_exp_a_chunks16', 'safe_exp_b_chunks20', 'safe_exp_c_chunks24', 'safe_exp_d_memory_halve']:
        exp_dirs = list(base_dir.glob(f"{exp}*"))
        if exp_dirs:
            log_file = exp_dirs[0] / "train.jsonl"
            if log_file.exists():
                try:
                    with open(log_file) as f:
                        lines = f.readlines()
                    if lines:
                        last = json.loads(lines[-1])
                        steps = last.get('step', 0)
                        if steps >= 50:
                            results[exp] = steps
                except:
                    pass
    
    return len(results) == 4

def launch_phase2():
    """启动第二阶段"""
    exps = [
        ('phase2_exp_a_chunks28.yaml', 'GPU4'),
        ('phase2_exp_b_chunks20_memory2.yaml', 'GPU5'),
        ('phase2_exp_c_chunks20_pool4.yaml', 'GPU6'),
    ]
    
    for config, gpu in exps:
        cmd = f"nohup python train_net.py --cfg configs/volmem/{config} > logs/{config}.log 2>&1 &"
        subprocess.run(cmd, shell=True)
        print(f"启动 {config} ({gpu})")
        time.sleep(2)

if __name__ == "__main__":
    # 每5分钟检查一次
    for _ in range(600):  # 50小时
        if check_phase1_ready():
            print("✅ 第一阶段完成，启动第二阶段")
            launch_phase2()
            break
        time.sleep(300)

