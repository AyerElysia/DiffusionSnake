#!/usr/bin/env python3
"""
V0.5 推理加速 - 自动化主控脚本
自主执行所有实验和决策，无需人工干预
"""

import os
import json
import subprocess
import time
from pathlib import Path
from datetime import datetime

class V05AccelerationMaster:
    def __init__(self):
        self.base_dir = Path("data/outputs/volmem")
        self.config_dir = Path("configs/volmem")
        self.result_dir = Path("volmem_acceleration/results")
        self.result_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.result_dir / "master_control.log"
        
        self.experiments = {
            'phase1': ['exp_a_feature_halve', 'exp_b_chunks_20', 'exp_c_feature_quarter', 'exp_d_combined'],
            'phase2': ['exp_e_aggressive', 'exp_f_extreme', 'exp_g_balanced'],
        }
        
        self.baseline = {
            'time_ms': 55339,
            'memory_gb': 30.1,
            'loss': 0.005,
        }
    
    def log(self, msg):
        """记录信息"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_msg = f"[{timestamp}] {msg}"
        print(log_msg)
        with open(self.log_file, 'a') as f:
            f.write(log_msg + "\n")
    
    def extract_results(self, exp_name):
        """从train.jsonl提取最新性能数据"""
        exp_dirs = list(self.base_dir.glob(f"{exp_name}*"))
        if not exp_dirs:
            return None
        
        log_file = exp_dirs[0] / "train.jsonl"
        if not log_file.exists():
            return None
        
        try:
            with open(log_file) as f:
                lines = f.readlines()[-10:]
            if not lines:
                return None
            
            last_data = json.loads(lines[-1])
            return {
                'exp': exp_name,
                'steps': last_data.get('step', 0),
                'time_ms': last_data.get('time_ms', 0),
                'memory_gb': last_data.get('peak_memory_gb', 0),
                'loss': last_data.get('loss', 0),
                'speedup': self.baseline['time_ms'] / last_data.get('time_ms', 1),
                'memory_saving': (self.baseline['memory_gb'] - last_data.get('peak_memory_gb', 0)) / self.baseline['memory_gb'],
            }
        except Exception as e:
            return None
    
    def check_phase1_completion(self):
        """检查第一阶段是否完成"""
        results = {}
        for exp in self.experiments['phase1']:
            data = self.extract_results(exp)
            if data and data['steps'] >= 50:  # 至少50步
                results[exp] = data
        
        return results if len(results) == len(self.experiments['phase1']) else None
    
    def analyze_and_decide(self, results):
        """分析结果并做出决策"""
        self.log(f"分析第一阶段结果...")
        
        best_exp = max(results.items(), key=lambda x: x[1]['speedup'])
        best_speedup = best_exp[1]['speedup']
        
        self.log(f"最优实验: {best_exp[0]}, 加速比: {best_speedup:.2f}x")
        
        if best_speedup >= 1.2:
            self.log(f"✅ 第一阶段成功! 启动第二阶段...")
            return 'phase2'
        elif best_speedup >= 1.1:
            self.log(f"△ 加速有效但不足, 启动并行代码优化...")
            return 'phase2_with_code'
        else:
            self.log(f"⚠️  加速不足, 启动激进代码优化...")
            return 'code_optimization'
    
    def launch_phase2(self):
        """启动第二阶段实验"""
        self.log(f"启动第二阶段实验 (Exp_E/F/G)...")
        for i, exp in enumerate(self.experiments['phase2']):
            gpu = 4 + i
            config_file = self.config_dir / f"{exp}.yaml"
            if config_file.exists():
                cmd = f"nohup python train_net.py --cfg {config_file} > logs/{exp}.log 2>&1 &"
                os.system(cmd)
                self.log(f"已启动 {exp} on GPU{gpu}")
    
    def auto_run(self):
        """自动运行主循环"""
        self.log("=" * 60)
        self.log("V0.5推理加速 - 自动化主控启动")
        self.log("=" * 60)
        
        phase1_complete = False
        phase2_launched = False
        
        iteration = 0
        while True:
            iteration += 1
            self.log(f"\n循环 #{iteration} - {datetime.now().strftime('%H:%M:%S')}")
            
            # 检查第一阶段
            if not phase1_complete:
                results = self.check_phase1_completion()
                if results:
                    self.log(f"第一阶段完成，找到{len(results)}个有效结果")
                    
                    # 保存结果
                    with open(self.result_dir / "phase1_results.json", 'w') as f:
                        json.dump(results, f, indent=2)
                    
                    # 决策
                    decision = self.analyze_and_decide(results)
                    
                    if decision in ['phase2', 'phase2_with_code']:
                        self.launch_phase2()
                        phase2_launched = True
                    
                    phase1_complete = True
            
            # 检查第二阶段
            if phase1_complete and phase2_launched:
                phase2_results = {}
                for exp in self.experiments['phase2']:
                    data = self.extract_results(exp)
                    if data and data['steps'] >= 50:
                        phase2_results[exp] = data
                
                if len(phase2_results) == len(self.experiments['phase2']):
                    self.log(f"第二阶段完成，启动最终分析...")
                    
                    with open(self.result_dir / "phase2_results.json", 'w') as f:
                        json.dump(phase2_results, f, indent=2)
                    
                    self.finalize_report(phase2_results)
                    break
            
            time.sleep(60)  # 每分钟检查一次
    
    def finalize_report(self, results):
        """生成最终报告"""
        self.log("=" * 60)
        self.log("生成最终报告...")
        
        best = max(results.items(), key=lambda x: x[1]['speedup'])
        
        report = f"""
# V0.5 推理加速 - 最终结果

## 最优方案
**实验**: {best[0]}  
**推理加速**: {best[1]['speedup']:.2f}x (从 {self.baseline['time_ms']:.0f}ms → {best[1]['time_ms']:.0f}ms)  
**显存节省**: {best[1]['memory_saving']*100:.1f}% (从 {self.baseline['memory_gb']:.1f}GB → {best[1]['memory_gb']:.1f}GB)  
**精度**: TBD (需验证)

## 所有结果
"""
        for exp, data in sorted(results.items(), key=lambda x: x[1]['speedup'], reverse=True):
            report += f"\n- {exp}: {data['speedup']:.2f}x, 显存{data['memory_saving']*100:.1f}%, Loss={data['loss']:.6f}"
        
        with open(self.result_dir / "FINAL_REPORT.md", 'w') as f:
            f.write(report)
        
        self.log("✅ 最终报告已生成")
        self.log("=" * 60)

if __name__ == "__main__":
    master = V05AccelerationMaster()
    master.auto_run()

