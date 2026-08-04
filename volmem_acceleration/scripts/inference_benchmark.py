#!/usr/bin/env python3
"""
VolMem 推理性能 Benchmark 脚本

目的：
  - 度量基线推理速度
  - 追踪显存使用
  - 记录性能指标便于优化对比
"""

import argparse
import json
import os
import pathlib
import sys
import time
from collections import OrderedDict, defaultdict
from contextlib import nullcontext
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.cuda


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class InferenceBenchmark:
    """推理性能基准测试工具"""
    
    def __init__(self, config_path: str, checkpoint_path: str, device: str = "cuda"):
        self.config_path = config_path
        self.checkpoint_path = checkpoint_path
        self.device = device
        
        # 性能计数器
        self.timings = []
        self.memory_peaks = []
        self.forward_passes = []
        
        self._setup()
    
    def _setup(self):
        """初始化模型和配置"""
        os.environ["CFG_FILE"] = self.config_path
        sys.argv = [sys.argv[0], "--cfg_file", self.config_path]
        
        # 这里需要根据实际情况导入和初始化模型
        print(f"[Benchmark] 配置: {self.config_path}")
        print(f"[Benchmark] 检查点: {self.checkpoint_path}")
        print(f"[Benchmark] 设备: {self.device}")
    
    def record_forward_time(self, batch_id: int, volume_id: str, n_slices: int, elapsed_ms: float):
        """记录单次前向推理时间"""
        self.forward_passes.append({
            "batch_id": batch_id,
            "volume_id": volume_id,
            "n_slices": n_slices,
            "elapsed_ms": elapsed_ms,
            "per_slice_ms": elapsed_ms / n_slices if n_slices > 0 else 0,
        })
        self.timings.append(elapsed_ms)
    
    def record_memory_peak(self, peak_gb: float):
        """记录显存峰值"""
        self.memory_peaks.append(peak_gb)
    
    def get_summary(self) -> Dict:
        """获取性能总结"""
        if not self.timings:
            return {"status": "no_data"}
        
        timings = np.array(self.timings)
        memory_peaks = np.array(self.memory_peaks)
        
        return {
            "forward_passes": len(self.forward_passes),
            "total_slices_processed": sum(p["n_slices"] for p in self.forward_passes),
            "timing_ms": {
                "mean": float(np.mean(timings)),
                "median": float(np.median(timings)),
                "std": float(np.std(timings)),
                "min": float(np.min(timings)),
                "max": float(np.max(timings)),
                "total": float(np.sum(timings)),
            },
            "per_slice_ms": {
                "mean": float(np.mean([p["per_slice_ms"] for p in self.forward_passes])),
                "median": float(np.median([p["per_slice_ms"] for p in self.forward_passes])),
            },
            "memory_gb": {
                "mean": float(np.mean(memory_peaks)),
                "max": float(np.max(memory_peaks)),
                "min": float(np.min(memory_peaks)),
            },
            "forward_passes": self.forward_passes[:10],  # 保存前10个作为样本
        }
    
    def save_results(self, output_path: str):
        """保存结果到文件"""
        results = {
            "config": self.config_path,
            "checkpoint": self.checkpoint_path,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "device": self.device,
            "summary": self.get_summary(),
            "all_passes": self.forward_passes,
        }
        
        output_dir = pathlib.Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        
        print(f"[Benchmark] 结果已保存到: {output_path}")
        return results


def main():
    parser = argparse.ArgumentParser(description="VolMem 推理性能基准测试")
    parser.add_argument("--cfg-file", required=True, help="配置文件路径")
    parser.add_argument("--ckpt", required=True, help="检查点路径")
    parser.add_argument("--output-dir", required=True, help="结果输出目录")
    parser.add_argument("--device", default="cuda", help="运行设备")
    parser.add_argument("--num-samples", type=int, default=10, help="测试样本数")
    parser.add_argument("--warmup", type=int, default=2, help="预热轮数")
    
    args = parser.parse_args()
    
    benchmark = InferenceBenchmark(
        config_path=args.cfg_file,
        checkpoint_path=args.ckpt,
        device=args.device,
    )
    
    # 注意：这只是框架，实际测试需要集成到训练代码中
    print("[Benchmark] 框架已建立")
    print("[Benchmark] 下一步：集成到实际推理循环中")
    
    # 保存结果
    output_path = os.path.join(
        args.output_dir,
        f"benchmark_{time.strftime('%Y%m%d_%H%M%S')}.json"
    )
    benchmark.save_results(output_path)


if __name__ == "__main__":
    main()
