#!/usr/bin/env python3
"""
批量推理所有单样本过拟合模型
"""
import os
import sys
import subprocess
from pathlib import Path

# 模型配置列表
MODELS = [
    {
        'name': 'V2.0',
        'cfg': 'configs/btcv_diffusion_dit_v2_single_overfit.yaml',
        'ckpt': 'data/outputs/btcv_diffusion_dit_v2_single_overfit/checkpoints/latest.pt',
        'output': 'visual/v2_0_single_overfit_infer'
    },
    {
        'name': 'V2.1',
        'cfg': 'configs/btcv_diffusion_dit_v2_1_single_overfit.yaml',
        'ckpt': 'data/outputs/btcv_diffusion_dit_v2_1_single_overfit/checkpoints/latest.pt',
        'output': 'visual/v2_1_single_overfit_infer'
    },
    {
        'name': 'V2.2',
        'cfg': 'configs/btcv_diffusion_dit_v2_2_single_overfit.yaml',
        'ckpt': 'data/outputs/btcv_diffusion_dit_v2_2_single_overfit/checkpoints/latest.pt',
        'output': 'visual/v2_2_single_overfit_infer'
    },
    {
        'name': 'V2.3',
        'cfg': 'configs/btcv_diffusion_dit_v2_3_single_overfit_gpu6.yaml',
        'ckpt': 'data/outputs/btcv_diffusion_dit_v2_3_single_overfit_gpu6/checkpoints/latest.pt',
        'output': 'visual/v2_3_single_overfit_infer'
    },
    {
        'name': 'V3.1',
        'cfg': 'configs/btcv_diffusion_dit_v3_1_single_overfit.yaml',
        'ckpt': 'data/outputs/btcv_diffusion_dit_v3_1_single_overfit/checkpoints/latest.pt',
        'output': 'visual/v3_1_single_overfit_infer'
    },
    {
        'name': 'V3.2',
        'cfg': 'configs/btcv_diffusion_dit_v3_2_single_overfit.yaml',
        'ckpt': 'data/outputs/btcv_diffusion_dit_v3_2_single_overfit/checkpoints/latest.pt',
        'output': 'visual/v3_2_single_overfit_infer'
    },
]

def main():
    root_dir = Path(__file__).parent
    os.chdir(root_dir)

    print("=" * 80)
    print("开始批量推理所有单样本过拟合模型")
    print("=" * 80)

    for model in MODELS:
        print(f"\n{'='*80}")
        print(f"推理模型: {model['name']}")
        print(f"配置文件: {model['cfg']}")
        print(f"权重文件: {model['ckpt']}")
        print(f"输出目录: {model['output']}")
        print(f"{'='*80}\n")

        # 检查权重文件是否存在
        if not Path(model['ckpt']).exists():
            print(f"[警告] 权重文件不存在，跳过: {model['ckpt']}")
            continue

        # 设置环境变量
        env = os.environ.copy()
        env['CFG_FILE'] = model['cfg']

        # 构建推理命令
        cmd = [
            sys.executable,
            'scripts/infer_v3_final.py',
            '--ckpt', model['ckpt'],
        ]

        try:
            # 执行推理
            result = subprocess.run(
                cmd,
                env=env,
                capture_output=False,
                text=True,
                check=True
            )
            print(f"\n[✓] {model['name']} 推理完成\n")
        except subprocess.CalledProcessError as e:
            print(f"\n[✗] {model['name']} 推理失败")
            print(f"错误信息: {e}")
            continue

    print("\n" + "=" * 80)
    print("所有模型推理完成")
    print("=" * 80)

if __name__ == '__main__':
    main()
