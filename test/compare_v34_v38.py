#!/usr/bin/env python
"""
V3.4 vs V3.8 对比推理脚本
用于验证64点轮廓是否改善小轮廓毛刺问题
"""
import os
import sys
import torch
import numpy as np
import cv2
import json
from pathlib import Path

# 设置环境
os.chdir('/mnt/sdb1/leijh/DiffusionSnake/DiffusionSnake-12-30')
sys.path.insert(0, '/mnt/sdb1/leijh/DiffusionSnake/DiffusionSnake-12-30')

def infer_single_version(cfg_file, ckpt_path, output_dir):
    """对单个版本进行推理"""
    os.environ['CFG_FILE'] = cfg_file

    # 重新导入以加载新配置
    import importlib
    if 'lib.config' in sys.modules:
        importlib.reload(sys.modules['lib.config'])

    from lib.config import cfg
    from lib.networks import make_network
    from lib.datasets import make_data_loader
    from lib.utils.snake import snake_gcn_utils, snake_config
    from lib.utils import net_utils

    print(f"\n{'='*60}")
    print(f"推理配置: {cfg_file}")
    print(f"Checkpoint: {ckpt_path}")
    print(f"poly_num: {snake_config.poly_num}")
    print(f"{'='*60}\n")

    # 加载模型
    network = make_network(cfg)
    net_utils.load_network(network, ckpt_path, strict=False)
    network = network.cuda().eval()

    # 加载数据
    data_loader = make_data_loader(cfg, is_train=False)

    # 推理
    results = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
            # 将batch移到GPU
            for k in batch:
                if isinstance(batch[k], torch.Tensor):
                    batch[k] = batch[k].cuda()

            # 推理
            output = network(batch['inp'], batch)

            # 提取预测轮廓
            if 'py' in output:
                pred_contours = output['py'].cpu().numpy()

                # 计算指标
                for i in range(pred_contours.shape[0]):
                    contour = pred_contours[i]

                    # 计算周长
                    perimeter = 0
                    for j in range(len(contour)):
                        p1 = contour[j]
                        p2 = contour[(j+1) % len(contour)]
                        perimeter += np.linalg.norm(p2 - p1)

                    # 计算面积
                    area = cv2.contourArea(contour.astype(np.float32))

                    # 计算曲率
                    curvatures = []
                    for j in range(len(contour)):
                        p_prev = contour[(j-1) % len(contour)]
                        p_curr = contour[j]
                        p_next = contour[(j+1) % len(contour)]

                        # 二阶差分
                        d2 = (p_next - p_curr) - (p_curr - p_prev)
                        curv = np.linalg.norm(d2)
                        curvatures.append(curv)

                    max_curv = np.max(curvatures)
                    mean_curv = np.mean(curvatures)

                    # 计算点密度
                    point_density = perimeter / len(contour) if len(contour) > 0 else 0

                    results.append({
                        'batch_idx': batch_idx,
                        'contour_idx': i,
                        'num_points': len(contour),
                        'perimeter': float(perimeter),
                        'area': float(area),
                        'point_density': float(point_density),
                        'max_curvature': float(max_curv),
                        'mean_curvature': float(mean_curv),
                    })

    # 保存结果
    os.makedirs(output_dir, exist_ok=True)
    result_file = os.path.join(output_dir, 'metrics.json')
    with open(result_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"结果已保存到: {result_file}")
    print(f"共处理 {len(results)} 个轮廓\n")

    return results

def compare_results(results_v34, results_v38, output_dir):
    """对比两个版本的结果"""
    print(f"\n{'='*60}")
    print("V3.4 vs V3.8 对比分析")
    print(f"{'='*60}\n")

    # 计算平均指标
    def calc_avg(results, key):
        values = [r[key] for r in results]
        return np.mean(values), np.std(values)

    metrics = ['max_curvature', 'mean_curvature', 'point_density']

    comparison = {}
    for metric in metrics:
        v34_mean, v34_std = calc_avg(results_v34, metric)
        v38_mean, v38_std = calc_avg(results_v38, metric)

        improvement = ((v34_mean - v38_mean) / v34_mean * 100) if v34_mean != 0 else 0

        comparison[metric] = {
            'v3.4_mean': float(v34_mean),
            'v3.4_std': float(v34_std),
            'v3.8_mean': float(v38_mean),
            'v3.8_std': float(v38_std),
            'improvement_%': float(improvement)
        }

        print(f"{metric}:")
        print(f"  V3.4: {v34_mean:.4f} ± {v34_std:.4f}")
        print(f"  V3.8: {v38_mean:.4f} ± {v38_std:.4f}")
        print(f"  改善: {improvement:+.2f}%")
        print()

    # 保存对比结果
    comparison_file = os.path.join(output_dir, 'comparison.json')
    with open(comparison_file, 'w') as f:
        json.dump(comparison, f, indent=2)

    print(f"对比结果已保存到: {comparison_file}\n")

    return comparison

def main():
    # V3.4推理
    v34_results = infer_single_version(
        cfg_file='configs/btcv_diffusion_dit_v3_4_single_overfit.yaml',
        ckpt_path='data/outputs/btcv_diffusion_dit_v3_4_single_overfit/checkpoints/epoch_10000.pt',
        output_dir='visual/comparison_v34_v38/v3.4'
    )

    # V3.8推理
    v38_results = infer_single_version(
        cfg_file='configs/btcv_diffusion_dit_v3_8_single_overfit.yaml',
        ckpt_path='data/outputs/btcv_diffusion_dit_v3_8_single_overfit/checkpoints/latest.pt',
        output_dir='visual/comparison_v34_v38/v3.8'
    )

    # 对比分析
    comparison = compare_results(v34_results, v38_results, 'visual/comparison_v34_v38')

    print("="*60)
    print("对比完成！")
    print("="*60)

if __name__ == '__main__':
    main()
