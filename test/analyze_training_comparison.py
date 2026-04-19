#!/usr/bin/env python
"""
V3.4 vs V3.8 训练日志对比分析
通过对比训练loss曲线来初步判断64点是否有效
"""
import json
import numpy as np
import matplotlib.pyplot as plt
import os

def load_training_logs(log_file):
    """加载训练日志"""
    epochs = []
    losses = []

    with open(log_file, 'r') as f:
        for line in f:
            data = json.loads(line)
            epochs.append(data['epoch'])
            losses.append(data['loss'])

    return np.array(epochs), np.array(losses)

def analyze_convergence(epochs, losses, window=100):
    """分析收敛情况"""
    if len(losses) < window:
        return None

    # 最后window个epoch的统计
    recent_losses = losses[-window:]
    mean_loss = np.mean(recent_losses)
    std_loss = np.std(recent_losses)
    min_loss = np.min(recent_losses)
    max_loss = np.max(recent_losses)

    # 计算趋势（最后window个epoch的线性拟合斜率）
    recent_epochs = epochs[-window:]
    coeffs = np.polyfit(recent_epochs, recent_losses, 1)
    trend = coeffs[0]  # 斜率

    return {
        'mean': mean_loss,
        'std': std_loss,
        'min': min_loss,
        'max': max_loss,
        'cv': std_loss / mean_loss if mean_loss > 0 else 0,  # 变异系数
        'trend': trend
    }

def main():
    print("="*60)
    print("V3.4 vs V3.8 训练日志对比分析")
    print("="*60)

    # 加载日志
    v34_log = 'data/outputs/btcv_diffusion_dit_v3_4_single_overfit/logs.jsonl'
    v38_log = 'data/outputs/btcv_diffusion_dit_v3_8_single_overfit/logs.jsonl'

    print("\n加载训练日志...")
    epochs_v34, losses_v34 = load_training_logs(v34_log)
    epochs_v38, losses_v38 = load_training_logs(v38_log)

    print(f"V3.4: {len(epochs_v34)} epochs")
    print(f"V3.8: {len(epochs_v38)} epochs")

    # 分析收敛情况
    print("\n收敛分析（最后100 epochs）:")
    print("-"*60)

    stats_v34 = analyze_convergence(epochs_v34, losses_v34)
    stats_v38 = analyze_convergence(epochs_v38, losses_v38)

    print(f"\nV3.4 (128点):")
    print(f"  平均loss: {stats_v34['mean']:.6f}")
    print(f"  标准差: {stats_v34['std']:.6f}")
    print(f"  变异系数: {stats_v34['cv']:.4f}")
    print(f"  最小loss: {stats_v34['min']:.6f}")
    print(f"  最大loss: {stats_v34['max']:.6f}")
    print(f"  趋势（斜率）: {stats_v34['trend']:.2e}")

    print(f"\nV3.8 (64点):")
    print(f"  平均loss: {stats_v38['mean']:.6f}")
    print(f"  标准差: {stats_v38['std']:.6f}")
    print(f"  变异系数: {stats_v38['cv']:.4f}")
    print(f"  最小loss: {stats_v38['min']:.6f}")
    print(f"  最大loss: {stats_v38['max']:.6f}")
    print(f"  趋势（斜率）: {stats_v38['trend']:.2e}")

    # 对比
    print("\n对比分析:")
    print("-"*60)
    loss_diff = ((stats_v34['mean'] - stats_v38['mean']) / stats_v34['mean']) * 100
    print(f"平均loss差异: {loss_diff:+.2f}% (V3.8相对V3.4)")

    if abs(loss_diff) < 5:
        print("结论: 两个版本的训练loss相近，说明64点模型训练效果与128点相当")
    elif loss_diff > 0:
        print("结论: V3.8 (64点) loss更低，训练效果可能更好")
    else:
        print("结论: V3.4 (128点) loss更低，但差异较小")

    # 绘制loss曲线
    print("\n生成loss曲线对比图...")
    plt.figure(figsize=(14, 6))

    # 完整曲线
    plt.subplot(1, 2, 1)
    plt.plot(epochs_v34, losses_v34, label='V3.4 (128 points)', alpha=0.7)
    plt.plot(epochs_v38, losses_v38, label='V3.8 (64 points)', alpha=0.7)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Comparison (Full)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 最后500 epochs
    plt.subplot(1, 2, 2)
    start_idx_v34 = max(0, len(epochs_v34) - 500)
    start_idx_v38 = max(0, len(epochs_v38) - 500)
    plt.plot(epochs_v34[start_idx_v34:], losses_v34[start_idx_v34:], label='V3.4 (128 points)', alpha=0.7)
    plt.plot(epochs_v38[start_idx_v38:], losses_v38[start_idx_v38:], label='V3.8 (64 points)', alpha=0.7)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Comparison (Last 500 Epochs)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    output_dir = 'visual/comparison_v34_v38'
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'training_loss_comparison.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"已保存: {output_file}")

    # 保存统计结果
    stats_file = os.path.join(output_dir, 'training_stats.json')
    with open(stats_file, 'w') as f:
        json.dump({
            'v3.4': stats_v34,
            'v3.8': stats_v38,
            'comparison': {
                'loss_diff_percent': loss_diff
            }
        }, f, indent=2)
    print(f"已保存: {stats_file}")

    print("\n="*60)
    print("分析完成！")
    print("="*60)

if __name__ == '__main__':
    main()
