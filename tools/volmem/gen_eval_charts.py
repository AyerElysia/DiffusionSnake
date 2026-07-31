#!/usr/bin/env python3
"""Generate charts from MemFlowDiT step2250 slices.json for appendix."""
import json
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SLICES_JSON = '/home/medteam/Zhrch/DiffusionSnake-12-30/data/outputs/volmem/verse_memflowdit_v0_3_2day_gpu6/eval_step_2250_autoregressive_gt/slices.json'
OUT_DIR = '/home/medteam/Zhrch/DiffusionSnake-12-30/data/outputs/volmem/verse_memflowdit_v0_3_2day_gpu6/eval_step_2250_autoregressive_gt/charts'

os.makedirs(OUT_DIR, exist_ok=True)

with open(SLICES_JSON) as f:
    data = json.load(f)

# Per-slice dataframe
volumes = {}
for r in data:
    cid = r['case_id']
    volumes.setdefault(cid, []).append(r)

for cid in volumes:
    volumes[cid].sort(key=lambda x: x['slice_idx'])

# 1. Volume-level metrics
vol_dice, vol_iou = [], []
vol_names = sorted(volumes.keys())
for cid in vol_names:
    slices = volumes[cid]
    fg_dice = [s['foreground_dice'] for s in slices if s.get('gt_foreground_pixels', 0) > 0]
    fg_iou = [s['foreground_iou'] for s in slices if s.get('gt_foreground_pixels', 0) > 0]
    vol_dice.append(float(np.mean(fg_dice)) if fg_dice else 1.0)
    vol_iou.append(float(np.mean(fg_iou)) if fg_iou else 1.0)

x = np.arange(len(vol_names))
fig, ax = plt.subplots(figsize=(14, 5))
width = 0.35
ax.bar(x - width/2, vol_dice, width, label='Foreground Dice', color='#2E86DE', alpha=0.85)
ax.bar(x + width/2, vol_iou, width, label='Foreground IoU', color='#A8D0F0', alpha=0.85)
ax.set_xlabel('Volume')
ax.set_ylabel('Metric')
ax.set_title('MemFlowDiT v0.3 step 2250: per-volume foreground Dice / IoU (GT box, autoregressive)')
ax.set_xticks(x[::2])
ax.set_xticklabels([vol_names[i] for i in range(0, len(vol_names), 2)], rotation=45, ha='right', fontsize=7)
ax.legend()
ax.set_ylim(0, 1.05)
ax.grid(axis='y', linestyle='--', alpha=0.4)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, '01_volume_metrics.png'), dpi=150)
plt.close()

# 2. Slice evolution for selected volumes (best, median, worst by mean dice)
mean_dice = {cid: float(np.mean([s['foreground_dice'] for s in vols if s.get('gt_foreground_pixels', 0) > 0] or [1.0]))
             for cid, vols in volumes.items()}
sorted_vols = sorted(mean_dice.items(), key=lambda x: x[1], reverse=True)
selected = [sorted_vols[0][0], sorted_vols[len(sorted_vols)//2][0], sorted_vols[-1][0]]

fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
colors = ['#2E86DE', '#F5A623', '#D0021B']
for ax, cid, color in zip(axes, selected, colors):
    sl = volumes[cid]
    xs = [s['slice_idx'] for s in sl]
    dice = [s['foreground_dice'] for s in sl]
    iou = [s['foreground_iou'] for s in sl]
    ax.plot(xs, dice, color=color, lw=1.5, label='Dice')
    ax.plot(xs, iou, color=color, lw=1.5, ls='--', alpha=0.7, label='IoU')
    ax.fill_between(xs, dice, alpha=0.1, color=color)
    ax.set_ylim(-0.05, 1.05)
    ax.set_ylabel('Metric')
    ax.set_title(f'{cid} (mean Dice={mean_dice[cid]:.3f})')
    ax.legend(loc='lower right')
    ax.grid(axis='y', linestyle='--', alpha=0.4)
axes[-1].set_xlabel('Slice index')
fig.suptitle('MemFlowDiT v0.3 step 2250: foreground Dice/IoU evolution across slices', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, '02_slice_evolution.png'), dpi=150)
plt.close()

# 3. Class-wise dice distribution (from class_dice dicts)
class_dice_sum = {}
class_counts = {}
for r in data:
    cd = r.get('class_dice', {})
    for cls, d in cd.items():
        class_dice_sum[cls] = class_dice_sum.get(cls, 0.0) + d
        class_counts[cls] = class_counts.get(cls, 0) + 1
classes = sorted(class_dice_sum.keys(), key=lambda c: int(c))
class_mean_dice = [class_dice_sum[c] / class_counts[c] for c in classes]

fig, ax = plt.subplots(figsize=(12, 5))
ax.bar(classes, class_mean_dice, color='#2E86DE', alpha=0.85)
ax.set_xlabel('Vertebra class')
ax.set_ylabel('Mean Dice')
ax.set_title('MemFlowDiT v0.3 step 2250: class-wise mean Dice')
ax.set_ylim(0, 1.05)
ax.grid(axis='y', linestyle='--', alpha=0.4)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, '03_class_dice.png'), dpi=150)
plt.close()

# 4. Histogram of per-slice foreground dice
all_fg_dice = [s['foreground_dice'] for s in data if s.get('gt_foreground_pixels', 0) > 0]
fig, ax = plt.subplots(figsize=(9, 5))
ax.hist(all_fg_dice, bins=50, color='#2E86DE', edgecolor='white', alpha=0.85)
ax.axvline(np.mean(all_fg_dice), color='#D0021B', ls='--', lw=2, label=f'mean={np.mean(all_fg_dice):.3f}')
ax.set_xlabel('Foreground Dice')
ax.set_ylabel('Slice count')
ax.set_title(f'MemFlowDiT v0.3 step 2250: distribution of per-slice foreground Dice (n={len(all_fg_dice)})')
ax.legend()
ax.grid(axis='y', linestyle='--', alpha=0.4)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, '04_slice_dice_hist.png'), dpi=150)
plt.close()

print('Charts saved to', OUT_DIR)
