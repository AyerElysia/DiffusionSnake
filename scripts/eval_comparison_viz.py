"""
Comparison visualization: Baseline vs GRPO V2 Best
Generates:
1. eval_comparison_summary.png  — bar + scatter plots
2. eval_comparison_gallery.png  — side-by-side overlays for 12 samples
3. eval_comparison_report.txt   — text summary
"""
import json, os, sys, glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

BASELINE_JSON  = os.path.join(ROOT, 'visual/eval_baseline/summary_rows_20260516_191403.json')
BEST_JSON      = os.path.join(ROOT, 'visual/eval_grpo_v2_best/summary_rows_20260516_191857.json')
BASE_META_JSON = os.path.join(ROOT, 'visual/eval_baseline/v3_7_full_test_iou_20260516_191403.json')
BEST_META_JSON = os.path.join(ROOT, 'visual/eval_grpo_v2_best/v3_7_full_test_iou_20260516_191857.json')
OUT_DIR = os.path.join(ROOT, 'visual/eval_comparison')
os.makedirs(OUT_DIR, exist_ok=True)

# ── load data ───────────────────────────────────────────────────────────────
with open(BASELINE_JSON) as f:  base_d = json.load(f)
with open(BEST_JSON)     as f:  best_d = json.load(f)
with open(BASE_META_JSON) as f: base_meta = json.load(f)
with open(BEST_META_JSON) as f: best_meta = json.load(f)

base_rows = base_d['rows']
best_rows = best_d['rows']

idx_base = {r['index']: r for r in base_rows}
idx_best = {r['index']: r for r in best_rows}
common   = sorted(set(idx_base) & set(idx_best))

b_iou   = np.array([idx_base[i]['mean_iou']      for i in common])
r_iou   = np.array([idx_best[i]['mean_iou']       for i in common])
b_dice  = np.array([idx_base[i]['mean_dice']      for i in common])
r_dice  = np.array([idx_best[i]['mean_dice']      for i in common])
b_mbf   = np.array([idx_base[i]['mean_mboundf']   for i in common])
r_mbf   = np.array([idx_best[i]['mean_mboundf']   for i in common])

delta_iou = r_iou - b_iou  # positive = RL better

print(f"Samples compared: {len(common)}")
print(f"Baseline  | IoU={b_iou.mean():.4f}  Dice={b_dice.mean():.4f}  BoundF={b_mbf.mean():.4f}")
print(f"V2 Best   | IoU={r_iou.mean():.4f}  Dice={r_dice.mean():.4f}  BoundF={r_mbf.mean():.4f}")
print(f"Delta IoU | mean={delta_iou.mean():.4f}  std={delta_iou.std():.4f}")
print(f"  Samples improved: {(delta_iou>0).sum()} / {len(common)}")
print(f"  Samples degraded: {(delta_iou<0).sum()} / {len(common)}")

# ── figure 1: summary charts ────────────────────────────────────────────────
fig = plt.figure(figsize=(18, 13))
gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

# 1a. metric bar chart
ax1 = fig.add_subplot(gs[0, 0])
metrics = ['IoU', 'Dice', 'BoundF']
base_vals = [b_iou.mean(), b_dice.mean(), b_mbf.mean()]
best_vals = [r_iou.mean(), r_dice.mean(), r_mbf.mean()]
x = np.arange(len(metrics))
w = 0.35
bars1 = ax1.bar(x - w/2, base_vals, w, label='Baseline', color='steelblue', alpha=0.85)
bars2 = ax1.bar(x + w/2, best_vals, w, label='V2 Best (step1625)', color='darkorange', alpha=0.85)
ax1.set_xticks(x); ax1.set_xticklabels(metrics)
ax1.set_ylim(0.7, 1.0); ax1.set_ylabel('Score')
ax1.set_title('Overall Metrics Comparison', fontweight='bold')
ax1.legend(fontsize=8)
for bar in bars1: ax1.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.003, f'{bar.get_height():.4f}', ha='center', va='bottom', fontsize=7)
for bar in bars2: ax1.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.003, f'{bar.get_height():.4f}', ha='center', va='bottom', fontsize=7)

# 1b. per-sample IoU scatter
ax2 = fig.add_subplot(gs[0, 1])
scatter = ax2.scatter(b_iou, r_iou, c=delta_iou, cmap='RdYlGn', vmin=-0.05, vmax=0.05, alpha=0.6, s=18)
lims = [min(b_iou.min(), r_iou.min())-0.01, max(b_iou.max(), r_iou.max())+0.01]
ax2.plot(lims, lims, 'k--', lw=1, label='y=x (no change)')
ax2.set_xlabel('Baseline IoU'); ax2.set_ylabel('V2 Best IoU')
ax2.set_title('Per-Sample IoU (color = Δ IoU)', fontweight='bold')
plt.colorbar(scatter, ax=ax2, label='ΔIoU (RL-Base)')
ax2.legend(fontsize=7)

# 1c. delta IoU histogram
ax3 = fig.add_subplot(gs[0, 2])
ax3.hist(delta_iou, bins=30, color='slateblue', edgecolor='white', alpha=0.85)
ax3.axvline(0, color='red', lw=1.5, linestyle='--', label='zero')
ax3.axvline(delta_iou.mean(), color='orange', lw=1.5, linestyle='-', label=f'mean={delta_iou.mean():.4f}')
ax3.set_xlabel('ΔIoU (V2 Best − Baseline)'); ax3.set_ylabel('Count')
ax3.set_title('Distribution of Per-Sample IoU Change', fontweight='bold')
ax3.legend(fontsize=8)
improved = (delta_iou > 0.01).sum()
degraded = (delta_iou < -0.01).sum()
neutral  = len(delta_iou) - improved - degraded
ax3.text(0.02, 0.95, f'Improved(>+1%): {improved}\nNeutral: {neutral}\nDegraded(<-1%): {degraded}',
         transform=ax3.transAxes, va='top', fontsize=8,
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

# 1d. per-sample IoU sorted
ax4 = fig.add_subplot(gs[1, :])
sort_idx = np.argsort(delta_iou)
sample_ids = np.array(common)[sort_idx]
di_sorted  = delta_iou[sort_idx]
colors = ['limegreen' if d > 0 else 'tomato' for d in di_sorted]
ax4.bar(range(len(di_sorted)), di_sorted, color=colors, alpha=0.7, width=1.0)
ax4.axhline(0, color='black', lw=0.8)
ax4.set_xlabel('Samples (sorted by ΔIoU)'); ax4.set_ylabel('ΔIoU')
ax4.set_title('Per-Sample IoU Change: V2 Best − Baseline  (green=improved, red=degraded)', fontweight='bold')
# annotate worst/best
for rank, (sid, di) in enumerate(zip(sample_ids[:5], di_sorted[:5])):
    ax4.text(rank, di - 0.003, f's{sid}', ha='center', va='top', fontsize=6, color='darkred')
for rank, (sid, di) in enumerate(zip(sample_ids[-5:], di_sorted[-5:])):
    r = len(di_sorted) - 5 + rank
    ax4.text(r, di + 0.002, f's{sid}', ha='center', va='bottom', fontsize=6, color='darkgreen')

# 1e. boxplots
ax5 = fig.add_subplot(gs[2, 0])
ax5.boxplot([b_iou, r_iou], labels=['Baseline', 'V2 Best'], patch_artist=True,
            boxprops=dict(facecolor='lightblue'), medianprops=dict(color='red', lw=2))
ax5.set_title('IoU Distribution', fontweight='bold'); ax5.set_ylabel('IoU')

ax6 = fig.add_subplot(gs[2, 1])
ax6.boxplot([b_dice, r_dice], labels=['Baseline', 'V2 Best'], patch_artist=True,
            boxprops=dict(facecolor='lightgreen'), medianprops=dict(color='red', lw=2))
ax6.set_title('Dice Distribution', fontweight='bold'); ax6.set_ylabel('Dice')

ax7 = fig.add_subplot(gs[2, 2])
ax7.boxplot([b_mbf, r_mbf], labels=['Baseline', 'V2 Best'], patch_artist=True,
            boxprops=dict(facecolor='lightyellow'), medianprops=dict(color='red', lw=2))
ax7.set_title('BoundF Distribution', fontweight='bold'); ax7.set_ylabel('BoundF')

fig.suptitle('BTCV V3.4-FM  ·  Baseline vs GRPO-V2 Best (step=1625)  ·  Full 150-Sample Evaluation',
             fontsize=13, fontweight='bold', y=0.98)
out1 = os.path.join(OUT_DIR, 'eval_comparison_summary.png')
fig.savefig(out1, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"[✔] Saved: {out1}")

# ── figure 2: gallery of 12 samples (6 most-improved + 6 most-degraded) ─────
def load_overlay(row):
    """Load overlay image from per_sample dir."""
    d = row.get('dir', '')
    if not d:
        return None
    # absolute path
    if not os.path.isabs(d):
        d = os.path.join(ROOT, d)
    candidates = [
        os.path.join(d, 'overlay.png'),
        os.path.join(d, 'overlay_contour.png'),
    ]
    for c in candidates:
        if os.path.exists(c):
            return np.array(Image.open(c).convert('RGB'))
    # fallback: any PNG
    pngs = glob.glob(os.path.join(d, '*.png'))
    if pngs:
        return np.array(Image.open(pngs[0]).convert('RGB'))
    return None

# pick top 6 improved and top 6 degraded
top_improved_idx = sort_idx[-6:][::-1]   # best delta first
top_degraded_idx = sort_idx[:6]          # worst delta first

def make_gallery_row(ax_row, sample_indices, label):
    for col, si in enumerate(sample_indices):
        sid = common[si]
        b_img = load_overlay(idx_base[sid])
        r_img = load_overlay(idx_best[sid])
        ax_b = ax_row[col*2]
        ax_r = ax_row[col*2+1]
        if b_img is not None:
            ax_b.imshow(b_img); ax_b.axis('off')
        else:
            ax_b.text(0.5,0.5,'N/A',ha='center',va='center',transform=ax_b.transAxes); ax_b.axis('off')
        if r_img is not None:
            ax_r.imshow(r_img); ax_r.axis('off')
        else:
            ax_r.text(0.5,0.5,'N/A',ha='center',va='center',transform=ax_r.transAxes); ax_r.axis('off')
        d_val = delta_iou[si]
        ax_b.set_title(f's{sid} Base\nIoU={b_iou[si]:.3f}', fontsize=6, pad=2)
        ax_r.set_title(f's{sid} RL\nIoU={r_iou[si]:.3f} (Δ{d_val:+.3f})', fontsize=6, pad=2,
                        color='green' if d_val > 0 else 'red')

n_cols = 6
fig2, axes = plt.subplots(4, n_cols * 2, figsize=(n_cols * 4, 14))
fig2.subplots_adjust(hspace=0.5, wspace=0.05)

make_gallery_row(axes[0], top_improved_idx[:3], 'improved')
make_gallery_row(axes[1], top_improved_idx[3:], 'improved')
make_gallery_row(axes[2], top_degraded_idx[:3], 'degraded')
make_gallery_row(axes[3], top_degraded_idx[3:], 'degraded')

# row labels
for ax_row, lbl, color in [(axes[0], '▲ Top 6 Most Improved (rows 1-2)', 'darkgreen'),
                             (axes[1], '', 'darkgreen'),
                             (axes[2], '▼ Top 6 Most Degraded (rows 3-4)', 'darkred'),
                             (axes[3], '', 'darkred')]:
    if lbl:
        ax_row[0].set_ylabel(lbl, fontsize=8, color=color, rotation=90, labelpad=4)

fig2.suptitle('Gallery: Baseline vs V2 Best  |  Left=Baseline, Right=RL (Δ shown in title)',
              fontsize=11, fontweight='bold')
out2 = os.path.join(OUT_DIR, 'eval_comparison_gallery.png')
fig2.savefig(out2, dpi=120, bbox_inches='tight')
plt.close(fig2)
print(f"[✔] Saved: {out2}")

# ── text report ─────────────────────────────────────────────────────────────
rpt = os.path.join(OUT_DIR, 'eval_comparison_report.txt')
with open(rpt, 'w') as f:
    f.write("=" * 70 + "\n")
    f.write("BTCV V3.4-FM  Baseline vs GRPO-V2 Best  Full Evaluation Report\n")
    f.write("=" * 70 + "\n\n")
    f.write(f"Eval date      : 2026-05-16\n")
    f.write(f"Test set       : BTCV 150 samples (btcv_png_test_new_snake)\n")
    f.write(f"ODE steps      : 10\n")
    f.write(f"Config         : btcv_v3_4_fm_grpo_v2_gpu1.yaml (YOLO-m)\n\n")
    f.write("-" * 70 + "\n")
    f.write("MODEL                  IoU       Dice      BoundF\n")
    f.write("-" * 70 + "\n")
    f.write(f"Baseline (latest.pt)   {b_iou.mean():.4f}    {b_dice.mean():.4f}    {b_mbf.mean():.4f}\n")
    f.write(f"V2 Best  (step 1625)   {r_iou.mean():.4f}    {r_dice.mean():.4f}    {r_mbf.mean():.4f}\n")
    delta_b = r_iou.mean()-b_iou.mean(); delta_d=r_dice.mean()-b_dice.mean(); delta_m=r_mbf.mean()-b_mbf.mean()
    f.write(f"Delta                  {delta_b:+.4f}   {delta_d:+.4f}   {delta_m:+.4f}\n")
    f.write(f"Delta %                {delta_b/b_iou.mean()*100:+.2f}%     {delta_d/b_dice.mean()*100:+.2f}%     {delta_m/b_mbf.mean()*100:+.2f}%\n\n")
    f.write("-" * 70 + "\n")
    f.write(f"Per-sample IoU change statistics:\n")
    f.write(f"  Samples improved (ΔIoU > 0)     : {(delta_iou>0).sum()} / {len(delta_iou)}\n")
    f.write(f"  Samples improved (ΔIoU > +1%)   : {(delta_iou>0.01).sum()} / {len(delta_iou)}\n")
    f.write(f"  Samples degraded (ΔIoU < 0)     : {(delta_iou<0).sum()} / {len(delta_iou)}\n")
    f.write(f"  Samples degraded (ΔIoU < -1%)   : {(delta_iou<-0.01).sum()} / {len(delta_iou)}\n")
    f.write(f"  Mean ΔIoU                        : {delta_iou.mean():+.4f}\n")
    f.write(f"  Std  ΔIoU                        : {delta_iou.std():.4f}\n")
    f.write(f"  Best single improvement          : +{delta_iou.max():.4f} (sample {common[delta_iou.argmax()]})\n")
    f.write(f"  Worst single degradation         : {delta_iou.min():+.4f} (sample {common[delta_iou.argmin()]})\n\n")
    f.write("-" * 70 + "\n")
    f.write("KEY FINDING:\n")
    f.write("  The in-training eval metric (eval_iou) reported 0.8357 → 0.9012 (+7.8%).\n")
    f.write("  However, the full 150-sample evaluation reveals:\n")
    f.write(f"    Baseline (full test): {b_iou.mean():.4f}\n")
    f.write(f"    V2 Best  (full test): {r_iou.mean():.4f}  (Δ = {delta_b:+.4f})\n")
    f.write("\n")
    f.write("  The small eval batch used during training was NOT representative of the\n")
    f.write("  full test distribution. The in-training 'improvement' was largely an\n")
    f.write("  artifact of the specific samples selected for the eval batch.\n\n")
    f.write("  The RL training achieved modest per-sample variance: about half the\n")
    f.write(f"  samples improved, half degraded. The net effect is a very small\n")
    f.write("  degradation of -0.7% IoU on the full test set.\n\n")
    f.write("RECOMMENDATION:\n")
    f.write("  1. Use a LARGER, FIXED held-out eval set (≥50 samples) during training\n")
    f.write("     to get a reliable signal. The current small batch is too noisy.\n")
    f.write("  2. Consider using the FULL test set mean IoU as the RL reward, not\n")
    f.write("     a surrogate that diverges from the actual metric.\n")
    f.write("  3. Alternatively, train for more steps and monitor the full eval curve\n")
    f.write("     periodically (every 500 steps) rather than relying on a proxy.\n")
    f.write("=" * 70 + "\n")

print(f"[✔] Saved: {rpt}")
with open(rpt) as f: print(f.read())
