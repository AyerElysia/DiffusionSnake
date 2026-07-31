import json, os, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

d = json.load(open('data/analysis/credit_diag/summary.json'))
os.makedirs('data/analysis/credit_diag', exist_ok=True)
steps = list(range(1, d['outer_steps']+1))

fig, axes = plt.subplots(2, 2, figsize=(13, 9))
ax1, ax2, ax3, ax4 = axes.flat

# Panel 1: partial score vs deterministic per step
ax1.plot(steps, d['det_partial_score_per_step_mean'], 'o-', color='#1f77b4', label='deterministic (per-step truncation)')
ax1.plot(steps, d['partial_score_per_step_mean'], 's-', color='#d62728', label='sampled rollout (mean of K=8)')
ax1.fill_between(steps, np.array(d['partial_score_per_step_mean']) - np.array(d['per_step_std_K_dispersion_mean']),
                            np.array(d['partial_score_per_step_mean']) + np.array(d['per_step_std_K_dispersion_mean']),
                 alpha=0.25, color='#d62728')
ax1.set_xlabel('outer step'); ax1.set_ylabel('partial score (IoU-weighted)');
ax1.set_title('Per-step truncation: sampled systematically below deterministic')
ax1.legend(); ax1.grid(alpha=0.3)

# Panel 2: fraction of rollouts better than det per step (info that a per-step gate would use)
ax2.bar(steps, d['per_step_frac_rollouts_better_than_det'], color='#2ca02c', alpha=0.7)
ax2.axhline(d['gate_active_frac_mean'], color='k', linestyle='--', label=f"terminal gate frac={d['gate_active_frac_mean']:.3f}")
ax2.set_ylim(0, 1); ax2.set_xlabel('outer step'); ax2.set_ylabel('fraction of K=8 rollouts beating det (at this step)')
ax2.set_title('Per-step gate would NOT match terminal gate: improves monotonically\nwith step no. so terminal gate kills more groups')
ax2.legend(); ax2.grid(alpha=0.3)

# Panel 3: sign-agreement step-vs-terminal + spearman
ax3.plot(steps, d['per_step_frac_sign_agree_per_step_mean'], '^-', color='#ff7f0e', label='sign(adv_step) == sign(adv_terminal)')
ax3.plot(steps, d['per_step_corr_partial_vs_terminal_mean']['spearman_per_step_mean'], 'D-', color='#9467bd', label='spearman(step_partial, terminal_partial)')
ax3.axhline(0.5, color='grey', linestyle=':', alpha=0.5)
ax3.set_ylim(0, 1.05); ax3.set_xlabel('outer step'); ax3.set_ylabel('agreement / correlation')
ax3.set_title('Credit mismatch: early steps correlated only ~0.34-0.52 with terminal\n25% of step-1 updates carry the WRONG direction')
ax3.legend(); ax3.grid(alpha=0.3)

# Panel 4: terminal quality distribution + gate frac per file
n_files = d['n_files']
files = sorted(__import__('glob').glob('data/outputs/1232_final_v5_geom8_baseline_bs6_gpu2/credit_diag_step*.json'),
                key=lambda p: int(''.join(c for c in os.path.basename(p) if c.isdigit())))
q_per_rollout_per_file = [np.array(json.load(open(f))['terminal_recompute']['all_krollouts_quality_per_contour_mean']) for f in files]
q_pos_frac = [(q>=0).mean() for q in q_per_rollout_per_file]
ax4.bar(range(1, n_files+1), q_pos_frac, color='#8c564b', alpha=0.7, label='frac rollouts with terminal_q>0')
ax4.axhline(np.mean(q_pos_frac), color='r', linestyle='--', label=f'mean={np.mean(q_pos_frac):.3f}')
ax4.set_xlabel('training batch index (diag step)'); ax4.set_ylabel('fraction rollouts terminal-positive')
ax4.set_title(f'Terminal reward \"negative\" is the norm: only {np.mean(q_pos_frac):.0%} of rollouts ever get >0')
ax4.legend(); ax4.grid(alpha=0.3)

fig.suptitle('RL V5 credit-assignment diagnostic — 20 batches x K=8 rollouts x 5 outer steps\n(repro of baseline_run best_iou.pt config)', y=0.995, fontsize=12)
fig.tight_layout()
out_png = 'data/analysis/credit_diag/credit_diag_evidence.png'
fig.savefig(out_png, dpi=130, bbox_inches='tight')
print('wrote', out_png)