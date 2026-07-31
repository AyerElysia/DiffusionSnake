#!/usr/bin/env python
"""Parse RL-V5 nohup logs and plot reward / burr / eval_iou curves for comparison.

Updated 2026-07-10: refreshed log set to reflect the currently-running lines
(curvmatch was stopped to free GPU4 for the perpoint v7 experiment; extrap and
delta_nsd continued past their earlier logs after resumes/OOM recoveries).
"""
import re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'Noto Sans CJK JP', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

LOGS = {
    'extrap_w1.0 (GPU6, 主线)': (
        'NOHUP_LOGS/geom8_extrap1p0_gpu6.log,'
        'NOHUP_LOGS/geom8_extrap1p0_gpu6_10k.log,'
        'NOHUP_LOGS/geom8_extrap1p0_gpu6_recovered.log'
    ),
    'curv_match验证 (GPU4, 已停止@3144)': 'NOHUP_LOGS/geom8_curvmatch_gpu4.log',
    'delta_nsd (GPU7→GPU4)': (
        'NOHUP_LOGS/geom8_delta_nsd_gpu7.log,'
        'NOHUP_LOGS/geom8_delta_nsd_gpu7_recovered.log,'
        'NOHUP_LOGS/geom8_delta_nsd_gpu4_10k.log'
    ),
    'perpoint_fmscale v7 grouped_adaptive_adv (GPU4)': (
        'NOHUP_LOGS/perpoint_fmscale_entropy_diag_gpu5.log,'
        'NOHUP_LOGS/perpoint_fmscale_v7_grouped_adaptive_adv_noentropy_gpu4.log'
    ),
    'perpoint_fmscale v8 grouped_centered (GPU5)': (
        'NOHUP_LOGS/perpoint_fmscale_entropy_diag_gpu5.log,'
        'NOHUP_LOGS/perpoint_fmscale_v8_grouped_centered_noentropy_gpu5.log'
    ),
    'perpoint_fmscale v9 grouped_localcredit (GPU7)': (
        'NOHUP_LOGS/perpoint_fmscale_entropy_diag_gpu5.log,'
        'NOHUP_LOGS/perpoint_fmscale_v9_grouped_localcredit_gpu7.log'
    ),
}

STEP_RE = re.compile(
    r'step=(\d+)/\d+\s+reward=([-+\d.]+)\s+(?:rstd=[-+\d.]+\s+)?best=([-+\d.]+)\s+dsig=[-+\d.]+\s+burr=([-+\d.]+).*?kl=([\d.]+)'
)
EVAL_RE = re.compile(r'eval_iou=([\d.]+)\s+mbf=([\d.]+)')

def parse(path_spec):
    steps, rewards, burrs, kls = [], [], [], []
    eval_steps, eval_ious = [], []
    seen_steps = set()
    for path in path_spec.split(','):
        try:
            with open(path, 'r', errors='ignore') as f:
                for line in f:
                    m = STEP_RE.search(line)
                    if not m:
                        continue
                    s = int(m.group(1))
                    if s in seen_steps:
                        continue
                    seen_steps.add(s)
                    steps.append(s)
                    rewards.append(float(m.group(2)))
                    burrs.append(float(m.group(4)))
                    kls.append(float(m.group(5)))
                    em = EVAL_RE.search(line)
                    if em:
                        eval_steps.append(s)
                        eval_ious.append(float(em.group(1)))
        except FileNotFoundError:
            pass
    order = sorted(range(len(steps)), key=lambda i: steps[i])
    steps = [steps[i] for i in order]
    rewards = [rewards[i] for i in order]
    burrs = [burrs[i] for i in order]
    kls = [kls[i] for i in order]
    ev_order = sorted(range(len(eval_steps)), key=lambda i: eval_steps[i])
    eval_steps = [eval_steps[i] for i in ev_order]
    eval_ious = [eval_ious[i] for i in ev_order]
    return steps, rewards, burrs, kls, eval_steps, eval_ious


def smooth(x, w=15):
    if len(x) < w:
        return x
    out = []
    for i in range(len(x)):
        lo = max(0, i - w + 1)
        out.append(sum(x[lo:i + 1]) / (i + 1 - lo))
    return out


fig, axes = plt.subplots(1, 3, figsize=(18, 6.8), sharex=False)

summary_lines = []
for name, path in LOGS.items():
    steps, rewards, burrs, kls, eval_steps, eval_ious = parse(path)
    if not steps:
        print(f'[skip] {name}: no data parsed from {path}')
        continue
    axes[0].plot(steps, smooth(rewards), label=f'{name} (n={len(steps)})')
    axes[1].plot(steps, smooth(burrs), label=name)
    if eval_steps:
        axes[2].plot(eval_steps, eval_ious, marker='o', markersize=3, label=name)
    line = (f'{name}: steps={steps[-1] if steps else 0}, '
            f'last_reward_smooth={smooth(rewards)[-1]:.4f}, '
            f'last_burr_smooth={smooth(burrs)[-1]:.4f}, '
            f'last_eval_iou={eval_ious[-1] if eval_ious else None}')
    print(line)
    summary_lines.append(line)

axes[0].set_title('Reward (smoothed, window=15)')
axes[0].set_xlabel('step')
axes[0].set_ylabel('reward')
axes[0].legend(fontsize=8)
axes[0].axhline(0, color='gray', lw=0.5)

axes[1].set_title('Burr penalty raw value (smoothed, window=15)')
axes[1].set_xlabel('step')
axes[1].set_ylabel('burr')
axes[1].legend(fontsize=8)

axes[2].set_title('Eval IoU')
axes[2].set_xlabel('step')
axes[2].set_ylabel('eval_iou')
axes[2].axhline(0.8546, color='gray', lw=0.8, ls='--', label='pretrain baseline 0.8546')
axes[2].legend(fontsize=8)

plt.tight_layout()
out_path = 'report/rl_curves_20260710.png'
plt.savefig(out_path, dpi=130)
print(f'Saved to {out_path}')

with open('report/rl_curves_20260710_summary.txt', 'w') as f:
    f.write('\n'.join(summary_lines) + '\n')
