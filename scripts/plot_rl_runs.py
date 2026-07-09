#!/usr/bin/env python
"""Parse RL-V5 nohup logs and plot reward / burr / eval_iou curves for comparison."""
import re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

LOGS = {
    # GPU6 resumed from step600 checkpoint: pre-600 steps live in the old log,
    # 600+ steps live in the new "_10k" log. Concatenate both (comma-separated).
    'extrap_w1.0 (GPU6, burr=0.06)': (
        'NOHUP_LOGS/geom8_extrap1p0_gpu6.log,NOHUP_LOGS/geom8_extrap1p0_gpu6_10k.log'
    ),
    'extrap_w1.0 noburr (GPU0)': 'NOHUP_LOGS/geom8_extrap1p0_noburr_gpu0.log',
    'delta_nsd (GPU7)': 'NOHUP_LOGS/geom8_delta_nsd_gpu7.log',
    'seq_delta (GPU1)': 'NOHUP_LOGS/geom8_seqdelta_gpu1.log',
    'perpoint_fmscale entropy+diag (GPU5)': 'NOHUP_LOGS/perpoint_fmscale_entropy_diag_gpu5.log',
}

STEP_RE = re.compile(
    r'step=(\d+)/\d+\s+reward=([-+\d.]+)\s+best=([-+\d.]+)\s+dsig=[-+\d.]+\s+burr=([-+\d.]+).*?kl=([\d.]+)'
)
EVAL_RE = re.compile(r'eval_iou=([\d.]+)\s+mbf=([\d.]+)')

def parse(path_spec):
    # path_spec may be a comma-separated list of files to concatenate in order
    # (e.g. resumed runs where pre-resume steps live in an older log file).
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
                        continue  # dedupe overlap between old/new log at resume boundary
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
    # sort by step in case concatenation order didn't match numeric order
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


fig, axes = plt.subplots(3, 1, figsize=(11, 12), sharex=False)

for name, path in LOGS.items():
    steps, rewards, burrs, kls, eval_steps, eval_ious = parse(path)
    if not steps:
        print(f'[skip] {name}: no data parsed from {path}')
        continue
    axes[0].plot(steps, smooth(rewards), label=f'{name} (n={len(steps)})')
    axes[1].plot(steps, smooth(burrs), label=name)
    if eval_steps:
        axes[2].plot(eval_steps, eval_ious, marker='o', label=name)
    print(f'{name}: steps={steps[-1] if steps else 0}, '
          f'last_reward_smooth={smooth(rewards)[-1]:.4f}, '
          f'last_burr_smooth={smooth(burrs)[-1]:.4f}, '
          f'last_eval_iou={eval_ious[-1] if eval_ious else None}')

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
axes[2].legend(fontsize=8)
axes[2].axhline(0.8546, color='gray', lw=0.8, ls='--', label='pretrain baseline 0.8546')

plt.tight_layout()
out_path = 'report/rl_runs_comparison.png'
plt.savefig(out_path, dpi=130)
print(f'Saved to {out_path}')
