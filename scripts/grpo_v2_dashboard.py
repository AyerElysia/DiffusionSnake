"""GRPO V2 monitoring dashboard.

Reads `posttrain_grpo_v2/logs.jsonl` and produces a multipanel PNG dashboard
showing the *health* of the RL run, not just raw rewards.

Usage:
    python scripts/grpo_v2_dashboard.py <log_dir>
    # log_dir defaults to data/outputs/btcv_v3_4_fm_grpo_v2_gpu1/posttrain_grpo_v2
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def load_logs(jsonl_path: Path):
    rows = []
    with open(jsonl_path) as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except Exception:
                pass
    return rows


def col(rows, key, default=0.0):
    xs = [r.get(key, default) for r in rows]
    return np.asarray(xs, dtype=np.float32)


def sparse_col(rows, key):
    steps = [r['step'] for r in rows if key in r]
    vals = [r[key] for r in rows if key in r]
    return np.asarray(steps), np.asarray(vals, dtype=np.float32)


def smoothed(y, win=20):
    if len(y) < 3:
        return y
    w = min(win, max(3, len(y) // 8))
    k = np.ones(w) / w
    return np.convolve(y, k, mode='same')


def main(log_dir: Path):
    log_path = log_dir / 'logs.jsonl'
    if not log_path.exists():
        print(f'no log file at {log_path}')
        sys.exit(1)
    rows = load_logs(log_path)
    if len(rows) < 2:
        print(f'too few rows ({len(rows)}); skipping')
        return
    steps = col(rows, 'step')

    fig, axes = plt.subplots(3, 3, figsize=(18, 12))
    fig.suptitle(f'GRPO V2 monitor — {log_dir}  ({len(rows)} steps)', fontsize=14)

    # 1. Reward curve (mean / best / p10-p90 band)
    ax = axes[0, 0]
    r_mean = col(rows, 'reward_mean'); r_best = col(rows, 'reward_best')
    r_p10 = col(rows, 'reward_p10'); r_p90 = col(rows, 'reward_p90')
    ax.fill_between(steps, r_p10, r_p90, color='tab:blue', alpha=0.15, label='p10–p90')
    ax.plot(steps, r_mean, color='tab:blue', label='reward_mean')
    ax.plot(steps, r_best, color='tab:green', label='reward_best')
    ax.plot(steps, smoothed(r_mean), color='tab:red', lw=1.0, label='smoothed mean')
    ax.set_title('Reward'); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # 2. Absolute scores (final / delta)
    ax = axes[0, 1]
    fs = col(rows, 'final_score_mean'); ds = col(rows, 'delta_score_mean')
    ax.plot(steps, fs, color='tab:blue', label='final_score (abs)')
    ax.plot(steps, ds, color='tab:purple', label='delta_score (final-init)')
    ax.axhline(0, color='gray', lw=0.5)
    ax.set_title('Score components'); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # 3. Eval IoU / mBoundF (sparse - only when computed)
    ax = axes[0, 2]
    s_iou, v_iou = sparse_col(rows, 'eval_iou')
    s_mbf, v_mbf = sparse_col(rows, 'eval_mboundf')
    s_dc, v_dc = sparse_col(rows, 'eval_dice')
    if len(s_iou) > 0:
        ax.plot(s_iou, v_iou, 'o-', color='tab:blue', label='eval IoU', markersize=3)
    if len(s_mbf) > 0:
        ax.plot(s_mbf, v_mbf, 's-', color='tab:red', label='eval mBoundF', markersize=3)
    if len(s_dc) > 0:
        ax.plot(s_dc, v_dc, '^-', color='tab:green', label='eval Dice', markersize=3)
    ax.set_title('Held-out eval (deterministic)'); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # 4. approx_kl: this is the most critical "is PPO working?" plot
    ax = axes[1, 0]
    kl_f = col(rows, 'approx_kl_first'); kl_l = col(rows, 'approx_kl_last')
    ax.plot(steps, kl_f, color='tab:blue', label='approx_kl first inner')
    ax.plot(steps, kl_l, color='tab:red', label='approx_kl last inner')
    ax.set_title('approx KL per update'); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    ax.set_yscale('symlog', linthresh=1e-4)

    # 5. clipfrac
    ax = axes[1, 1]
    cf = col(rows, 'clipfrac_last')
    ax.plot(steps, cf, color='tab:orange'); ax.set_ylim(0, 1)
    ax.set_title('clipfrac (last inner epoch)'); ax.grid(alpha=0.3)

    # 6. ratio range
    ax = axes[1, 2]
    rmin = col(rows, 'ratio_min'); rmax = col(rows, 'ratio_max'); rmean = col(rows, 'ratio_mean')
    ax.fill_between(steps, rmin, rmax, color='tab:gray', alpha=0.3, label='[min,max]')
    ax.plot(steps, rmean, color='black', label='mean')
    ax.axhline(1.0, color='tab:red', lw=0.5)
    ax.set_title('Importance ratio'); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # 7. policy loss / kl loss
    ax = axes[2, 0]
    pl = col(rows, 'policy_loss'); kll = col(rows, 'kl_loss')
    ax.plot(steps, pl, color='tab:blue', label='policy_loss')
    ax.plot(steps, kll, color='tab:red', label='kl_loss')
    ax.set_title('Loss terms'); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # 8. grad norm & action std
    ax = axes[2, 1]
    gn = col(rows, 'grad_norm'); act = col(rows, 'action_std')
    ax2 = ax.twinx()
    ax.plot(steps, gn, color='tab:blue', label='grad_norm')
    ax2.plot(steps, act, color='tab:green', label='action_std')
    ax.set_title('Grad norm vs action std'); ax.grid(alpha=0.3)
    ax.set_ylabel('grad_norm', color='tab:blue'); ax2.set_ylabel('action_std', color='tab:green')

    # 9. Inner epochs used (KL early stop usage)
    ax = axes[2, 2]
    ie = col(rows, 'inner_epochs')
    ax.plot(steps, ie, color='tab:purple')
    ax.set_title('Inner epochs used (early stop active when < max)')
    ax.grid(alpha=0.3); ax.set_ylim(0, max(ie.max() + 0.5, 4))

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_path = log_dir / 'dashboard.png'
    fig.savefig(str(out_path), dpi=110)
    plt.close(fig)
    print(f'wrote {out_path}')

    # also write a slim summary JSON
    last = rows[-1]
    summary = {
        'rows': len(rows),
        'last_step': int(last.get('step', -1)),
        'reward_mean_last': float(last.get('reward_mean', 0)),
        'final_score_last': float(last.get('final_score_mean', 0)),
        'eval_iou_last': float(last.get('eval_iou', float('nan'))) if 'eval_iou' in last else None,
        'approx_kl_first_last': float(last.get('approx_kl_first', 0)),
        'clipfrac_last_last': float(last.get('clipfrac_last', 0)),
        'ratio_range_last': [float(last.get('ratio_min', 1)), float(last.get('ratio_max', 1))],
        'grad_norm_last': float(last.get('grad_norm', 0)),
        'action_std_last': float(last.get('action_std', 0)),
    }
    with open(log_dir / 'dashboard_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    default_dir = Path(__file__).resolve().parents[1] / 'data' / 'outputs' / 'btcv_v3_4_fm_grpo_v2_gpu1' / 'posttrain_grpo_v2'
    log_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else default_dir
    main(log_dir)
