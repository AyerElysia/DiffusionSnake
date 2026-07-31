#!/usr/bin/env python3
"""Offline comparison: terminal-only vs per-step-blended advantage.

The credit-diag probe runs BEFORE the PPO update and always dumps the raw
per-rollout truncation scores. That raw signal is identical regardless of how
we later assign advantage, so we cannot see the effect of per_step_reward_weight
by re-running training under the STOP probe. Instead we reconstruct both
advantage schemes offline from the dumped JSON and measure how well each one's
per-step advantage SIGN agrees with the step's true local contribution.

Definitions (per contour, per training step JSON):
  partials[ri]      : cumulative quality at each truncation for rollout ri  (len n_outer)
  det_partials      : cumulative quality at each truncation for deterministic (len n_outer)
  partial_init      : quality at init (before any step)

  true local contribution of step si for rollout ri:
      loc[ri, si] = partials[ri][si] - partials[ri][si-1]      (si>0)
                    partials[ri][0]  - partial_init            (si==0)
  det local contribution:
      det_loc[si] = det_partials[si] - det_partials[si-1] / - partial_init

  terminal advantage (current baseline):
      q_term[ri]   = partials[ri][-1] - det_partials[-1]
      adv_term[ri] = clamp( q_term[ri] / std_K(q_term).clamp_min(0.1) ) * gate_term
      -> same scalar copied to every step si

  per-step blended advantage (proposed, weight w):
      step_q[ri,si]   = loc[ri,si] - det_loc[si]
      step_adv[ri,si] = clamp( step_q[ri,si] / std_K(step_q[:,si]).clamp_min(0.1) ) * gate_step[si]
      adv_ps[ri,si]   = (1-w)*adv_term[ri] + w*step_adv[ri,si]

sign_agree at step si = fraction of rollouts where sign(adv) == sign(loc - det_loc).
This is exactly the diagnostic metric §2.3 of the credit report tracks.
"""
import argparse
import glob
import json
import os

import numpy as np

ADV_CLIP_MAX = 2.0
STD_FLOOR = 0.1
GATE_MARGIN = 0.0


def _std_K(x):
    # population std across the K axis (axis 0)
    return np.std(x, axis=0)


def _clamp(x, lo, hi):
    return np.clip(x, lo, hi)


def load_step(path):
    with open(path) as fh:
        d = json.load(fh)
    rollouts = d['rollouts']
    n_outer = len(d['fractions'])
    K = len(rollouts)
    if K == 0:
        return None
    # partials: [K, n_outer]
    partials = np.array([r['partials'] for r in rollouts], dtype=np.float64)
    if partials.shape[1] != n_outer:
        n_outer = partials.shape[1]
    det_partials = np.array(rollouts[0]['det_partials'], dtype=np.float64)  # shared
    partial_init = float(rollouts[0].get('partial_init', det_partials[0]))
    return {
        'partials': partials,           # [K, n_outer]
        'det_partials': det_partials,   # [n_outer]
        'partial_init': partial_init,
        'K': K,
        'n_outer': n_outer,
    }


def local_contribs(partials, det_partials, partial_init):
    K, n_outer = partials.shape
    # prepend init column
    prev = np.concatenate([np.full((K, 1), partial_init), partials[:, :-1]], axis=1)
    loc = partials - prev                       # [K, n_outer]
    det_prev = np.concatenate([[partial_init], det_partials[:-1]])
    det_loc = det_partials - det_prev           # [n_outer]
    return loc, det_loc


def terminal_adv(partials, det_partials):
    K, n_outer = partials.shape
    q_term = partials[:, -1] - det_partials[-1]           # [K]
    std = max(np.std(q_term), STD_FLOOR)
    adv = _clamp(q_term / std, -ADV_CLIP_MAX, ADV_CLIP_MAX)
    gate = 1.0 if q_term.max() > GATE_MARGIN else 0.0
    adv = adv * gate
    # broadcast to steps
    return np.repeat(adv[:, None], n_outer, axis=1)       # [K, n_outer]


def perstep_adv(loc, det_loc):
    K, n_outer = loc.shape
    step_q = loc - det_loc[None, :]                       # [K, n_outer]
    std = np.maximum(_std_K(step_q), STD_FLOOR)           # [n_outer]
    adv = _clamp(step_q / std[None, :], -ADV_CLIP_MAX, ADV_CLIP_MAX)
    gate = (step_q.max(axis=0) > GATE_MARGIN).astype(np.float64)  # [n_outer]
    adv = adv * gate[None, :]
    return adv


def sign_agree(adv, loc, det_loc):
    """fraction of (K) rollouts whose advantage sign matches true local contribution sign, per step."""
    K, n_outer = adv.shape
    true_dir = np.sign(loc - det_loc[None, :])            # [K, n_outer]
    adv_dir = np.sign(adv)
    # only count entries where both signs are nonzero
    mask = (true_dir != 0) & (adv_dir != 0)
    agree = (adv_dir == true_dir) & mask
    out = []
    for si in range(n_outer):
        m = mask[:, si].sum()
        out.append(float(agree[:, si].sum() / m) if m > 0 else float('nan'))
    return np.array(out)


def spearman_step_vs_terminal(partials, det_partials, partial_init):
    """Spearman corr between per-step local contribution and terminal quality across K, per step."""
    from scipy.stats import spearmanr
    K, n_outer = partials.shape
    loc, det_loc = local_contribs(partials, det_partials, partial_init)
    q_term = partials[:, -1] - det_partials[-1]           # [K]
    out = []
    for si in range(n_outer):
        step_q = loc[:, si] - det_loc[si]
        if np.std(step_q) < 1e-9 or np.std(q_term) < 1e-9:
            out.append(float('nan'))
            continue
        rho, _ = spearmanr(step_q, q_term)
        out.append(float(rho))
    return np.array(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--diag_dir', required=True)
    ap.add_argument('--weights', type=float, nargs='+', default=[0.0, 0.5, 1.0])
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.diag_dir, 'credit_diag_step*.json')))
    if not files:
        print(f'no credit_diag_step*.json under {args.diag_dir}')
        return
    print(f'loaded {len(files)} step files from {args.diag_dir}')

    steps = [load_step(f) for f in files]
    steps = [s for s in steps if s is not None]

    n_outer = steps[0]['n_outer']

    # accumulate sign_agree per step for terminal and each weight
    has_scipy = True
    try:
        import scipy.stats  # noqa
    except Exception:
        has_scipy = False

    agree_term = []
    agree_w = {w: [] for w in args.weights}
    spearman_acc = []

    for s in steps:
        loc, det_loc = local_contribs(s['partials'], s['det_partials'], s['partial_init'])
        a_term = terminal_adv(s['partials'], s['det_partials'])
        a_ps = perstep_adv(loc, det_loc)
        agree_term.append(sign_agree(a_term, loc, det_loc))
        for w in args.weights:
            a_blend = (1.0 - w) * a_term + w * a_ps
            agree_w[w].append(sign_agree(a_blend, loc, det_loc))
        if has_scipy:
            spearman_acc.append(
                spearman_step_vs_terminal(s['partials'], s['det_partials'], s['partial_init'])
            )

    agree_term = np.nanmean(np.stack(agree_term, 0), 0)
    print('\n=== per-step SIGN-AGREE (advantage sign vs true local contribution) ===')
    print('step:                 ' + '  '.join(f's{i+1}' for i in range(n_outer)))
    print('terminal-only (w=0):  ' + '  '.join(f'{v:.3f}' for v in agree_term))
    for w in args.weights:
        if w == 0.0:
            continue
        am = np.nanmean(np.stack(agree_w[w], 0), 0)
        print(f'blended w={w:<4}:        ' + '  '.join(f'{v:.3f}' for v in am))

    if has_scipy and spearman_acc:
        sp = np.nanmean(np.stack(spearman_acc, 0), 0)
        print('\n=== spearman(step local contribution, terminal quality) across K ===')
        print('(low early-step value = terminal signal is a poor proxy for early steps)')
        print('step:      ' + '  '.join(f's{i+1}' for i in range(n_outer)))
        print('spearman:  ' + '  '.join(f'{v:.3f}' for v in sp))

    print('\nInterpretation:')
    print('- w=0 reproduces the report`s step-1 sign_agree ~0.75 problem.')
    print('- if blended w>0 raises early-step sign_agree toward 1.0, the per-step')
    print('  reward is delivering correct-direction signal to early steps.')


if __name__ == '__main__':
    main()
