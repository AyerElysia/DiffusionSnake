"""Credit-assignment diagnostic analysis (offline, no training)."""
import argparse, glob, json, math, os, sys
import numpy as np

def spearman(xs, ys):
    if len(xs) < 3: return float('nan')
    rx = np.argsort(np.argsort(xs)).astype(float); ry = np.argsort(np.argsort(ys)).astype(float)
    rx -= rx.mean(); ry -= ry.mean()
    den = math.sqrt((rx**2).sum() * (ry**2).sum())
    return float((rx*ry).sum()/den) if den > 0 else float('nan')

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pattern', default='data/outputs/1232_final_v5_geom8_baseline_bs6_gpu2/credit_diag_step*.json')
    ap.add_argument('--out', default='data/analysis/credit_diag/summary.json')
    a = ap.parse_args()

    files = sorted(glob.glob(a.pattern), key=lambda p: int(''.join(c for c in os.path.basename(p) if c.isdigit())))
    if not files: print('no diag files at', a.pattern); sys.exit(2)
    print(f'[analyze_credit] {len(files)} files')

    # accumulators (per-step arrays averaged over files)
    n_steps = 0; K = 0
    files_data = []
    for f in files:
        d = json.load(open(f))
        n_steps = int(d['outer_steps']); K = int(d['k_rollouts'])
        ros = d['rollouts']
        # rollout x step partials / det / deltas / sign-vs-terminal-per-rollout
        sp  = np.array([ro['partials'] for ro in ros])             # K x S
        dms = np.array([ro['det_partials'] for ro in ros])         # K x S (same across rollouts)
        dp  = sp - dms
        ds  = np.sign(dp)
        term_sign = np.sign(sp[:, -1] - dms[:, -1])                # K
        # treat det partial for the steel rails (same in each rollout)
        det_parts = np.array(ros[0]['det_partials'])               # S
        # per-step stats aggregated across this batch (K rollouts)
        partial_better_per_step = (sp >= det_parts[None, :]).mean(axis=0)  # S
        sign_agree_per_step = np.array([
            (ds[:, t] == term_sign).mean() for t in range(n_steps)
        ]) if n_steps else np.array([])
        std_per_step = sp.std(axis=0)                            # S
        actions_norm_per_step = np.array([ro['actions_norm_px'] for ro in ros]).mean(axis=0)
        # terminal > 0 count and gate terms
        tr = d['terminal_recompute']
        # correlation of step-partial with terminal across K rollouts
        corr_per_step = []
        for t in range(n_steps):
            if sp[:, t].std() > 1e-9 and sp[:, -1].std() > 1e-9:
                corr_per_step.append(spearman(sp[:, t].tolist(), sp[:, -1].tolist()))
            else:
                corr_per_step.append(float('nan'))
        files_data.append({
            'init': d['init_score'], 'det_final': d['det_final_score'],
            'det_parts': det_parts, 'sp': sp, 'dp': dp, 'ds': ds,
            'term_sign': term_sign,
            'partial_better_per_step': partial_better_per_step,
            'sign_agree_per_step': sign_agree_per_step,
            'std_per_step': std_per_step,
            'actions_norm_per_step': actions_norm_per_step,
            'corr_per_step': corr_per_step,
            'terminal_quality_mean': tr['quality_mean'],
            'gate_active_frac': tr['gate_active_frac'],
            'terminal_adv_abs': tr['adv_after_gate_abs_mean'],
            'terminal_q_per_rollout': tr['all_krollouts_quality_per_contour_mean'],
            'terminal_std': tr['quality_std_mean'],
        })

    # aggregate per-step over files (stack)
    stack_partial_better = np.array([fd['partial_better_per_step'] for fd in files_data])    # F x S
    stack_sign_agree     = np.array([fd['sign_agree_per_step']     for fd in files_data])
    stack_std_per_step   = np.array([fd['std_per_step']            for fd in files_data])
    stack_actions_norm   = np.array([fd['actions_norm_per_step']   for fd in files_data])
    stack_corr           = np.array([fd['corr_per_step']           for fd in files_data])
    stack_sp             = np.array([fd['sp'].mean(axis=0)         for fd in files_data])     # F x S
    stack_dp             = np.array([fd['dp'].mean(axis=0)         for fd in files_data])     # F x S
    stack_ds             = np.array([fd['ds'].mean(axis=0)         for fd in files_data])     # F x S
    stack_det_parts      = np.array([fd['det_parts']                for fd in files_data])     # F x S
    stack_terminal_q     = np.array([fd['terminal_quality_mean']    for fd in files_data])
    stack_gate           = np.array([fd['gate_active_frac']         for fd in files_data])
    stack_term_adv_abs   = np.array([fd['terminal_adv_abs']        for fd in files_data])
    stack_term_std       = np.array([fd['terminal_std']            for fd in files_data])

    # per-step dispersion / info content as ratio of std across K
    info_ratio = stack_std_per_step.mean(axis=0) / max(stack_term_std.mean(), 1e-9)

    out = {
        'n_files': len(files),
        'outer_steps': n_steps,
        'k_rollouts': K,
        'fractions': json.load(open(files[0]))['fractions'],
        'geom_sigma_px': json.load(open(files[0]))['geom_sigma_px'],
        'terminal_quality_mean': float(stack_terminal_q.mean()),
        'terminal_quality_std_mean': float(stack_term_std.mean()),
        'terminal_adv_after_gate_abs_mean': float(stack_term_adv_abs.mean()),
        'gate_active_frac_mean': float(stack_gate.mean()),
        'gate_examples_below_50pct': float((stack_gate < 0.5).mean()),
        'per_step_delta_vs_det_mean': stack_dp.mean(axis=0).tolist(),
        'per_step_sign_terminal_agree_per_step': stack_sign_agree.mean(axis=0).tolist(),
        'per_step_frac_rollouts_better_than_det': stack_partial_better.mean(axis=0).tolist(),
        'per_step_frac_sign_agree_per_step_mean': stack_sign_agree.mean(axis=0).tolist(),
        'per_step_corr_partial_vs_terminal_mean': ({
            'spearman_per_step_mean': np.nanmean(stack_corr, axis=0).tolist() if len(files) else [],
        }),
        'per_step_std_K_dispersion_mean': stack_std_per_step.mean(axis=0).tolist(),
        'terminal_std_mean': float(stack_term_std.mean()),
        'info_ratio_per_step (std_per_step / terminal_std)': info_ratio.tolist(),
        'mean_actions_norm_px_per_step': stack_actions_norm.mean(axis=0).tolist(),
        'partial_score_per_step_mean': stack_sp.mean(axis=0).tolist(),
        'det_partial_score_per_step_mean': stack_det_parts.mean(axis=0).tolist(),
        'per_step_sign_delta_mean': stack_ds.mean(axis=0).tolist(),
    }
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, 'w') as fh:
        json.dump(out, fh, indent=2)
    print(json.dumps(out, indent=2))

if __name__ == '__main__':
    main()