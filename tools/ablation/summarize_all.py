"""Collect every finished ablation eval into one comparison table.

Scans data/outputs/{baseline_ep77_eval_gt400, abl_*/eval_gt400_ep*} for
slices.json, recomputes both metric conventions (see score_dual.py) and prints
one row per arm with the delta against the A0 ep7 reference.

Two conventions are reported because 26 of the 209 foreground val slices are
dropped to n_inst=0 by the GT polygon area filter: snake_voc_utils.py:233
`filter_tiny_polys` keeps only polygons with area > 5, and it runs in the
128x128 *output* space, where a real vertebra of 84-217 px2 in the 616x473
original shrinks to 0.3-4.8 px2. So the model never receives a GT box for them:
  fg400    - all 209 foreground slices, the 26 counted as dice 0 (hard ceiling)
  valid183 - only the 183 slices where a prediction was possible
The h_areafix arm lowers that floor to 0.5 via cfg.min_poly_area_output.
"""
import json
import os
import re
import sys

ROOT = '/home/medteam/Zhrch/DiffusionSnake-12-30'
OUT = os.path.join(ROOT, 'data/outputs')

# arm key -> (layers, upscale, gridfix, gt_area_floor_in_output_space)
ARM_SPEC = {
    'baseline_ep77': ('single', 2, 'legacy', 5.0),
    'a0_u2': ('single', 2, 'legacy', 5.0),
    'b_u2_dual': ('dual', 2, 'legacy', 5.0),
    'c_u4': ('single', 4, 'legacy', 5.0),
    'd_u4_dual': ('dual', 4, 'legacy', 5.0),
    'e_u4_gridfix': ('single', 4, 'half_pixel', 5.0),
    'f_u2_dual_gridfix': ('dual', 2, 'half_pixel', 5.0),
    'g_u2_dual_lnorm':   ('dual', 2, 'lnorm', 5.0),
    'h_areafix': ('single', 2, 'legacy', 0.5),
    'i_areafix_gridfix': ('single', 2, 'half_pixel', 0.5),
}

DIR_TO_ARM = {
    'abl_a0_u2_single': 'a0_u2',
    'abl_b_u2_dual': 'b_u2_dual',
    'abl_c_u4_single': 'c_u4',
    'abl_d_u4_dual': 'd_u4_dual',
    'abl_e_u4_single_gridfix': 'e_u4_gridfix',
    'abl_f_u2_dual_gridfix': 'f_u2_dual_gridfix',
    'abl_g_u2_dual_lnorm': 'g_u2_dual_lnorm',
    'abl_h_areafix': 'h_areafix',
    'abl_i_areafix_gridfix': 'i_areafix_gridfix',
}


def score(slices_path):
    """Return (fg_dice, fg_iou, v_dice, v_iou, n_fg, n_valid)."""
    with open(slices_path) as f:
        rows = json.load(f)
    if isinstance(rows, dict):
        rows = rows.get('slices', rows.get('results', []))
    fg_d, fg_i, v_d, v_i = [], [], [], []
    for r in rows:
        if r.get('gt_foreground_pixels', 0) <= 0:
            continue                      # background-only slice, excluded
        d = float(r.get('foreground_dice', 0.0))
        i = float(r.get('foreground_iou', 0.0))
        fg_d.append(d)
        fg_i.append(i)
        if r.get('n_pred', 0) > 0:        # a prediction was actually possible
            v_d.append(d)
            v_i.append(i)
    m = lambda xs: sum(xs) / len(xs) if xs else float('nan')
    return m(fg_d), m(fg_i), m(v_d), m(v_i), len(fg_d), len(v_d)


def collect():
    found = []
    p = os.path.join(OUT, 'baseline_ep77_eval_gt400', 'slices.json')
    if os.path.isfile(p):
        found.append(('baseline_ep77', 77, p))
    for d, arm in sorted(DIR_TO_ARM.items()):
        base = os.path.join(OUT, d)
        if not os.path.isdir(base):
            continue
        for sub in sorted(os.listdir(base)):
            mo = re.match(r'eval_gt400_ep(\d+)$', sub)
            if not mo:
                continue
            p = os.path.join(base, sub, 'slices.json')
            if os.path.isfile(p):
                found.append((arm, int(mo.group(1)), p))
    return found


def main():
    rows = []
    for arm, ep, path in collect():
        try:
            fd, fi, vd, vi, nfg, nv = score(path)
        except Exception as exc:                       # keep going on bad files
            print('  [skip] %s ep%s: %s' % (arm, ep, exc))
            continue
        spec = ARM_SPEC.get(arm, ('?', '?', '?', '?'))
        layers, up, grid, area = spec
        rows.append(dict(arm=arm, ep=ep, layers=layers, up=up, grid=grid,
                         area=area,
                         fd=fd, fi=fi, vd=vd, vi=vi, nfg=nfg, nv=nv))

    ref = next((r for r in rows if r['arm'] == 'a0_u2' and r['ep'] == 7), None)
    ref_fd = ref['fd'] if ref else None

    hdr = ('%-20s %4s %-6s %2s %-10s %5s %8s %8s %8s %8s %9s'
           % ('arm', 'ep', 'layers', 'up', 'gridfix', 'area',
              'fg_dice', 'fg_iou', 'v_dice', 'v_iou', 'd_vs_A0'))
    print(hdr)
    print('-' * len(hdr))
    rows.sort(key=lambda r: (r['arm'], r['ep']))
    for r in rows:
        delta = ('%+.4f' % (r['fd'] - ref_fd)) if ref_fd is not None else '  n/a'
        print('%-20s %4d %-6s %2s %-10s %5s %8.4f %8.4f %8.4f %8.4f %9s'
              % (r['arm'], r['ep'], r['layers'], r['up'], r['grid'], r['area'],
                 r['fd'], r['fi'], r['vd'], r['vi'], delta))
    if rows:
        print('\nslice counts (first row): fg=%d n_pred>0=%d'
              % (rows[0]['nfg'], rows[0]['nv']))
        print('note: v_* excludes slices with n_pred==0; for area=5.0 arms those'
              ' are the 26 slices whose GT instances are all dropped by'
              ' filter_tiny_polys.')
    print('\n%d eval(s) collected' % len(rows))


if __name__ == '__main__':
    sys.exit(main())
