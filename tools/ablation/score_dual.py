"""双口径打分：fg400（全部前景切片）与 valid183（排除零预测前景切片）。

用法:
    python tools/ablation/score_dual.py <slices.json> [<slices.json> ...]
    python tools/ablation/score_dual.py --table <name>=<slices.json> ...
"""
import json
import sys


def score(path):
    items = json.load(open(path))
    fg = [it for it in items if it.get('gt_foreground_pixels', 0) > 0]
    valid = [it for it in fg if it.get('n_pred', 0) > 0]

    def mean(xs, key):
        return sum(x[key] for x in xs) / len(xs) if xs else float('nan')

    return {
        'n_slices': len(items),
        'n_fg': len(fg),
        'n_valid': len(valid),
        'n_zero_pred': len(fg) - len(valid),
        'fg_dice': mean(fg, 'foreground_dice'),
        'fg_iou': mean(fg, 'foreground_iou'),
        'valid_dice': mean(valid, 'foreground_dice'),
        'valid_iou': mean(valid, 'foreground_iou'),
        'all_dice': mean(items, 'foreground_dice'),
    }


def main():
    args = sys.argv[1:]
    table = False
    if args and args[0] == '--table':
        table = True
        args = args[1:]
    if not args:
        print(__doc__)
        return 1

    rows = []
    for a in args:
        if '=' in a:
            name, path = a.split('=', 1)
        else:
            name, path = a, a
        rows.append((name, score(path)))

    if table:
        hdr = ('name', 'n_fg', 'n0', 'fg_dice', 'fg_iou', 'v_dice', 'v_iou')
        print('| %-28s | %4s | %3s | %7s | %7s | %7s | %7s |' % hdr)
        print('|%s|%s|%s|%s|%s|%s|%s|' % ('-' * 30, '-' * 6, '-' * 5,
                                          '-' * 9, '-' * 9, '-' * 9, '-' * 9))
        for name, s in rows:
            print('| %-28s | %4d | %3d | %7.4f | %7.4f | %7.4f | %7.4f |' % (
                name, s['n_fg'], s['n_zero_pred'], s['fg_dice'], s['fg_iou'],
                s['valid_dice'], s['valid_iou']))
    else:
        for name, s in rows:
            print(name)
            for k in ('n_slices', 'n_fg', 'n_valid', 'n_zero_pred',
                      'fg_dice', 'fg_iou', 'valid_dice', 'valid_iou', 'all_dice'):
                print('  %-12s %s' % (k, s[k]))
    return 0


if __name__ == '__main__':
    sys.exit(main())
