"""Full pairwise bit-exact loss matrix over the six fair-RNG arms.

Fair-RNG guarantees step i == same batch in every arm, so any loss difference is
purely architectural. Prediction under the dead-width hypothesis:
  B1,B2 ~= A0   (width dead -> same as L6/F256)
  C1    ~= A1   (width dead -> same as L8/F256)
  A0 != A1 != A2  (depth is real)
"""
import json
import os

ROOT = ('/home/medteam/Zhrch/DiffusionSnake-12-30-pure2d-scaleup-outputs-20260808/'
        'fair_rng_six_arm_matrix_v1_r1_20260808/training')

ARMS = [('A0', 'L6 /F256'), ('A1', 'L8 /F256'), ('A2', 'L10/F256'),
        ('B1', 'L6 /F384'), ('B2', 'L6 /F512'), ('C1', 'L8 /F384')]

loss = {}
for arm, _ in ARMS:
    d = {}
    with open(os.path.join(ROOT, arm, 'logs.jsonl')) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            if 'step' in r and 'loss' in r:
                d[int(r['step'])] = float(r['loss'])
    loss[arm] = d
    print('%s: %d steps' % (arm, len(d)))

names = [a for a, _ in ARMS]
print()
print('=== pairwise: n_bit_identical_steps / n_common , max|delta| ===')
print('%-5s %s' % ('', ' '.join('%-24s' % n for n in names)))
for a in names:
    row = []
    for b in names:
        if a == b:
            row.append('%-24s' % '-')
            continue
        common = sorted(set(loss[a]) & set(loss[b]))
        ident = sum(1 for s in common if loss[a][s] == loss[b][s])
        mx = max(abs(loss[a][s] - loss[b][s]) for s in common) if common else 0.0
        row.append('%-24s' % ('%4d/%4d  %.2e' % (ident, len(common), mx)))
    print('%-5s %s' % (a, ' '.join(row)))

print()
print('=== the two key comparisons ===')
for a, b, claim in [('B1', 'A0', 'width dead => B1 == A0'),
                    ('B2', 'A0', 'width dead => B2 == A0'),
                    ('C1', 'A1', 'width dead => C1 == A1'),
                    ('C1', 'A0', 'C1 vs A0 (should differ: depth)'),
                    ('A1', 'A0', 'depth real => A1 != A0'),
                    ('A2', 'A0', 'depth real => A2 != A0')]:
    common = sorted(set(loss[a]) & set(loss[b]))
    ident = sum(1 for s in common if loss[a][s] == loss[b][s])
    mx = max(abs(loss[a][s] - loss[b][s]) for s in common)
    mean_abs = sum(abs(loss[a][s] - loss[b][s]) for s in common) / len(common)
    ref = sum(loss[b][s] for s in common) / len(common)
    print('%-3s vs %-3s : identical %4d/%4d  max|d|=%.3e  mean|d|=%.3e  (%.4f%% of loss)  <- %s'
          % (a, b, ident, len(common), mx, mean_abs, 100.0 * mean_abs / ref, claim))
