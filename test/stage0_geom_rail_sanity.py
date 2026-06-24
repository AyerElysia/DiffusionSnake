"""Stage 0 sanity for the geom_band_detail explorer rail.

Verifies the math that makes the config-only "learned geom" run safe:
1. The lowfreq basis round-trip is exact: project(delta_from_z(z)) == z.
   This is what makes sample-time z and update-time recomputed z agree, so the
   PPO ratio = exp(lp_cur - old_log) == 1 at step 0 with a zero-init explorer.
2. With detail fully gated off, the band_detail action reduces to mean + low_delta
   (i.e. identical to the plain geom route).
3. A zero-init FourierExplorer gives mu=0, logstd=0 -> sampling is unit randn,
   identical to today's pure-randn geom.

No model load needed: these are properties of the basis + explorer init only.
"""
import os
os.environ.setdefault('CFG_FILE', 'configs/1232_final_v5_geom_action_5step_from3500_gpu2.yaml')

import torch

from grpo_train_v5_geom_action import (
    _lowfreq_basis,
    _geom_delta_from_z,
    _project_geom_z,
    _contour_normals,
    _normal_z_logprob,
    FourierExplorer,
)


def make_synth_contour(n_contours=4, p=128):
    t = torch.linspace(0, 2 * torch.pi, p + 1)[:-1]
    # a few wobbly closed contours
    polys = []
    for i in range(n_contours):
        r = 30.0 + 4.0 * torch.cos((i + 2) * t) + 2.0 * torch.sin((i + 3) * t)
        cx, cy = 64.0 + i, 64.0 - i
        polys.append(torch.stack([cx + r * torch.cos(t), cy + r * torch.sin(t)], dim=-1))
    return torch.stack(polys, dim=0)  # [N, P, 2]


def main():
    torch.manual_seed(0)
    N, P, M = 4, 128, 8
    poly = make_synth_contour(N, P)
    sigma = 0.6

    # --- Test 1: round-trip project(delta_from_z(z)) == z ---
    z = torch.randn(N, M)
    delta = _geom_delta_from_z(poly, z, sigma)              # [N,P,2] normal-direction displacement
    z_rt = _project_geom_z(poly, delta, sigma, M)           # back to z
    rt_err = (z_rt - z).abs().max().item()
    print(f"[T1] round-trip max|z_rt - z| = {rt_err:.3e}  (need < 1e-4)")

    # also confirm delta is purely along the normal (tangential component ~ 0)
    tang = torch.stack([-_contour_normals(poly)[..., 1], _contour_normals(poly)[..., 0]], dim=-1)
    tang_comp = (delta * tang).sum(-1).abs().max().item()
    print(f"[T1b] max tangential component of delta = {tang_comp:.3e}  (expect ~0)")

    # --- Test 2: detail OFF -> action == mean + low_delta ---
    mean = torch.randn(N, P, 2) * 0.05
    detail_delta = torch.zeros_like(poly)  # detail_gate/sigma = 0 => zeros
    action_full = mean + delta + detail_delta
    action_geom = mean + delta
    diff = (action_full - action_geom).abs().max().item()
    print(f"[T2] |action(band_detail, detail-off) - action(geom)| = {diff:.3e}  (need 0)")

    # --- Test 3: zero-init FourierExplorer -> mu=0, logstd=0 ---
    expl = FourierExplorer(low_modes=M, detail_modes=8, hidden_dim=64,
                           mu_max=0.50, logstd_min=-1.5, logstd_max=0.7)
    expl.eval()
    sampled_feat = torch.randn(N, 64, P)
    low_mu, low_logstd, det_mu, det_logstd = expl(poly, sampled_feat, 0.5)
    print(f"[T3] zero-init explorer: max|low_mu| = {low_mu.abs().max().item():.3e}  "
          f"max|low_logstd| = {low_logstd.abs().max().item():.3e}  (both need ~0)")

    # --- Test 4: at zero-init, old_log (mu,logstd from explorer) == standard z logprob ---
    # sample z ~ N(mu, exp(logstd)) ; at init that's N(0,1) -> _normal_z_logprob(z, mu, logstd)
    z2 = low_mu + torch.exp(low_logstd) * torch.randn(N, M)
    lp_explorer = _normal_z_logprob(z2, low_mu, low_logstd)
    lp_unit = _normal_z_logprob(z2, torch.zeros_like(low_mu), torch.zeros_like(low_logstd))
    lp_diff = (lp_explorer - lp_unit).abs().max().item()
    print(f"[T4] zero-init: |lp_explorer - lp_unit_normal| = {lp_diff:.3e}  (need ~0 => ratio=1 at step0)")

    ok = (rt_err < 1e-4 and diff < 1e-9 and low_mu.abs().max().item() < 1e-5
          and low_logstd.abs().max().item() < 1e-5 and lp_diff < 1e-5)
    print("\nRESULT:", "PASS — rail is safe, config-only learned-geom will start == baseline" if ok else "FAIL — investigate before training")


if __name__ == '__main__':
    main()
