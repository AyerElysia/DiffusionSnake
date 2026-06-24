"""Stage 0b sanity for image-conditioned FourierExplorer.

Checks:
1. Zero-init explorer remains identity even with sampled_feat input.
2. Low-frequency delta/project round-trip works for larger mode counts.
3. Sample-time stored sampled_feat gives the same update-time z/logprob.
"""
import os
os.environ.setdefault('CFG_FILE', 'configs/1232_final_v5_geom_learned_probe_gpu0.yaml')

import torch

from grpo_train_v5_geom_action import (
    FourierExplorer,
    _band_detail_delta_from_z,
    _normal_z_logprob,
    _geom_delta_from_z,
    _project_band_detail_z,
    _project_geom_z,
)


def make_synth_contour(n_contours=4, p=128):
    t = torch.linspace(0, 2 * torch.pi, p + 1)[:-1]
    polys = []
    for i in range(n_contours):
        r = 30.0 + 4.0 * torch.cos((i + 2) * t) + 2.0 * torch.sin((i + 3) * t)
        cx, cy = 64.0 + i, 64.0 - i
        polys.append(torch.stack([cx + r * torch.cos(t), cy + r * torch.sin(t)], dim=-1))
    return torch.stack(polys, dim=0)


def main():
    torch.manual_seed(7)
    n, p, low_modes = 4, 128, 16
    detail_k_min, detail_k_max = 9, 16
    detail_modes = 2 * (detail_k_max - detail_k_min + 1)
    low_sigma = 0.35
    detail_sigma = 0.20
    detail_gate = 0.55
    detail_weight = 0.35

    state = make_synth_contour(n, p)
    sampled_feat = torch.randn(n, 64, p)
    mean = torch.randn(n, p, 2) * 0.01
    expl = FourierExplorer(
        low_modes=low_modes,
        detail_modes=detail_modes,
        hidden_dim=64,
        mu_max=0.50,
        logstd_min=-1.5,
        logstd_max=0.7,
        feature_dim=64,
        feature_embed_dim=32,
    )
    expl.eval()

    low_mu, low_logstd, detail_mu, detail_logstd = expl(state, sampled_feat, 0.5)
    zero_err = max(
        low_mu.abs().max().item(),
        low_logstd.abs().max().item(),
        detail_mu.abs().max().item(),
        detail_logstd.abs().max().item(),
    )
    print(f"[T1] zero-init image explorer max output abs = {zero_err:.3e}  (need < 1e-5)")

    z_low = low_mu + torch.exp(low_logstd) * torch.randn(n, low_modes)
    low_delta = _geom_delta_from_z(state, z_low, low_sigma)
    z_low_rt = _project_geom_z(state, low_delta, low_sigma, low_modes)
    low_rt_err = (z_low_rt - z_low).abs().max().item()
    print(f"[T2] low-mode round-trip max|z_rt - z| = {low_rt_err:.3e}  (need < 1e-4)")

    z_damped = torch.randn(n, low_modes)
    damped_delta = _geom_delta_from_z(state, z_damped, low_sigma, damp_highfreq=True)
    z_damped_rt = _project_geom_z(
        state,
        damped_delta,
        low_sigma,
        low_modes,
        damp_highfreq=True,
    )
    damped_rt_err = (z_damped_rt - z_damped).abs().max().item()
    print(f"[T2b] damped low-mode round-trip max|z_rt - z| = {damped_rt_err:.3e}  (need < 1e-4)")

    z_detail = detail_mu + torch.exp(detail_logstd) * torch.randn(n, detail_modes)
    detail_delta = _band_detail_delta_from_z(
        state,
        z_detail,
        detail_sigma,
        detail_k_min,
        detail_k_max,
        detail_gate,
    )
    action = mean + low_delta + detail_delta
    residual = action.detach() - mean

    old_log = _normal_z_logprob(z_low, low_mu, low_logstd) + detail_weight * _normal_z_logprob(
        z_detail,
        detail_mu,
        detail_logstd,
    )

    # PPO update path must use the same sampled_feat stored at sample time.
    low_mu_u, low_logstd_u, detail_mu_u, detail_logstd_u = expl(state, sampled_feat, 0.5)
    z_low_u = _project_geom_z(state, residual, low_sigma, low_modes)
    z_detail_u = _project_band_detail_z(
        state,
        residual,
        detail_sigma,
        detail_k_min,
        detail_k_max,
        detail_gate,
    )
    lp_cur = _normal_z_logprob(z_low_u, low_mu_u, low_logstd_u) + detail_weight * _normal_z_logprob(
        z_detail_u,
        detail_mu_u,
        detail_logstd_u,
    )
    z_err = max((z_low_u - z_low).abs().max().item(), (z_detail_u - z_detail).abs().max().item())
    lp_err = (lp_cur - old_log).abs().max().item()
    print(f"[T3] sample/update z max error = {z_err:.3e}  (need < 1e-5)")
    print(f"[T4] sample/update logprob max error = {lp_err:.3e}  (need < 1e-5)")

    ok = (
        zero_err < 1e-5
        and low_rt_err < 1e-4
        and damped_rt_err < 1e-4
        and z_err < 1e-5
        and lp_err < 1e-5
    )
    print("\nRESULT:", "PASS" if ok else "FAIL")
    raise SystemExit(0 if ok else 1)


if __name__ == '__main__':
    main()
