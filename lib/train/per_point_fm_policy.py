"""Shared per-point flow-matching action policies."""

import torch
import torch.nn as nn


class PerPointFMScalePolicy(nn.Module):
    """
    Per-point multiplicative FM scale policy — tanh-bounded parameterisation.

    Sampling:
        raw_pp ~ N(mu_pp, exp(logstd_pp)^2)          # unbounded Gaussian in raw space
        scale_pp = tanh(raw_pp) * max_scale            # bounded to (-max_scale, +max_scale)
        action   = fm_velocity * (1 + scale_pp)        # multiplier in (1-A, 1+A)

    Design rationale:
      • scale=0 → pure FM prediction (correct initialisation: last-layer bias=0)
      • multiplier always in (1-A, 1+A), e.g. (0.75, 1.25) with A=0.25
      • no reversal possible; exploration is a fine-tuning of per-point amplitude
      • tanh reparameterisation yields unbiased log_prob (no truncated-Gaussian bias)
      • stored key 'fm_velocity_scales' now holds raw_pp (pre-tanh) for PPO reuse

    forward() returns (mu_pp: B×N, logstd_pp: B×1) for the raw (pre-tanh) Gaussian.
    """

    def __init__(
        self,
        outer_steps: int,
        feature_dim: int = 128,
        feature_embed_dim: int = 32,
        hidden_dim: int = 64,
        init_logstd: float = -0.5,
        logstd_min: float = -3.0,
        logstd_max: float = 2.0,
        offset_scale: float = 0.5,
        max_scale: float = 0.25,
        zero_mean_local: bool = False,
    ):
        super().__init__()
        self.outer_steps = int(outer_steps)
        self.init_logstd = float(init_logstd)
        self.logstd_min = float(logstd_min)
        self.logstd_max = float(logstd_max)
        self.offset_scale = float(offset_scale)
        self.max_scale = float(max_scale)  # tanh bound: scale_pp ∈ (-A,+A), multiplier ∈ (1-A,1+A)
        self.zero_mean_local = bool(zero_mean_local)

        self.step_embed = nn.Embedding(
            max(int(outer_steps), 1), max(int(feature_embed_dim), 4)
        )
        # Global branch: pool(sampled_feat) + step_emb → (mu_g, raw_logstd_g)
        global_in = max(int(feature_dim), 1) + max(int(feature_embed_dim), 4)
        self.global_net = nn.Sequential(
            nn.Linear(global_in, max(int(hidden_dim), 16)),
            nn.ReLU(),
            nn.Linear(max(int(hidden_dim), 16), 2),  # mu_g, raw_logstd_g
        )
        nn.init.zeros_(self.global_net[-1].weight)
        nn.init.constant_(self.global_net[-1].bias, 0.0)

        # Per-point branch: per-point feat + step_emb + vel_mag_norm → mu_offset
        point_in = max(int(feature_dim), 1) + max(int(feature_embed_dim), 4) + 1
        self.point_net = nn.Sequential(
            nn.Linear(point_in, max(int(hidden_dim), 16)),
            nn.ReLU(),
            nn.Linear(max(int(hidden_dim), 16), 1),  # raw mu_offset per point
        )
        nn.init.zeros_(self.point_net[-1].weight)
        nn.init.zeros_(self.point_net[-1].bias)

    def forward(
        self,
        step_idx: int,
        poly: torch.Tensor,
        c_poly: torch.Tensor,
        mean_action: torch.Tensor,  # FM velocity (B, N_poly, 2)
        sampled_feat: torch.Tensor,  # (B, N_samp, C) or (B, C, N_samp)
        frac: float,
    ):
        """Returns (mu_pp: B×N, logstd_pp: B×1 broadcast-ready)."""
        B = poly.size(0)
        N_poly = mean_action.size(1)

        # Normalise sampled_feat to (B, N_samp, C). Use N_poly (known ground truth)
        # to pick the point axis — channel count (64) can be smaller OR larger than
        # N_poly (128) depending on config, so a size-comparison heuristic is unsafe.
        if sampled_feat.dim() == 3 and sampled_feat.size(1) != N_poly and sampled_feat.size(2) == N_poly:
            sampled_feat = sampled_feat.transpose(1, 2)
        sf = sampled_feat.to(device=poly.device, dtype=poly.dtype)

        idx = max(0, min(int(step_idx), self.step_embed.num_embeddings - 1))
        step_t = torch.full((B,), idx, device=poly.device, dtype=torch.long)
        s_emb = self.step_embed(step_t).to(dtype=poly.dtype)  # (B, E)

        # Global branch
        global_feat = sf.mean(dim=1)  # (B, C)
        g_in = torch.cat([global_feat, s_emb], dim=-1)
        g_out = self.global_net(g_in)  # (B, 2)
        mu_g = g_out[:, :1]            # (B, 1)
        raw_logstd = g_out[:, 1:2]     # (B, 1)
        logstd_pp = (raw_logstd + self.init_logstd).clamp(self.logstd_min, self.logstd_max)  # (B,1)

        # FM velocity magnitude feature
        vel_mag = mean_action.norm(dim=-1, keepdim=True)       # (B, N_poly, 1)
        vel_mag_norm = vel_mag / (vel_mag.mean(dim=1, keepdim=True).clamp_min(1e-6))

        # Per-point mu offset -- use the UN-pooled per-point feature (sf) so the
        # point_net actually sees each point's local image content, instead of
        # a copy-pasted global-pooled feature (which erased all spatial signal).
        # sf is sampled at the N_poly contour-point locations (get_gcn_feature
        # on i_it_py), so N_samp == N_poly here and no re-alignment is needed.
        s_emb_exp = s_emb.unsqueeze(1).expand(-1, N_poly, -1)      # (B, N_poly, E)
        p_in = torch.cat([sf, s_emb_exp, vel_mag_norm], dim=-1)
        mu_offset_raw = self.point_net(p_in).squeeze(-1)  # (B, N_poly)

        local_offset = torch.tanh(mu_offset_raw) * self.offset_scale
        if self.zero_mean_local:
            local_offset = local_offset - local_offset.mean(dim=1, keepdim=True).detach()
            mu_pp = local_offset
        else:
            mu_pp = mu_g + local_offset
        return mu_pp, logstd_pp  # (B, N), (B, 1)
