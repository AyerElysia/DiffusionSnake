# Config Layout

Root `configs/` keeps only current training configs and key reproducible baselines.

## Current / Key Configs

- `btcv_diffusion_dit_v3_4_fm_full_noleak.yaml`: V3.4-FM baseline.
- `btcv_diffusion_dit_v4_6c_fm_mlp_shared_moe_gpu3.yaml`: strongest completed MLP shared-MoE baseline.
- `btcv_diffusion_dit_v4_7_fm_looped_shared_moe_gpu1.yaml`: completed looped latent reasoning baseline.
- `btcv_diffusion_dit_v4_9_fm_looped_mlp_shared_moe_richstate_gpu3.yaml`: completed V4.9 rich-state baseline.
- `btcv_diffusion_dit_v4_6d_no_p3_detail_long_from_v46c_gpu0.yaml`: active no-P3/detail long run from V4.6c.
- `btcv_diffusion_dit_v4_9c_fm_richstate_precision_gpu1.yaml`: active precision-state fine-tune from V4.9.
- `btcv_diffusion_dit_v4_10_fm_dit_ffn_moe_no_p3_detail_gpu2.yaml`: active DiT-FFN-MoE no-P3/detail run.
- `btcv_diffusion_dit_v4_10_full_fm_dit_ffn_moe_gpu3.yaml`: active full V4.9 + DiT-FFN-MoE run.

## Archive

Older configs are archived under `configs/archive/20260519/`:

- `ablations/`: V3.4/V4.1 ablation configs.
- `rl/`: GRPO/RL configs.
- `sam_swin_yolo/`: SAM, SAMSnake, Swin, and detect-only configs.
- `retired_diffusion/`: retired diffusion/FM experiment configs.
- `misc/`: heatmap and other non-mainline configs.
