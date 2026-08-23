from .yacs import CfgNode as CN
import argparse
import os
import sys
from pathlib import Path

# Modified by Zhang Ruicheng on 2024.04.21

cfg = CN()

cfg.train_split_rate = 0.8
cfg.multistage = False
cfg.random_num = 0

# 中间结果可视化
cfg.vis_zrc = 0

# model
cfg.model = 'ZRC'
cfg.model_dir = 'data/model'

# network
cfg.network = 'dla_34'

cfg.inital_drn22 = '/data/lyc/EnergeSnake/data/model/initial_darnet/drn_d_22-4bd2f8ea.pth'

# network heads
cfg.heads = CN()

# task
cfg.task = ''

# gpus
cfg.gpus = [0]

# if load the pretrained network
cfg.resume = True
cfg.resume_weights_only = False
cfg.resume_path = ''
cfg.resume_exclude_prefixes = []

# Diffusion/Snake integration switches
cfg.use_diffusion_evolution = False
cfg.use_diffusion_trainer = False
cfg.freeze_snake = False
cfg.skip_diffusion_forward = False
cfg.detector_only_warmup = False
cfg.detector_backend = 'yolo'
cfg.heatmap_backbone = 'resnet18'
cfg.heatmap_pretrained = False
cfg.freeze_heatmap_detector = False
cfg.skip_heatmap_detector_when_gt = False
cfg.heatmap_wh_weight = 0.1
cfg.heatmap_class_offset = 0
cfg.gt_detection_class_offset = 0
cfg.convnext_model_name = 'convnextv2_tiny'
cfg.convnext_pretrained = False
cfg.convnext_pretrained_path = ''
cfg.convnext_allow_fallback = True
cfg.convnext_fallback_model_name = 'convnext_tiny'
cfg.convnext_out_channels = 64
cfg.convnext_head_conv = 256
cfg.moonvit_root = 'Eagle/Embodied'
cfg.moonvit_pretrained_path = ''
cfg.moonvit_freeze = False
cfg.moonvit_input_norm = 'none'
cfg.moonvit_patch_size = 14
cfg.moonvit_num_layers = 6
cfg.moonvit_num_heads = 6
cfg.moonvit_hidden_size = 384
cfg.moonvit_intermediate_size = 1536
cfg.moonvit_pos_h = 64
cfg.moonvit_pos_w = 64
cfg.moonvit_out_channels = 64
cfg.moonvit_head_conv = 256
cfg.moonvit_target_stride = 4
cfg.moonvit_max_input_size = 0
cfg.eagle_teacher_json = ''
cfg.use_eagle_teacher_init = False
cfg.eagle_teacher_loss_weight = 0.0
cfg.eagle_teacher_conf_thresh = 0.0
cfg.locate_feat_inject = False
cfg.locate_feat_replace = False
cfg.locate_feat_cache_root = 'data/locate_feat_cache'
cfg.locate_feat_cache_dir = ''
cfg.locate_feat_keys = ['feat']
cfg.locate_feat_dim = 2304
# Optional offline LocateAnything detection source. The default preserves the
# inherited detector path byte-for-byte; cached detections are evaluation-only.
cfg.box_source = 'detector'
cfg.locany_cache_path = ''
cfg.locany_min_score = 1e-4
cfg.locany_min_box_side = 1.0
cfg.locany_min_box_area = 4.0
cfg.locany_nms_iou = 0.5
cfg.locany_max_detections = 32
cfg.locany_class_aware_nms = True
cfg.locany_missing = 'error'
# Upsampling factor inside LocateFeatReplacer (PixelShuffle). MoonViT grids are
# coarse (patch 14 -> ~10.6 original pixels per cell), so 2 leaves neighbouring
# contour points inside a single feature cell. 4 halves that effective stride.
cfg.locate_feat_replace_upscale = 2
# Number of stacked MoonViT layers fed to LocateFeatReplacer. When > 1 the
# cached layers are concatenated along channels, and their raw scales differ a
# lot (layer_18 std ~0.93 vs layer_26 std ~2.64), so layer_26 dominates the
# 1x1 input conv. Setting this to the layer count enables a parameter-free
# per-layer GroupNorm(affine=False) in front of the projection, which equalises
# the two layers without changing the state_dict.
cfg.locate_feat_input_layers = 1
# Per-point contour sampling in snake_gcn_utils.get_gcn_feature().
# 'legacy' keeps the original ResNet/YOLOv8-era formula (2x/w - 1), which sits
# half a pixel off the pixel-center convention; 'half_pixel' corrects it and
# agrees with the align_corners=True grid used by locate-feature replacement.
# 'border' padding stops contour points outside the map from sampling zeros.
cfg.gcn_sample_mode = 'legacy'
cfg.gcn_sample_padding_mode = 'zeros'
# Minimum GT polygon area, measured in the 1/4-resolution output space, below
# which an instance is dropped from supervision (snake_voc_utils.filter_tiny
# _polys). The inherited VOC value of 5.0 was tuned for large natural objects;
# sagittal vertebrae of 84-217 px^2 in a 616x473 slice shrink to 0.3-4.8 px^2
# after the 512-input / 128-output transform and are silently discarded, which
# is why 26 foreground val slices predict nothing at all.
cfg.min_poly_area_output = 5.0
cfg.sagittal_moonvit_cache_root = ''
cfg.sagittal_moonvit_feature_key = 'layer_18'
cfg.sagittal_moonvit_fusion_mode = 'center_neighbor_mean'
cfg.sagittal_moonvit_expected_input_size = 448
cfg.sagittal_moonvit_expected_patch_size = 14
cfg.sagittal_moonvit_expected_normalization = 'moonvit_pretrained_rgb_mean_std'
cfg.sagittal_moonvit_expected_checkpoint = ''
cfg.sagittal_train_exclude_case_ids = []
cfg.sagittal_eval_include_case_ids = []
cfg.sagittal_forbidden_case_ids = []
cfg.sagittal_expected_train_case_count = 0
cfg.sagittal_expected_train_row_count = 0
cfg.sagittal_expected_eval_case_count = 0
cfg.sagittal_expected_eval_row_count = 0
cfg.use_swin_snake_feature = False
cfg.swin_model_name = 'swin_tiny_patch4_window7_224'
cfg.swin_img_size = 672
cfg.swin_pretrained = False
cfg.swin_pretrained_path = ''
cfg.swin_freeze = False
cfg.diffusion_timesteps = 1000
cfg.use_ddim_inference = True
cfg.diffusion_loss_weight = 1.0
cfg.diffusion_disp_norm = False
cfg.diffusion_disp_stats = ''
cfg.use_dit_denoiser = False
cfg.use_dit_v3 = False
cfg.use_dit_v3_1 = False
cfg.use_dit_v3_2 = False
cfg.use_dit_v3_3 = False
cfg.use_dit_v3_4 = False
cfg.use_dit_v4 = False
cfg.use_dit_v4_1 = False
cfg.use_dit_v4_2 = False
cfg.use_dit_v3_6 = False
cfg.use_dit_v3_7 = False
cfg.use_dit_v3_8 = False
cfg.circular_conv_kernel = 5
cfg.use_flow_matching = False
cfg.flow_2d_s_conditioning = False
cfg.use_curve_inference = False
cfg.curve_alpha = 2.0
cfg.curve_steps = 20
cfg.curve_s_max = 0.97
cfg.curve_resample_feat = True
cfg.flow_ode_steps = 10
cfg.flow_train_noise_scale = 1.0
cfg.flow_use_disp_gate = False
cfg.flow_disp_gate_apply_inference = True
cfg.flow_disp_gate_apply_training_pred = False
cfg.flow_disp_gate_loss_weight = 0.0
cfg.flow_disp_gate_hidden_dim = 128
cfg.flow_disp_gate_init_bias = 4.0
cfg.flow_disp_gate_detach_input = True
cfg.dit_num_layers = 6
cfg.dit_num_heads = 8
cfg.dit_state_dim = 256

# GRPO / reward post-training defaults
cfg.use_grpo = False
cfg.use_grpo_kl = False
cfg.grpo_first_contour_only = False
cfg.grpo_train_steps = 1000
cfg.grpo_seed = 20260504
cfg.grpo_steps = 20
cfg.grpo_k = 4
cfg.grpo_window_size = 0
cfg.grpo_window_range = [0, 19]
cfg.grpo_clip_range = 0.2
cfg.grpo_adv_clip_max = 5.0
cfg.grpo_loss_weight = 1.0
cfg.grpo_action_std = 0.75
cfg.grpo_normalize_adv = True
cfg.grpo_reward_image_scale = False
cfg.grpo_reward_delta = False
cfg.grpo_adv_center = 'group'
cfg.grpo_pure_rl_loss = False
cfg.reward_w_region = 1.0
cfg.reward_w_dice = 0.0
cfg.reward_w_iou = 0.0

# Contour point configuration (default 128, can be overridden per version)
cfg.poly_num = 128
cfg.gt_poly_num = 128
cfg.adaptive_points_enabled = False
cfg.min_points = 32
cfg.max_points = 512
cfg.target_density = 2.5
cfg.round_to_multiple = 8
cfg.point_strategy = 'perimeter'
cfg.adaptive_use_area_threshold = False
cfg.adaptive_area_threshold = 4096.0
cfg.adaptive_small_points = 64
cfg.adaptive_large_points = 128

# V3.4: Multi-step iterative refinement
cfg.use_iterative_refinement = False
cfg.iterative_num_steps = 3
cfg.iterative_fractions = [0.3333, 0.5, 1.0]
cfg.iterative_ddim_steps = 20
cfg.iterative_ode_steps = 0

# V3.4-FM detail experiment knobs
cfg.v3_4_use_p3_features = False
cfg.v3_4_use_detail_context = False
cfg.v3_4_detail_context_mode = 'normal'
cfg.v3_7_use_p3_features = False
cfg.v3_7_use_detail_context = False
cfg.v3_7_use_detail_curve_context = False
cfg.v3_7_detail_context_mode = 'normal'
cfg.v4_use_p3_features = False
cfg.v4_use_detail_context = False
cfg.v4_detail_context_mode = 'normal_tangent'
cfg.v4_use_per_point_delta = True
cfg.v4_per_point_delta_scale = 0.25
cfg.v4_per_point_delta_reg_weight = 0.0
cfg.v4_1_use_p3_features = False
cfg.v4_1_use_detail_context = False
cfg.v4_1_detail_context_mode = 'normal'
cfg.v4_1_final_head_type = 'standard'
cfg.v4_1_use_per_point_delta = True
cfg.v4_1_per_point_delta_scale = 0.10
cfg.v4_1_per_point_delta_reg_weight = 0.0
# V4.6 output-head MoE balancing. Legacy remains the compatibility default;
# hard_phi converts measured hard Top-K congestion into a differentiable price.
cfg.v4_6_moe_balance_mode = 'legacy'
cfg.v4_6_moe_hard_phi_ema_decay = 0.99
cfg.v4_1_use_curvature_reweight = False
cfg.v4_1_curvature_loss_weight = 1.5
cfg.v4_1_curvature_reweight_power = 1.0
cfg.v4_1_small_disp_prob = 0.0
cfg.v4_1_small_disp_min_frac = 0.80
cfg.v4_1_small_disp_max_frac = 0.95
cfg.v4_2_use_p3_features = False
cfg.v4_2_use_detail_context = False
cfg.v4_2_detail_context_mode = 'normal_band'
cfg.v4_2_use_per_point_delta = True
cfg.v4_2_per_point_delta_scale = 0.10
cfg.v4_2_per_point_delta_reg_weight = 0.0
cfg.v4_2_use_curvature_conditioning = True
cfg.v4_2_curvature_embed_scale = 0.10
cfg.v4_2_use_delta_gate = True
cfg.v4_2_delta_gate_bias = -2.0
cfg.v4_2_use_curvature_reweight = True
cfg.v4_2_curvature_loss_weight = 1.5
cfg.v4_2_curvature_reweight_power = 1.0
cfg.v4_2_small_disp_prob = 0.10
cfg.v4_2_small_disp_min_frac = 0.80
cfg.v4_2_small_disp_max_frac = 0.95

# V4.9: richer interpolation-state sampling for multi-step flow training.
cfg.v4_9_use_rich_state_sampling = False
cfg.v4_9_continuous_state_prob = 0.60
cfg.v4_9_small_state_prob = 0.25
cfg.v4_9_hard_far_state_prob = 0.10
cfg.v4_9_near_zero_state_prob = 0.05
cfg.v4_9_continuous_min_frac = 0.05
cfg.v4_9_continuous_max_frac = 0.85
cfg.v4_9_hard_far_min_frac = 0.0
cfg.v4_9_hard_far_max_frac = 0.20
cfg.v4_9_near_zero_min_frac = 0.95
cfg.v4_9_near_zero_max_frac = 0.995

# V4.10: routed experts inside each DiT FFN branch.
cfg.v4_10_use_dit_ffn_moe = False
cfg.v4_10_dit_ffn_moe_num_experts = 4
cfg.v4_10_dit_ffn_moe_top_k = 2
cfg.v4_10_dit_ffn_moe_hidden_dim = 256
cfg.v4_10_dit_ffn_moe_balance_weight = 1e-3
cfg.v4_10_dit_ffn_moe_router_noise_std = 0.01
cfg.v4_10_dit_ffn_moe_expert_init_std = 1e-4
cfg.v4_10_dit_ffn_moe_routed_scale = 1.0
cfg.v4_10_dit_ffn_moe_use_point_embed = True
cfg.v4_10_dit_ffn_moe_use_cyclic_router = True

# V5.1 research: prototype routing + population-level phi balancing.
cfg.v5_1_use_prototype_phi_moe = False
cfg.v5_1_prototype_phi_moe_layers = 'odd'
cfg.v5_1_prototype_phi_num_experts = 4
cfg.v5_1_prototype_phi_top_k = 1
cfg.v5_1_prototype_phi_hidden_dim = 0
cfg.v5_1_prototype_phi_router_temperature = 0.20
cfg.v5_1_prototype_phi_balance_weight = 1e-3
cfg.v5_1_prototype_phi_ema_decay = 0.99
cfg.v5_1_prototype_phi_contrastive_weight = 1e-3
cfg.v5_1_prototype_phi_use_shared_expert = False
cfg.v5_1_prototype_phi_shared_hidden_dim = 0

# V5.2 research: causal output-head comparison.  The dense residual width of
# 1024 matches the total MLP parameters of four hidden-256 routed experts;
# modern_moe uses contour-level prototype routing and true sparse dispatch.
cfg.v5_2_output_dense_residual_hidden_dim = 1024
cfg.v5_2_output_moe_num_experts = 4
cfg.v5_2_output_moe_top_k = 2
cfg.v5_2_output_moe_hidden_dim = 256
cfg.v5_2_output_moe_router_temperature = 0.20
cfg.v5_2_output_moe_balance_weight = 1e-3
cfg.v5_2_output_moe_phi_ema_decay = 0.99
cfg.v5_2_output_moe_contrastive_weight = 1e-3
cfg.v5_2_output_moe_expert_init_std = 1e-4

# V4.6d: route shared expert together with routed experts instead of always-on residual addition.
cfg.v4_6_moe_route_shared_expert = False

# V3.5: Fourier low-pass post-processing
cfg.fourier_smooth_k = 0  # 0=disabled, >0=keep lowest K freq components per side

# V3.4/V3.6: Hybrid post-processing for jagged contour cleanup
cfg.hybrid_postprocess_k = 0
cfg.hybrid_postprocess_low_gain = 0.15
cfg.hybrid_postprocess_outlier_z = 2.5
cfg.hybrid_postprocess_neighbor_span = 1
cfg.hybrid_postprocess_fill_iters = 2
cfg.hybrid_postprocess_blend = 0.85

# V3.5: Fourier-space diffusion
cfg.use_dit_v3_5 = False
cfg.fourier_k = 16  # number of Fourier coefficients (K)
cfg.fourier_disp_stats = ''  # path to Fourier-domain mean/std stats JSON

# YOLO/NMS related toggles
cfg.use_nms_for_snake = True
cfg.det_conf_thresh = 0.20
cfg.det_iou_thresh = 0.30
cfg.det_max_det = 300
cfg.per_class_nms = True
cfg.yolo_pretrained = ''
cfg.load_yolo_pretrained = False
cfg.yolo_num_classes = 0
cfg.yolo_model_scale = ''
cfg.yolo_train_scope = 'head'
cfg.disable_lr_flip = False
cfg.pseudo3d_symmetric_stem_init = False
cfg.require_yolo_pretrained = False

# Sagittal pseudo-3D input defaults (three grayscale slices in three channels).
cfg.pseudo3d_mean = 0.5
cfg.pseudo3d_std = 0.5
cfg.pseudo3d_color_aug = False
cfg.pseudo3d_lr_flip = False
cfg.pseudo3d_random_crop = True
cfg.pseudo3d_input_mode = 'neighbors'

cfg.dataloader_persistent_workers = True
cfg.dataloader_prefetch_factor = 4

# Distributed training/runtime tuning.
cfg.ddp_sync_mode = 'standard'
cfg.ddp_find_unused_parameters = True
cfg.ddp_gradient_as_bucket_view = True
cfg.ddp_bucket_cap_mb = 25
cfg.use_amp = False
cfg.amp_dtype = 'bfloat16'
cfg.enable_tf32 = True
cfg.cudnn_benchmark = True
cfg.cuda_empty_cache_interval = 0

cfg.contour_init_method = 'octagon'
cfg.fm_max_disp_frac = 0.0

# -----------------------------------------------------------------------------
# train
# -----------------------------------------------------------------------------
cfg.train = CN()

cfg.train.dataset = 'VolMemTrain'
cfg.train.epoch = 140
cfg.train.max_steps = 0
cfg.train.num_workers = 8
cfg.train.gradient_accumulation_steps = 1
cfg.train.gradient_clip = 40.0
cfg.train.drop_last = False

# use adam as default
cfg.train.optim = 'adam'
cfg.train.lr = 1e-4
cfg.train.weight_decay = 5e-4

cfg.train.warmup = False
cfg.train.scheduler = ''
cfg.train.milestones = [80, 120, 200, 240]
cfg.train.gamma = 0.5

cfg.train.batch_size = 4

cfg.test = CN()
cfg.test.dataset = 'VolMemDev8'
cfg.test.batch_size = 1
cfg.test.epoch = -1
cfg.test.visual_save_root = 'data/eval_vis'

# recorder
cfg.record_dir = 'data/record'

# result
cfg.result_dir = 'data/result'

# evaluation
cfg.skip_eval = False

cfg.save_ep = 100
cfg.eval_ep = 5

cfg.use_gt_det = False
cfg.use_gt_det_train_only = False
cfg.diffusion_init_source = 'extreme'
cfg.ex_box_jitter_scale = 0.0
cfg.ex_box_jitter_shift = 0.0
cfg.pred_extreme_init_prob = -1.0  # <0 keeps legacy use_pred_extreme_init_for_diffusion behavior.

# Route-B detector-box robustness augmentation.  This is deliberately
# independent of ``flow_train_noise_scale`` (state noise) and the legacy
# ``ex_box_jitter_*`` extreme-refiner augmentation.  Severity index 0 must be
# the exact clean GT-box route; later entries describe progressively stronger
# center, log-scale, and asymmetric edge perturbations.
cfg.routeb_box_jitter_enabled = False
cfg.routeb_box_jitter_probabilities = [1.0, 0.0, 0.0, 0.0]
cfg.routeb_box_jitter_shift_fractions = [0.0, 0.05, 0.10, 0.15]
cfg.routeb_box_jitter_log_scale_fractions = [0.0, 0.10, 0.20, 0.30]
cfg.routeb_box_jitter_edge_fractions = [0.0, 0.03, 0.08, 0.15]
cfg.routeb_box_jitter_min_iou = 0.20

# -----------------------------------------------------------------------------
# snake
# -----------------------------------------------------------------------------
cfg.ct_score = 0.05


def parse_cfg(cfg, args):
    if len(cfg.task) == 0:
        raise ValueError('task must be specified')

    # assign the gpus
    grpo_gpu_override = os.environ.get('GRPO_V2_GPU', '').strip()
    if grpo_gpu_override:
        cfg.gpus = [int(grpo_gpu_override)]
    # CUDA rejects the whole list when it contains spaces (it then falls back to
    # "all devices visible"), so join without a separator space. An externally
    # provided CUDA_VISIBLE_DEVICES wins: under torchrun the launcher pins the
    # devices before this runs.
    if not os.environ.get('CUDA_VISIBLE_DEVICES', '').strip():
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(gpu) for gpu in cfg.gpus)

    cfg.det_dir = os.path.join(cfg.model_dir, cfg.task, args.det)

    # assign the network head conv
    cfg.head_conv = 64 if 'res' in cfg.network else 256

    #cfg.model_dir = os.path.join(cfg.model_dir)
    cfg.record_dir = os.path.join(cfg.record_dir, cfg.task, cfg.model)
    cfg.result_dir = os.path.join(cfg.result_dir, cfg.task, cfg.model)


def _infer_default_cfg(args):
    """Select default config file based on argv or env."""
    if getattr(args, 'cfg_file', None):
        return args.cfg_file
    env_cfg = os.environ.get("CFG_FILE", "")
    if env_cfg:
        return env_cfg
    script = Path(sys.argv[0]).stem.lower()
    if "train_rl" in script:
        return "configs/stage2_rl.yaml"
    return "configs/stage1.yaml"


def make_cfg(args):
    cfg_file = _infer_default_cfg(args)
    cfg.merge_from_file(cfg_file)  # 从推断的配置文件中加载配置项
    extra_opts = args.opts if args.opts is not None else []
    cfg.merge_from_list(extra_opts)  # 从命令行参数列表中合并额外的配置项
    parse_cfg(cfg, args)
    return cfg


parser = argparse.ArgumentParser()
parser.add_argument("--cfg_file", default="", type=str)
parser.add_argument("--ckpt", default="", type=str)
parser.add_argument('--vis_GT', action='store_true', dest='vis_GT', default=False)
parser.add_argument("--type", type=str, default="")
parser.add_argument('--det', type=str, default='')
parser.add_argument('-f', type=str, default='')
parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)
args = parser.parse_args()

if len(args.type) > 0:
    cfg.task = "run"
    print("!111！")
cfg = make_cfg(args)
