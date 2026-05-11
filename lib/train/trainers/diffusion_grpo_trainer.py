import torch
import torch.nn as nn
import torch.nn.functional as F
import contextlib
import math
import os
import numpy as np
import copy
from pathlib import Path
from lib.config import cfg
from lib.networks.YOLOV8.utils.loss import v8DetectionLoss
from lib.utils.snake import snake_gcn_utils, snake_config
from lib.train.rewards.region_reward import compute_region_reward, compute_region_score
from lib.networks.YOLOV8.utils.ops import non_max_suppression
from lib.networks.diffusion.ddpm_with_logprob import ddpm_step_with_logprob
from lib.networks.diffusion.pretrain_evolution import remap_legacy_state_dict


class DiffusionGRPONetworkWrapper(nn.Module):
    """
    GRPO 训练专用：包含 YOLO 检测损失、Diffusion 去噪损失与 GRPO 策略损失。
    """

    def __init__(self, net: nn.Module):
        super().__init__()
        self.net = net
        self.disable_grpo = os.environ.get('DISABLE_GRPO', '0') != '0'
        raw_use_grpo = bool(getattr(cfg, 'use_grpo', False)) or bool(getattr(cfg, 'use_grpo_kl', False)) or bool(getattr(getattr(cfg, 'train', cfg), 'use_grpo', False))
        self.flow_posttrain_mode = bool(getattr(cfg, 'use_flow_matching', False))
        self.allow_grpo = raw_use_grpo and (not self.disable_grpo)
        self.use_flow_grpo = self.allow_grpo and self.flow_posttrain_mode
        self.enable_reward_logging = raw_use_grpo and (not self.disable_grpo)

        # YOLO 损失
        try:
            self.det_crit = self.net.yolo.init_criterion()
        except Exception:
            self.det_crit = v8DetectionLoss(self.net.yolo)

        # 先验损失
        self.L_crit = F.mse_loss

        # 损失权重
        default_scales = {'det': 1.0, 'diff': 1.0, 'prior': 1.0}
        self.loss_scales = getattr(cfg, 'loss_scales', default_scales)
        for k, v in default_scales.items():
            if k not in self.loss_scales:
                self.loss_scales[k] = v

        # 冻结开关
        self.freeze_snake = bool(getattr(cfg, 'freeze_snake', False))
        self.freeze_yolo = bool(getattr(cfg, 'freeze_yolo', False))

        if self.freeze_yolo:
            try:
                for p in self.net.yolo.parameters():
                    p.requires_grad = False
                self.net.yolo.eval()
            except Exception:
                pass

        # GRPO KL & PPO settings (Flow-GRPO style names)
        beta_train = None
        try:
            beta_train = getattr(cfg.train, 'beta')
        except Exception:
            beta_train = None
        if beta_train is not None and float(beta_train) > 0.0:
            self.use_kl = True
            self.kl_beta = float(beta_train)
        else:
            self.use_kl = bool(getattr(cfg, 'use_grpo_kl', False))
            self.kl_beta = float(getattr(cfg, 'grpo_kl_beta', 0.0))
        try:
            self.clip_range = float(getattr(cfg.train, 'clip_range'))
        except Exception:
            self.clip_range = float(getattr(cfg, 'grpo_clip_range', 0.2))
        try:
            self.adv_clip_max = float(getattr(cfg.train, 'adv_clip_max'))
        except Exception:
            self.adv_clip_max = float(getattr(cfg, 'grpo_adv_clip_max', 5.0))
        self.ref_gcn = None  # lazy snapshot after weights are loaded
        self.ref_flow = None  # lazy snapshot for flow-matching PPO/GRPO KL
        self.ref_flow_path = ''
        self.ref_flow_load_ratio = 0.0

        # 若不允许 GRPO，则移除 gcn sampler，避免 logprob/采样
        if not self.allow_grpo:
            try:
                if hasattr(self.net, 'gcn') and hasattr(self.net.gcn, 'sampler'):
                    self.net.gcn.sampler = None
            except Exception:
                pass

    def _set_unregistered_ref(self, name: str, module):
        if hasattr(self, '_modules') and name in self._modules:
            del self._modules[name]
        self.__dict__[name] = module

    @staticmethod
    def _extract_state_dict(ckpt_obj):
        if not isinstance(ckpt_obj, dict):
            return ckpt_obj
        for key in ('state_dict', 'model', 'net', 'network'):
            if key in ckpt_obj and isinstance(ckpt_obj[key], dict):
                return ckpt_obj[key]
        return ckpt_obj

    @staticmethod
    def _resolve_project_path(path_like: str) -> str:
        path = Path(str(path_like)).expanduser()
        if path.is_absolute():
            return str(path)
        return str((Path(__file__).resolve().parents[3] / path).resolve())

    @staticmethod
    def _slice_submodule_state_dict(state_dict, submodule, prefixes):
        if not isinstance(state_dict, dict):
            return {}
        state_dict = remap_legacy_state_dict(state_dict)
        module_state = submodule.state_dict()
        sliced = {}
        for raw_key, value in state_dict.items():
            key = raw_key[len('module.'):] if raw_key.startswith('module.') else raw_key
            candidates = [key]
            for prefix in prefixes:
                if key.startswith(prefix):
                    candidates.append(key[len(prefix):])
            for candidate in candidates:
                if candidate in module_state and tuple(module_state[candidate].shape) == tuple(value.shape):
                    sliced[candidate] = value
                    break
        return sliced

    def _load_fixed_ref_flow(self, base_device):
        ref_path = str(
            os.environ.get('GRPO_REF_PATH', '')
            or getattr(cfg, 'grpo_reference_path', '')
            or getattr(cfg, 'grpo_ref_path', '')
            or ''
        ).strip()

        ref_flow = copy.deepcopy(self.net.gcn).to(base_device)
        if ref_path:
            resolved = self._resolve_project_path(ref_path)
            ckpt_obj = torch.load(resolved, map_location='cpu')
            sd = self._extract_state_dict(ckpt_obj)
            ref_sd = self._slice_submodule_state_dict(
                sd,
                ref_flow,
                prefixes=('net.gcn.', 'gcn.'),
            )
            if not ref_sd:
                raise RuntimeError(f'No gcn keys could be loaded from GRPO reference: {resolved}')
            missing, unexpected = ref_flow.load_state_dict(ref_sd, strict=False)
            total = max(len(ref_flow.state_dict()), 1)
            loaded = total - len(missing)
            load_ratio = float(loaded) / float(total)
            min_ratio = float(getattr(cfg, 'grpo_reference_min_load_ratio', 0.95))
            if load_ratio < min_ratio:
                raise RuntimeError(
                    f'GRPO reference load ratio too low: {load_ratio:.4f} < {min_ratio:.4f} '
                    f'for {resolved}; unexpected={len(unexpected)}'
                )
            self.ref_flow_path = resolved
            self.ref_flow_load_ratio = load_ratio
            print(f'[GRPO] fixed reference loaded: {resolved} ({loaded}/{total}, {load_ratio:.2%})')
        else:
            self.ref_flow_path = '<current-model-snapshot>'
            self.ref_flow_load_ratio = 1.0
            print('[GRPO] no fixed reference path configured; using current model snapshot')

        for p in ref_flow.parameters():
            p.requires_grad = False
        ref_flow.eval()
        return ref_flow

    @staticmethod
    def _shape_smoothness_penalty(poly_py: torch.Tensor) -> torch.Tensor:
        if poly_py.numel() == 0:
            return poly_py.new_zeros((0,))
        prev_pt = torch.roll(poly_py, 1, dims=1)
        next_pt = torch.roll(poly_py, -1, dims=1)
        second = next_pt - 2.0 * poly_py + prev_pt
        bbox_min = poly_py.amin(dim=1)
        bbox_max = poly_py.amax(dim=1)
        scale = (bbox_max - bbox_min).norm(dim=-1).clamp_min(1.0)
        return second.norm(dim=-1).mean(dim=1) / scale

    @staticmethod
    def _align_gt_to_init(i_init_py: torch.Tensor, i_gt_py: torch.Tensor) -> torch.Tensor:
        def _signed_area(poly: torch.Tensor) -> torch.Tensor:
            x = poly[..., 0]
            y = poly[..., 1]
            x1 = torch.roll(x, shifts=-1, dims=1)
            y1 = torch.roll(y, shifts=-1, dims=1)
            return 0.5 * torch.sum(x * y1 - x1 * y, dim=1)

        if i_init_py.numel() == 0 or i_gt_py.numel() == 0:
            return i_gt_py

        i_gt_py = i_gt_py.clone()
        area_init = _signed_area(i_init_py)
        area_gt = _signed_area(i_gt_py)
        orient_mismatch = ((area_init >= 0) ^ (area_gt >= 0))
        if orient_mismatch.any():
            i_gt_py[orient_mismatch] = torch.flip(i_gt_py[orient_mismatch], dims=[1])

        d2 = (i_init_py[:, :1, :] - i_gt_py).pow(2).sum(-1)
        nearest = torch.argmin(d2, dim=1)
        return torch.stack([
            torch.roll(i_gt_py[i], shifts=-int(nearest[i].item()), dims=0)
            for i in range(i_gt_py.size(0))
        ], dim=0)

    @staticmethod
    def _first_contour_filter(py_ind: torch.Tensor) -> torch.Tensor:
        seen = set()
        keep_idx = []
        for idx in range(py_ind.numel()):
            b = int(py_ind[idx].item())
            if b not in seen:
                seen.add(b)
                keep_idx.append(idx)
        return torch.as_tensor(keep_idx, device=py_ind.device, dtype=torch.long)

    @staticmethod
    def _max_contour_filter(py_ind: torch.Tensor, max_per_image: int):
        if py_ind.numel() == 0 or max_per_image <= 0:
            return None
        keep_idx = []
        for b in torch.unique(py_ind.detach()).tolist():
            idx = torch.nonzero(py_ind == int(b), as_tuple=False).flatten()
            if idx.numel() > max_per_image:
                idx = idx[torch.randperm(idx.numel(), device=idx.device)[:max_per_image]]
            keep_idx.append(idx)
        if not keep_idx:
            return None
        return torch.sort(torch.cat(keep_idx, dim=0))[0]

    def _flow_grpo_loss(self, output, batch, base_device):
        cnn_feature = output.get('cnn_feature', None)
        i_init_py = output.get('i_it_py', None)
        c_init_py = output.get('c_it_py', None)
        i_gt_py = output.get('i_gt_py', None)
        py_ind = output.get('py_ind', None)
        feat_hw = output.get('feat_hw', None)

        zero = torch.tensor(0.0, device=base_device)
        if (
            not isinstance(cnn_feature, torch.Tensor)
            or not isinstance(i_init_py, torch.Tensor)
            or not isinstance(c_init_py, torch.Tensor)
            or not isinstance(i_gt_py, torch.Tensor)
            or i_init_py.numel() == 0
        ):
            return zero, {
                'grpo_loss': zero,
                'grpo_policy_loss_raw': zero,
                'grpo_reward_mean': zero,
                'grpo_reward_std': zero,
                'grpo_reward_best': zero,
                'grpo_approx_kl': zero,
                'grpo_clipfrac': zero,
                'grpo_action_std': zero,
            }

        if isinstance(py_ind, torch.Tensor) and bool(getattr(cfg, 'grpo_first_contour_only', False)):
            keep = self._first_contour_filter(py_ind)
            i_init_py = i_init_py.index_select(0, keep)
            c_init_py = c_init_py.index_select(0, keep)
            i_gt_py = i_gt_py.index_select(0, keep)
            py_ind = py_ind.index_select(0, keep)
        elif isinstance(py_ind, torch.Tensor):
            keep = self._max_contour_filter(py_ind, int(getattr(cfg, 'grpo_max_contours_per_image', 0)))
            if keep is not None:
                i_init_py = i_init_py.index_select(0, keep)
                c_init_py = c_init_py.index_select(0, keep)
                i_gt_py = i_gt_py.index_select(0, keep)
                py_ind = py_ind.index_select(0, keep)

        if i_init_py.numel() == 0:
            return zero, {
                'grpo_loss': zero,
                'grpo_policy_loss_raw': zero,
                'grpo_reward_mean': zero,
                'grpo_reward_std': zero,
                'grpo_reward_best': zero,
                'grpo_approx_kl': zero,
                'grpo_clipfrac': zero,
                'grpo_action_std': zero,
            }

        i_gt_py = self._align_gt_to_init(i_init_py, i_gt_py)

        if isinstance(feat_hw, (tuple, list)) and len(feat_hw) == 2:
            H, W = int(feat_hw[0]), int(feat_hw[1])
        else:
            H = max(int(batch['inp'].shape[-2] // int(snake_config.down_ratio)), 1)
            W = max(int(batch['inp'].shape[-1] // int(snake_config.down_ratio)), 1)
        reward_coord_scale = 1.0
        if bool(getattr(cfg, 'grpo_reward_image_scale', False)) and 'inp' in batch:
            H = int(batch['inp'].shape[-2])
            W = int(batch['inp'].shape[-1])
            reward_coord_scale = float(snake_config.down_ratio)

        k = max(int(getattr(cfg, 'grpo_k', 4)), 1)
        action_std = max(float(getattr(cfg, 'grpo_action_std', 0.75)), 1e-6)
        reward_w = float(getattr(cfg, 'reward_w_region', 1.0))
        reward_w_dice = float(getattr(cfg, 'reward_w_dice', 0.0))
        reward_w_iou = float(getattr(cfg, 'reward_w_iou', 0.0))
        reward_delta = bool(getattr(cfg, 'grpo_reward_delta', False))
        reward_abs_weight_raw = getattr(cfg, 'grpo_reward_abs_weight', None)
        reward_delta_weight_raw = getattr(cfg, 'grpo_reward_delta_weight', None)
        if reward_abs_weight_raw is None and reward_delta_weight_raw is None:
            reward_abs_weight = 0.0 if reward_delta else 1.0
            reward_delta_weight = 1.0 if reward_delta else 0.0
        else:
            reward_abs_weight = float(0.0 if reward_abs_weight_raw is None else reward_abs_weight_raw)
            reward_delta_weight = float(0.0 if reward_delta_weight_raw is None else reward_delta_weight_raw)
        smoothness_weight = float(getattr(cfg, 'grpo_smoothness_weight', 0.0))
        adv_center = str(getattr(cfg, 'grpo_adv_center', 'group')).lower()
        loss_weight = float(getattr(cfg, 'grpo_loss_weight', 1.0))
        clip_eps = float(self.clip_range)
        adv_clip_max = float(getattr(cfg, 'grpo_adv_clip_max', self.adv_clip_max))

        if self.use_kl and self.ref_flow is None:
            try:
                self._set_unregistered_ref('ref_flow', self._load_fixed_ref_flow(base_device))
            except Exception:
                self._set_unregistered_ref('ref_flow', None)
                raise

        rollout_steps = max(int(getattr(cfg, 'grpo_steps', getattr(cfg, 'flow_ode_steps', 20))), 1)
        window_size = int(getattr(cfg, 'grpo_window_size', 0))
        window_range = getattr(cfg, 'grpo_window_range', (0, max(rollout_steps - 1, 0)))

        rewards_k_list = []
        final_scores_k_list = []
        delta_scores_k_list = []
        smooth_penalty_k_list = []
        old_logs_list = []
        trajs_list = []
        init_reward = None
        if reward_delta or reward_delta_weight != 0.0:
            init_reward = compute_region_score(
                i_init_py,
                i_gt_py,
                H=H,
                W=W,
                w_boundary=reward_w,
                w_dice=reward_w_dice,
                w_iou=reward_w_iou,
                coord_scale=reward_coord_scale,
            ).detach().cpu()
        for _ in range(k):
            with torch.no_grad():
                ret = self.net.gcn.sample_with_logprob(
                    cnn_feature,
                    i_init_py,
                    c_init_py,
                    py_ind,
                    steps=rollout_steps,
                    window_size=window_size,
                    window_range=window_range,
                    action_std=action_std,
                )
            if isinstance(ret.get('log_probs', None), list) and len(ret['log_probs']) > 0:
                lp_steps = torch.stack(ret['log_probs'], dim=0).transpose(0, 1).contiguous().detach().cpu()
                trajs_list.append({
                    'timesteps': [t.detach().cpu() for t in ret['timesteps']],
                    'step_indices': [s.detach().cpu() for s in ret['step_indices']],
                    'x_ts': [x.detach().cpu() for x in ret['x_ts']],
                    'x_prevs': [x.detach().cpu() for x in ret['x_prevs']],
                    'x_self_conds': [None if x is None else x.detach().cpu() for x in ret['x_self_conds']],
                })
                old_logs_list.append(lp_steps)
                final_reward = compute_region_reward(
                    i_init_py,
                    ret['disp'],
                    i_gt_py,
                    H=H,
                    W=W,
                    w1=reward_w,
                    w_dice=reward_w_dice,
                    w_iou=reward_w_iou,
                    coord_scale=reward_coord_scale,
                )
                final_scores_k_list.append(final_reward.detach().cpu())
                final_reward_cpu = final_reward.detach().cpu()
                if init_reward is not None:
                    delta_reward_cpu = final_reward_cpu - init_reward
                else:
                    delta_reward_cpu = torch.zeros_like(final_reward_cpu)
                final_poly = (i_init_py + ret['disp']).detach()
                smooth_penalty = self._shape_smoothness_penalty(final_poly).detach().cpu()
                policy_reward = (
                    reward_abs_weight * final_reward_cpu
                    + reward_delta_weight * delta_reward_cpu
                    - smoothness_weight * smooth_penalty
                )
                rewards_k_list.append(policy_reward)
                delta_scores_k_list.append(delta_reward_cpu)
                smooth_penalty_k_list.append(smooth_penalty)
            del ret
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if len(rewards_k_list) == 0:
            return zero, {
                'grpo_loss': zero,
                'grpo_policy_loss_raw': zero,
                'grpo_reward_mean': zero,
                'grpo_reward_std': zero,
                'grpo_reward_best': zero,
                'grpo_approx_kl': zero,
                'grpo_clipfrac': zero,
                'grpo_action_std': torch.tensor(action_std, device=base_device),
            }

        rewards_k = torch.stack(rewards_k_list, dim=0).to(base_device)
        final_scores_k = torch.stack(final_scores_k_list, dim=0).to(base_device)
        delta_scores_k = torch.stack(delta_scores_k_list, dim=0).to(base_device) if delta_scores_k_list else torch.zeros_like(final_scores_k)
        smooth_penalty_k = torch.stack(smooth_penalty_k_list, dim=0).to(base_device) if smooth_penalty_k_list else torch.zeros_like(final_scores_k)
        if adv_center == 'global':
            advantages = rewards_k - rewards_k.mean()
            adv_scale = rewards_k.std(unbiased=False).clamp_min(1e-6)
        else:
            advantages = rewards_k - rewards_k.mean(dim=0, keepdim=True)
            adv_scale = rewards_k.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-6)
        if bool(getattr(cfg, 'grpo_normalize_adv', True)) and k > 1:
            advantages = advantages / adv_scale
        advantages = advantages.clamp(-adv_clip_max, adv_clip_max).detach()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        grpo_cnn_feature = cnn_feature.detach()
        flow_ctx = self.net.gcn.prepare_sampling_context(grpo_cnn_feature, i_init_py, py_ind)
        policy_loss_raw_sum = torch.zeros((), device=base_device)
        policy_loss_term_count = 0
        grpo_kl_loss_sum = torch.zeros((), device=base_device)
        grpo_kl_loss_term_count = 0
        approx_kl_sum = torch.zeros((), device=base_device)
        approx_kl_term_count = 0
        clipfrac_sum = torch.zeros((), device=base_device)
        clipfrac_term_count = 0

        for traj_idx, traj in enumerate(trajs_list):
            old_logs = old_logs_list[traj_idx].to(base_device)
            adv = advantages[traj_idx]
            traj_policy_loss = torch.zeros((), device=base_device)
            traj_step_count = 0
            traj_kl_sum = torch.zeros((), device=base_device)
            traj_kl_count = 0

            traj_timesteps = [t.to(base_device) for t in traj['timesteps']]
            traj_step_indices = [s.to(base_device) for s in traj['step_indices']]
            traj_x_ts = [x.to(base_device) for x in traj['x_ts']]
            traj_x_prevs = [x.to(base_device) for x in traj['x_prevs']]
            traj_x_self_conds = [None if x is None else x.to(base_device) for x in traj['x_self_conds']]

            for s_idx in range(len(traj_timesteps)):
                x_self_cond = traj_x_self_conds[s_idx] if s_idx < len(traj_x_self_conds) else None
                _, lp_cur, mean_cur, std_cur, _ = self.net.gcn.step_with_logprob(
                    grpo_cnn_feature,
                    i_init_py,
                    c_init_py,
                    py_ind,
                    x_t=traj_x_ts[s_idx],
                    t_value=traj_timesteps[s_idx],
                    step_index=int(traj_step_indices[s_idx].item()),
                    total_steps=rollout_steps,
                    action_std=action_std,
                    prev_sample=traj_x_prevs[s_idx],
                    sampled_feat=flow_ctx['sampled_feat'],
                    detail_feat=flow_ctx['detail_feat'],
                    contour_scale=flow_ctx['contour_scale'],
                    x_self_cond=x_self_cond,
                )

                old_log = old_logs[:, s_idx].to(base_device)
                ratio = torch.exp(lp_cur - old_log)
                unclipped = -adv * ratio
                clipped = -adv * torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps)
                traj_policy_loss = traj_policy_loss + torch.maximum(unclipped, clipped).mean()
                traj_step_count += 1
                approx_kl_sum = approx_kl_sum + (0.5 * torch.mean((lp_cur - old_log) ** 2)).detach()
                approx_kl_term_count += 1
                clipfrac_sum = clipfrac_sum + torch.mean((torch.abs(ratio - 1.0) > clip_eps).float()).detach()
                clipfrac_term_count += 1

                if self.use_kl and (self.ref_flow is not None) and self.kl_beta > 0.0:
                    with torch.no_grad():
                        _, _, mean_ref, _, _ = self.ref_flow.step_with_logprob(
                            grpo_cnn_feature,
                            i_init_py,
                            c_init_py,
                            py_ind,
                            x_t=traj_x_ts[s_idx],
                            t_value=traj_timesteps[s_idx],
                            step_index=int(traj_step_indices[s_idx].item()),
                            total_steps=rollout_steps,
                            action_std=action_std,
                            prev_sample=traj_x_prevs[s_idx],
                            sampled_feat=flow_ctx['sampled_feat'],
                            detail_feat=flow_ctx['detail_feat'],
                            contour_scale=flow_ctx['contour_scale'],
                            x_self_cond=x_self_cond,
                        )
                    var = std_cur.pow(2).clamp_min(1e-12)
                    traj_kl_sum = traj_kl_sum + (((mean_cur - mean_ref) ** 2) / (2.0 * var)).mean()
                    traj_kl_count += 1

            if traj_step_count > 0:
                policy_loss_raw_sum = policy_loss_raw_sum + traj_policy_loss / max(traj_step_count, 1)
                policy_loss_term_count += 1
                if self.use_kl and self.kl_beta > 0.0 and traj_kl_count > 0:
                    grpo_kl_loss_sum = grpo_kl_loss_sum + traj_kl_sum / max(traj_kl_count, 1)
                    grpo_kl_loss_term_count += 1

            del traj_x_ts, traj_x_prevs, traj_x_self_conds, traj_timesteps, traj_step_indices
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if policy_loss_term_count == 0:
            return zero, {
                'grpo_loss': zero,
                'grpo_policy_loss_raw': zero,
                'grpo_reward_mean': rewards_k.mean(),
                'grpo_reward_std': rewards_k.std(unbiased=False),
                'grpo_reward_best': rewards_k.max(dim=0)[0].mean(),
                'grpo_final_score_mean': final_scores_k.mean(),
                'grpo_final_score_best': final_scores_k.max(dim=0)[0].mean(),
                'grpo_delta_score_mean': delta_scores_k.mean(),
                'grpo_smoothness_penalty_mean': smooth_penalty_k.mean(),
                'grpo_adv_mean': advantages.mean(),
                'grpo_adv_std': advantages.std(unbiased=False),
                'grpo_approx_kl': zero,
                'grpo_clipfrac': zero,
                'grpo_action_std': torch.tensor(action_std, device=base_device),
            }

        policy_loss_raw = policy_loss_raw_sum / max(policy_loss_term_count, 1)
        policy_loss = loss_weight * policy_loss_raw
        grpo_kl_loss = zero
        if self.use_kl and self.kl_beta > 0.0 and grpo_kl_loss_term_count > 0:
            grpo_kl_loss = grpo_kl_loss_sum / max(grpo_kl_loss_term_count, 1)
            policy_loss = policy_loss + self.kl_beta * grpo_kl_loss

        return policy_loss, {
            'grpo_loss': policy_loss,
            'grpo_policy_loss_raw': policy_loss_raw,
            'grpo_reward_mean': rewards_k.mean(),
            'grpo_reward_std': rewards_k.std(unbiased=False),
            'grpo_reward_best': rewards_k.max(dim=0)[0].mean(),
            'grpo_final_score_mean': final_scores_k.mean(),
            'grpo_final_score_best': final_scores_k.max(dim=0)[0].mean(),
            'grpo_delta_score_mean': delta_scores_k.mean(),
            'grpo_smoothness_penalty_mean': smooth_penalty_k.mean(),
            'grpo_reward_abs_weight': torch.tensor(reward_abs_weight, device=base_device),
            'grpo_reward_delta_weight': torch.tensor(reward_delta_weight, device=base_device),
            'grpo_smoothness_weight': torch.tensor(smoothness_weight, device=base_device),
            'grpo_adv_mean': advantages.mean(),
            'grpo_adv_std': advantages.std(unbiased=False),
            'grpo_action_std': torch.tensor(action_std, device=base_device),
            'grpo_approx_kl': approx_kl_sum / max(approx_kl_term_count, 1) if approx_kl_term_count > 0 else zero,
            'grpo_clipfrac': clipfrac_sum / max(clipfrac_term_count, 1) if clipfrac_term_count > 0 else zero,
            'grpo_kl_loss': grpo_kl_loss,
            'grpo_ref_load_ratio': torch.tensor(float(self.ref_flow_load_ratio), device=base_device),
        }

    def forward(self, batch):
        use_grpo = self.allow_grpo
        if not hasattr(self, '_log_use_grpo_once'):
            try:
                print(
                    f"[DEBUG] DiffusionGRPONetworkWrapper use_grpo={use_grpo}, "
                    f"flow_posttrain_mode={self.flow_posttrain_mode}, "
                    f"use_flow_grpo={self.use_flow_grpo}, "
                    f"freeze_yolo={self.freeze_yolo}, freeze_snake={self.freeze_snake}, "
                    f"disable_grpo_env={self.disable_grpo}"
                )
            except Exception:
                pass
            self._log_use_grpo_once = True
        if use_grpo and self.flow_posttrain_mode:
            output = self.net(batch['inp'], batch)
        elif use_grpo:
            # 惰性创建参考策略（冻结的GCN副本），仅在启用KL时使用
            if self.use_kl and self.ref_gcn is None:
                try:
                    ref_gcn = copy.deepcopy(self.net.gcn).cuda()  # 深拷贝GCN到GPU
                    for p in ref_gcn.parameters():
                        p.requires_grad = False
                    ref_gcn.eval()
                    self._set_unregistered_ref('ref_gcn', ref_gcn)
                except Exception:
                    self._set_unregistered_ref('ref_gcn', None)
            x = batch['inp']
            # 强制YOLO在eval+no_grad下运行，避免GRPO期间BN统计被修改
            if self.freeze_yolo:
                was_training = bool(self.net.yolo.training)
                self.net.yolo.eval()
                ctx = torch.no_grad()
            else:
                was_training = False
                ctx = contextlib.nullcontext()
            with ctx:
                yolo_out = self.net.yolo(x)
            if self.freeze_yolo and was_training:
                self.net.yolo.train()
            if isinstance(yolo_out, tuple) and len(yolo_out) >= 2:
                yolo_y, yolo_feats = yolo_out[0], yolo_out[1]
            else:
                yolo_y, yolo_feats = yolo_out, []
            p2 = yolo_feats[0] if isinstance(yolo_feats, (list, tuple)) and len(yolo_feats) > 0 else None
            if p2 is None:
                raise RuntimeError("YOLO head features are not available; expected a list with P2 at index 0.")
            if self.freeze_yolo:
                p2 = p2.detach()

            cnn_feature = self.net.cnn_proj(p2)
            h, w = cnn_feature.size(2), cnn_feature.size(3)
            h_img, w_img = h * int(snake_config.down_ratio), w * int(snake_config.down_ratio)
            raw_det = self.net.decode_detection_from_yolo(yolo_y, h_img, w_img)
            use_nms = bool(getattr(cfg, 'use_nms_for_snake', True))
            conf_thres = float(getattr(cfg, 'det_conf_thresh', 0.20))
            iou_thres = float(getattr(cfg, 'det_iou_thresh', 0.30))
            max_det = int(getattr(cfg, 'det_max_det', 300))
            per_class = bool(getattr(cfg, 'per_class_nms', True))
            if use_nms:
                y = yolo_y.permute(0, 2, 1).contiguous()
                xywh = y[..., :4]
                cls_logits = y[..., 4:]
                cls_prob = cls_logits.sigmoid()
                x_c, y_c, w_box, h_box = xywh.unbind(-1)
                x1 = x_c - w_box / 2
                y1 = y_c - h_box / 2
                x2 = x_c + w_box / 2
                y2 = y_c + h_box / 2
                boxes = torch.stack([x1, y1, x2, y2], dim=-1)
                boxes = boxes.clamp(min=0)
                pred = torch.cat([boxes, cls_prob], dim=-1).permute(0, 2, 1).contiguous()
                nms_out = non_max_suppression(
                    pred,
                    conf_thres=conf_thres,
                    iou_thres=iou_thres,
                    classes=None,
                    agnostic=not per_class,
                    multi_label=False,
                    labels=(),
                    max_det=max_det,
                    nc=cls_prob.shape[-1],
                    in_place=True,
                    rotated=True,
                )
                max_len = max((d.size(0) for d in nms_out), default=0)
                if max_len == 0:
                    detection = raw_det.new_zeros((raw_det.size(0), 0, 6))
                else:
                    detection = raw_det.new_zeros((raw_det.size(0), max_len, 6))
                    for b, det_b in enumerate(nms_out):
                        if det_b is not None and det_b.size(0) > 0:
                            detection[b, :det_b.size(0)] = det_b[:, :6]
            else:
                detection = raw_det
            output = {'detection': detection, 'feat_hw': (h, w)}
            if getattr(cfg, 'use_gt_det', False):
                self.net.use_gt_detection(output, batch)
        else:
            output = self.net(batch['inp'], batch)

        # 总损失与标量统计
        base_device = output['detection'].device if isinstance(output.get('detection', None), torch.Tensor) else batch['inp'].device
        loss = torch.zeros(1, device=base_device).squeeze()
        scalar_stats = {}
        image_stats = {}
        if not self.training:
            scalar_stats.update({
                'loss': loss,
                'det_loss': loss,
                'diff_loss': loss,
                'grpo_loss': loss,
            })
            return output, loss, scalar_stats, image_stats

        # 1) YOLO 检测损失
        if (not self.freeze_yolo) and ('yolo_preds' in output) and (self.det_crit is not None):
            det_loss, det_items = self.det_crit(output['yolo_preds'], batch)
            box_l, cls_l, dfl_l = det_items[0], det_items[1], det_items[2]
            det_weight = float(self.loss_scales.get('det', 1.0))
            loss = loss + det_weight * det_loss
            bs = int(batch.get('inp').shape[0]) if isinstance(batch, dict) and isinstance(batch.get('inp', None), torch.Tensor) else 1
            bs = max(bs, 1)
            det_loss_log = det_loss / float(bs)
            scalar_stats.update({
                'det_loss': det_loss_log,
                'det_box': box_l,
                'det_cls': cls_l,
                'det_dfl': dfl_l,
                'det_loss_scaled': det_weight * det_loss_log,
            })
        else:
            scalar_stats.update({'det_loss': torch.tensor(0.0, device=base_device)})

        # 2) Diffusion 去噪损失 或 GRPO 策略损失
        if use_grpo and self.flow_posttrain_mode and (not self.freeze_snake):
            if 'diff_loss' not in output:
                raise RuntimeError('Flow-GRPO requires Flow Matching output with diff_loss.')
            diff_weight = float(getattr(cfg, 'diffusion_loss_weight', 1.0))
            diff_loss = diff_weight * output['diff_loss']
            pure_rl_loss = bool(getattr(cfg, 'grpo_pure_rl_loss', False))
            grpo_loss, grpo_stats = self._flow_grpo_loss(output, batch, base_device)
            loss = loss + grpo_loss
            if not pure_rl_loss:
                loss = loss + diff_loss
            scalar_stats.update({
                'diff_loss': output['diff_loss'],
                'diff_loss_scaled': torch.zeros_like(diff_loss) if pure_rl_loss else diff_loss,
                'grpo_pure_rl_loss': torch.tensor(float(pure_rl_loss), device=base_device),
            })
            scalar_stats.update(grpo_stats)
            try:
                for k_, v_ in output.items():
                    if isinstance(k_, str) and k_.startswith('diff_loss'):
                        scalar_stats[k_] = v_
            except Exception:
                pass
        elif use_grpo and (not self.freeze_snake):
            with torch.no_grad():
                init = snake_gcn_utils.prepare_training(output, batch)
            i_gt_py = init['i_gt_py']
            py_ind = init['py_ind']
            i_init_train_py = init['i_it_py']
            c_init_train_py = init['c_it_py']

            def _signed_area(poly: torch.Tensor) -> torch.Tensor:
                x = poly[..., 0]
                y = poly[..., 1]
                x1 = torch.roll(x, shifts=-1, dims=1)
                y1 = torch.roll(y, shifts=-1, dims=1)
                return 0.5 * torch.sum(x * y1 - x1 * y, dim=1)

            area_init = _signed_area(i_init_train_py)
            area_gt = _signed_area(i_gt_py)
            orient_mismatch = ((area_init >= 0) ^ (area_gt >= 0))
            if orient_mismatch.any():
                i_gt_py[orient_mismatch] = torch.flip(i_gt_py[orient_mismatch], dims=[1])
            d2 = (i_init_train_py[:, :1, :] - i_gt_py).pow(2).sum(-1)
            nearest = torch.argmin(d2, dim=1)
            if i_gt_py.size(0) > 0:
                rolled = []
                for i in range(i_gt_py.size(0)):
                    s = int(nearest[i].item())
                    if s != 0:
                        rolled.append(torch.roll(i_gt_py[i], shifts=-s, dims=0))
                    else:
                        rolled.append(i_gt_py[i])
                i_gt_py = torch.stack(rolled, dim=0)

            i_it_py = i_init_train_py
            c_it_py = c_init_train_py
            if bool(getattr(cfg, 'grpo_first_contour_only', False)) and i_it_py.size(0) > 0:
                with torch.no_grad():
                    uniq = []
                    keep_idx = []
                    for idx in range(py_ind.numel()):
                        b = int(py_ind[idx].item())
                        if b not in uniq:
                            uniq.append(b)
                            keep_idx.append(idx)
                    keep = torch.as_tensor(keep_idx, device=py_ind.device, dtype=torch.long)
                i_it_py = i_it_py.index_select(0, keep)
                c_it_py = c_it_py.index_select(0, keep)
                i_gt_py = i_gt_py.index_select(0, keep)
                py_ind = py_ind.index_select(0, keep)
            if i_it_py.size(0) == 0:
                zero_term = cnn_feature.sum() * 0.0
                loss = loss + zero_term
                scalar_stats.update({'grpo_loss': torch.tensor(0.0, device=base_device), 'grpo_reward_mean': torch.tensor(0.0, device=base_device), 'grpo_reward_std': torch.tensor(0.0, device=base_device)})
            else:
                steps = int(getattr(cfg, 'grpo_steps', 20))
                k = int(getattr(cfg, 'grpo_k', 4))
                window_size = int(getattr(cfg, 'grpo_window_size', 6))
                window_range = getattr(cfg, 'grpo_window_range', (2, 12))
                w1 = float(getattr(cfg, 'reward_w_region', 1.0))

                total_policy_loss = torch.zeros((), device=base_device)

                rewards_list = []
                old_logs_list = []
                trajs_list = []
                for _ in range(k):
                    with torch.no_grad():
                        ret = self.net.gcn.sampler.sample_with_logprob(
                            cnn_feature, i_it_py, c_it_py, py_ind,
                            steps=steps, window_size=window_size, window_range=window_range
                        )
                    if isinstance(ret['log_probs'], list) and len(ret['log_probs']) > 0:
                        lp_steps = torch.stack(ret['log_probs'], dim=0).transpose(0, 1).contiguous()
                        old_logs_list.append(lp_steps.detach())
                        disp = ret['disp']
                        rew = compute_region_reward(i_it_py, disp, i_gt_py, H=h, W=w, w1=w1)
                        rewards_list.append(rew.detach())
                        trajs_list.append((ret['timesteps'], ret['x_ts'], ret['x_prevs']))
                if len(rewards_list) == 0:
                    loss = loss + (cnn_feature.sum() * 0.0)
                    scalar_stats.update({'grpo_loss': torch.tensor(0.0, device=base_device), 'grpo_reward_mean': torch.tensor(0.0, device=base_device), 'grpo_reward_std': torch.tensor(0.0, device=base_device)})
                else:
                    rewards_k = torch.stack(rewards_list, dim=0)
                    adv_k = rewards_k - rewards_k.mean(dim=0, keepdim=True)
                    scalar_stats.update({
                        'grpo_reward_mean': rewards_k.mean(),
                        'grpo_reward_std': rewards_k.std(unbiased=False)
                    })

                    clip_eps = self.clip_range
                    adv_clip_max = self.adv_clip_max

                    pl_terms = []
                    kl_terms = []
                    num_traj = len(trajs_list)
                    for j in range(num_traj):
                        t_seq, x_ts, x_prevs = trajs_list[j]
                        old_logs_j = old_logs_list[j]
                        step_losses = []
                        kl_sum = torch.zeros((), device=base_device)
                        for s_idx in range(len(t_seq)):
                            t_scalar = t_seq[s_idx]
                            t_batch = torch.full((i_it_py.size(0),), int(t_scalar), device=base_device, dtype=torch.long)
                            x_t = x_ts[s_idx]
                            x_prev = x_prevs[s_idx]
                            eps_cur, _ = self.net.gcn.predict_eps(cnn_feature, i_it_py, c_it_py, py_ind, x_t, t_batch)
                            _, lp_cur, mean_cur, std_cur = ddpm_step_with_logprob(self.net.gcn.scheduler, eps_cur, t_scalar, x_t, prev_sample=x_prev)
                            ratio = torch.exp(lp_cur - old_logs_j[:, s_idx].to(base_device))
                            adv_j = adv_k[j].clamp(-adv_clip_max, adv_clip_max).detach()
                            unclipped = -adv_j * ratio
                            clipped = -adv_j * torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps)
                            step_losses.append(torch.maximum(unclipped, clipped).mean())
                            if self.use_kl and (self.ref_gcn is not None) and self.kl_beta > 0.0:
                                with torch.no_grad():
                                    eps_ref, _ = self.ref_gcn.predict_eps(cnn_feature, i_it_py, c_it_py, py_ind, x_t, t_batch)
                                _, _, mean_ref, _ = ddpm_step_with_logprob(self.net.gcn.scheduler, eps_ref, t_scalar, x_t, prev_sample=x_prev)
                                var = (std_cur ** 2).clamp_min(1e-12)
                                kl_step = ((mean_cur - mean_ref) ** 2) / (2.0 * var)
                                kl_sum = kl_sum + kl_step.mean()
                        pl_terms.append(torch.stack(step_losses).mean())
                        if self.use_kl and self.kl_beta > 0.0:
                            kl_terms.append(kl_sum / max(len(t_seq), 1))
                    if len(pl_terms) > 0:
                        total_policy_loss = torch.stack(pl_terms).mean()
                        if self.use_kl and self.kl_beta > 0.0 and len(kl_terms) > 0:
                            total_policy_loss = total_policy_loss + self.kl_beta * torch.stack(kl_terms).mean()
                        loss = loss + total_policy_loss
                        scalar_stats.update({'grpo_loss': total_policy_loss})
                    else:
                        loss = loss + (cnn_feature.sum() * 0.0)
                        scalar_stats.update({'grpo_loss': torch.tensor(0.0, device=base_device)})
        elif self.flow_posttrain_mode and (not self.freeze_snake) and ('diff_loss' in output):
            diff_weight = float(getattr(cfg, 'diffusion_loss_weight', 1.0))
            loss = loss + diff_weight * output['diff_loss']
            scalar_stats.update({
                'diff_loss': output['diff_loss'],
                'diff_loss_scaled': diff_weight * output['diff_loss'],
                'grpo_loss': torch.tensor(0.0, device=base_device),
            })
            try:
                for k_, v_ in output.items():
                    if isinstance(k_, str) and k_.startswith('diff_loss'):
                        scalar_stats[k_] = v_
            except Exception:
                pass

            if self.enable_reward_logging:
                pred_disp = output.get('pred_disp', None)
                i_init_train_py = output.get('i_it_py', None)
                i_gt_py = output.get('i_gt_py', None)
                feat_hw = output.get('feat_hw', None)
                reward_w = float(getattr(cfg, 'reward_w_region', 1.0))
                reward_w_dice = float(getattr(cfg, 'reward_w_dice', 0.0))
                if pred_disp is None and isinstance(output.get('pred_contours', None), torch.Tensor) and isinstance(i_init_train_py, torch.Tensor):
                    pred_disp = output['pred_contours'] - i_init_train_py
                if (
                    isinstance(pred_disp, torch.Tensor)
                    and isinstance(i_init_train_py, torch.Tensor)
                    and isinstance(i_gt_py, torch.Tensor)
                    and pred_disp.numel() > 0
                ):
                    if isinstance(feat_hw, (tuple, list)) and len(feat_hw) == 2:
                        H, W = int(feat_hw[0]), int(feat_hw[1])
                    else:
                        H = max(int(batch['inp'].shape[-2] // int(snake_config.down_ratio)), 1)
                        W = max(int(batch['inp'].shape[-1] // int(snake_config.down_ratio)), 1)
                    with torch.no_grad():
                        rew = compute_region_reward(i_init_train_py, pred_disp, i_gt_py, H=H, W=W, w1=reward_w, w_dice=reward_w_dice)
                    scalar_stats.update({
                        'grpo_reward_mean': rew.mean(),
                        'grpo_reward_std': rew.std(unbiased=False),
                    })
                else:
                    scalar_stats.update({
                        'grpo_reward_mean': torch.tensor(0.0, device=base_device),
                        'grpo_reward_std': torch.tensor(0.0, device=base_device),
                    })
        elif (not self.freeze_snake) and ('diff_loss' in output):
            diff_weight = float(getattr(cfg, 'diffusion_loss_weight', 1.0))
            loss = loss + diff_weight * output['diff_loss']
            scalar_stats.update({'diff_loss': output['diff_loss'], 'diff_loss_scaled': diff_weight * output['diff_loss']})
            try:
                for k_, v_ in output.items():
                    if isinstance(k_, str) and k_.startswith('diff_loss'):
                        scalar_stats[k_] = v_
            except Exception:
                pass
        else:
            scalar_stats.update({'diff_loss': torch.tensor(0.0, device=base_device)})

        # Combined metric for monitoring on wandb/json: detection + diffusion (or detection + grpo)
        try:
            det_l = scalar_stats.get('det_loss', None)
            diff_l = scalar_stats.get('diff_loss', None)
            grpo_l = scalar_stats.get('grpo_loss', None)
            if isinstance(det_l, torch.Tensor) and isinstance(diff_l, torch.Tensor):
                scalar_stats['det_plus_diff_loss'] = det_l + diff_l
            elif isinstance(det_l, torch.Tensor) and isinstance(grpo_l, torch.Tensor):
                scalar_stats['det_plus_diff_loss'] = det_l + grpo_l
        except Exception:
            pass

        # 3) CMAM 先验（可选）
        if 'L' in output and 'L_star' in output and (not self.freeze_snake):
            L_loss = self.L_crit(output['L'], output['L_star'])
            prior_weight = float(self.loss_scales.get('prior', 1.0))
            loss = loss + prior_weight * L_loss
            scalar_stats.update({'L_loss': L_loss, 'L_loss_scaled': prior_weight * L_loss})
        else:
            scalar_stats.update({'L_loss': torch.tensor(0.0, device=base_device)})

        scalar_stats.update({'loss': loss})
        return output, loss, scalar_stats, image_stats
