"""
本文件从 ddpo-pytorch 的带对数概率版本改造而来，并适配到
diffusers 的 FlowMatchEulerDiscreteScheduler（flow matching）接口。

目标
- 在一次 SDE/ODE 反向步进中，既返回上一步样本，又计算该步的样本级对数概率；
- 便于在训练/强化学习中使用逐步的 log-prob 作为奖励或损失项。

数值注意
- bf16 在计算均值/方差时容易溢出，函数内统一转为 float32 计算，返回前与外部保持一致；
- 提供 `sde` 与 `cps` 两种更新形式（cps 对应 cosine-power schedule 近似）。
"""

# Copied from https://github.com/kvablack/ddpo-pytorch/blob/main/ddpo_pytorch/diffusers_patch/ddim_with_logprob.py
# We adapt it from flow to flow matching.

import math
from typing import Optional, Union
import torch

from diffusers.utils.torch_utils import randn_tensor
from diffusers.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler

def sde_step_with_logprob(
    self: FlowMatchEulerDiscreteScheduler,
    model_output: torch.FloatTensor,
    timestep: Union[float, torch.FloatTensor],
    sample: torch.FloatTensor,
    noise_level: float = 0.7,
    prev_sample: Optional[torch.FloatTensor] = None,
    generator: Optional[torch.Generator] = None,
    sde_type: Optional[str] = 'sde',
):
    """
    单步 SDE 反演（带对数概率）。

    功能
    - 根据当前 `timestep` 与模型输出 `model_output`，对 `sample` 做一次反向步进；
    - 若未提供 `prev_sample`，内部按噪声标准差采样得到随机项；
    - 返回：新样本、该步对数概率、该步均值、该步标准差（或其等价形式）。

    参数
    - model_output: 模型直接输出（通常是速度场/噪声预测）。
    - timestep: 当前离散时间步，可为标量或 batch 张量。
    - sample: 当前样本 x_t。
    - noise_level: 噪声强度（仅在 `sde` 形式中生效，控制扩散项）。
    - prev_sample: 若给定，表示已知的 x_{t-1}，将用于对数概率评估；
    - generator: 采样用的随机数发生器（与 prev_sample 互斥）。
    - sde_type: 'sde' 或 'cps'，两种不同的更新/噪声调度近似。

    返回
    - prev_sample: 本步生成的上一时间样本 x_{t-1}
    - log_prob: 本步的样本对数概率（除 batch 外均值聚合）
    - prev_sample_mean: 条件高斯的均值项
    - std_dev_t: 条件高斯的标准差（或近似量）
    """
    # bf16 can overflow here when compute prev_sample_mean, we must convert all variable to fp32
    model_output=model_output.float()
    sample=sample.float()
    if prev_sample is not None:
        prev_sample=prev_sample.float()

    # 注意：scheduler 的 `sigmas` 通常按从大到小排列，反演时步进到更小的噪声。
    # 将标量/张量的时间步转换为索引，并取前一个/后一个步的 sigma。
    step_index = [self.index_for_timestep(t) for t in timestep]
    prev_step_index = [step+1 for step in step_index]
    sigma = self.sigmas[step_index].view(-1, *([1] * (len(sample.shape) - 1)))
    sigma_prev = self.sigmas[prev_step_index].view(-1, *([1] * (len(sample.shape) - 1)))
    sigma_max = self.sigmas[1].item()
    # 由于 sigmas 递减，dt = sigma_prev - sigma 通常为负数（反向积分）
    dt = sigma_prev - sigma

    if sde_type == 'sde':
        # 扩散强度：根据当前 sigma 与最大 sigma 构造，乘以 noise_level 控制注入噪声
        std_dev_t = torch.sqrt(sigma / (1 - torch.where(sigma == 1, sigma_max, sigma)))*noise_level

        # 反向 SDE 的均值项推导（保持与外部实现一致）
        prev_sample_mean = sample*(1+std_dev_t**2/(2*sigma)*dt)+model_output*(1+std_dev_t**2*(1-sigma)/(2*sigma))*dt

        if prev_sample is None:
            # 当未给定上一样本时，从条件高斯中采样
            variance_noise = randn_tensor(
                model_output.shape,
                generator=generator,
                device=model_output.device,
                dtype=model_output.dtype,
            )
            prev_sample = prev_sample_mean + std_dev_t * torch.sqrt(-1*dt) * variance_noise

        # 条件高斯密度的对数（按像素/通道独立），后续在 batch 外维度上取均值
        log_prob = (
            -((prev_sample.detach() - prev_sample_mean) ** 2) / (2 * ((std_dev_t * torch.sqrt(-1*dt))**2))
            - torch.log(std_dev_t * torch.sqrt(-1*dt))
            - torch.log(torch.sqrt(2 * torch.as_tensor(math.pi)))
        )
    
    elif sde_type == 'cps':
        # cosine-power schedule 近似：以论文里的 sigma_t 与 x0/x1 近似组合
        std_dev_t = sigma_prev  * math.sin(noise_level * math.pi / 2) # sigma_t in paper
        pred_original_sample = sample - sigma * model_output # predicted x_0 in paper
        noise_estimate = sample + model_output * (1 - sigma) # predicted x_1 in paper
        prev_sample_mean = pred_original_sample * (1 - sigma_prev) + noise_estimate * torch.sqrt(sigma_prev**2 - std_dev_t**2)

        if prev_sample is None:
            # 从近似条件高斯中采样
            variance_noise = randn_tensor(
                model_output.shape,
                generator=generator,
                device=model_output.device,
                dtype=model_output.dtype,
            )
            prev_sample = prev_sample_mean + std_dev_t * variance_noise

        # 仅保留与样本相关的二次型项（去除常数项）
        log_prob = -((prev_sample.detach() - prev_sample_mean) ** 2)

    # 沿除 batch 维之外的所有维度取均值，得到每个样本的标量 log-prob
    log_prob = log_prob.mean(dim=tuple(range(1, log_prob.ndim)))
    
    return prev_sample, log_prob, prev_sample_mean, std_dev_t
