import math
from typing import Optional, Tuple  # 可选类型与元组类型提示
import torch  # PyTorch 张量与数值计算库


def _get_prev_timestep(scheduler, timestep: int) -> int:
    # 尝试从调度器中获取时间步序列，并找到当前时间步的“上一个”时间步（按调度器顺序）
    try:
        ts = scheduler.timesteps  # 期望为时间步列表或张量，按采样顺序排列
        if isinstance(ts, torch.Tensor):
            tl = ts.tolist()  # 张量转为 Python 列表
        else:
            tl = list(ts)  # 确保为列表
        if timestep in tl:  # 如果当前时间步在列表中
            i = tl.index(int(timestep))  # 找到索引
            if i + 1 < len(tl):
                return int(tl[i + 1])  # 返回序列中的下一个元素，表示“上一个”真实时间（因序列多为降序）
            return 0  # 否则到头了，返回 0
    except Exception:
        pass  # 若上述流程失败，则进入兜底策略
    return max(int(timestep) - 1, 0)  # 兜底：简单地返回 t-1，且不小于 0


def ddpm_step_with_logprob(
    scheduler,
    model_output: torch.FloatTensor,
    timestep: torch.Tensor,
    sample: torch.FloatTensor,
    generator: Optional[torch.Generator] = None,
    prev_sample: Optional[torch.FloatTensor] = None,
) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
    """
    One DDPM-like step with log-prob under a Gaussian transition.
    Returns: (prev_sample, log_prob, prev_mean, std)
    """
    model_output = model_output.float()  # 确保模型输出为 float32 精度
    sample = sample.float()  # 当前样本 xt 转为 float32

    # 解析标量时间步 t
    if isinstance(timestep, torch.Tensor):
        t_val = int(timestep.flatten()[0].item())  # 支持张量形式的时间步
    else:
        t_val = int(timestep)  # 直接为 int 的情况

    device = sample.device  # 设备信息（CPU/GPU）
    dtype = sample.dtype  # 目标返回的数据类型（保持一致）

    # 获取累计 alpha（alpha_bar），与前一时刻 alpha_bar_prev
    try:
        alpha_bar_all = scheduler.alphas_cumprod
        if isinstance(alpha_bar_all, torch.Tensor):
            alpha_bar_t = alpha_bar_all[t_val].to(device=device, dtype=torch.float32)
        else:
            alpha_bar_t = torch.tensor(alpha_bar_all[t_val], device=device, dtype=torch.float32)
    except Exception:
        raise RuntimeError("DDPM scheduler missing alphas_cumprod")

    prev_t = _get_prev_timestep(scheduler, t_val)
    if isinstance(scheduler.alphas_cumprod, torch.Tensor):
        alpha_bar_prev = scheduler.alphas_cumprod[prev_t].to(device=device, dtype=torch.float32)
    else:
        alpha_bar_prev = torch.tensor(scheduler.alphas_cumprod[prev_t], device=device, dtype=torch.float32)

    # 由累计项恢复单步 alpha_t 与 beta_t
    eps_ = 1e-12
    alpha_t = torch.clamp(alpha_bar_t / torch.clamp(alpha_bar_prev, min=eps_), min=eps_, max=1.0)
    beta_t = torch.clamp(1.0 - alpha_t, min=eps_)

    # 预测 x0（epsilon 参数化）：x0 = (xt - sqrt(1-ab_t)*eps) / sqrt(ab_t)
    sqrt_ab_t = torch.sqrt(torch.clamp(alpha_bar_t, min=eps_))
    sqrt_one_minus_ab_t = torch.sqrt(torch.clamp(1.0 - alpha_bar_t, min=eps_))
    pred_x0 = (sample - sqrt_one_minus_ab_t * model_output) / torch.clamp(sqrt_ab_t, min=eps_)

    # 标准 DDPM 后验：q(x_{t-1} | x_t, x_0) = N(mean, var I)
    coef_x0 = torch.sqrt(torch.clamp(alpha_bar_prev, min=eps_)) * (beta_t / torch.clamp(1.0 - alpha_bar_t, min=eps_))
    coef_xt = torch.sqrt(torch.clamp(alpha_t, min=eps_)) * ((1.0 - alpha_bar_prev) / torch.clamp(1.0 - alpha_bar_t, min=eps_))
    prev_mean = coef_x0 * pred_x0 + coef_xt * sample
    posterior_variance = ((1.0 - alpha_bar_prev) / torch.clamp(1.0 - alpha_bar_t, min=eps_)) * beta_t
    std = torch.sqrt(torch.clamp(posterior_variance, min=1e-12))

    # 采样上一时刻：t=0 时使用确定性（无噪声）
    if prev_sample is None:
        if t_val == 0:
            prev_sample = prev_mean
        else:
            try:
                noise = torch.randn(sample.shape, device=sample.device, dtype=torch.float32, generator=generator)
            except TypeError:
                noise = torch.randn(sample.shape, device=sample.device, dtype=torch.float32)
            prev_sample = prev_mean + std * noise
    else:
        prev_sample = prev_sample.float()

    # 在 N(prev_mean, std^2 I) 下的对数概率
    var = torch.clamp(std ** 2, min=1e-12)
    log_prob = -((prev_sample.detach() - prev_mean) ** 2) / (2.0 * var) - torch.log(torch.clamp(std, min=1e-12)) - 0.5 * math.log(2.0 * math.pi)
    reduce_dims = tuple(range(1, log_prob.ndim))
    log_prob = log_prob.mean(dim=reduce_dims)

    return prev_sample.to(dtype), log_prob.to(dtype), prev_mean.to(dtype), std.to(dtype)