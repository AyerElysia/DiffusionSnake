import torch  # 张量与数值运算
import random  # 采样窗口的随机起点
from typing import Tuple, List, Optional  # 类型提示
from .ddpm_with_logprob import ddpm_step_with_logprob  # 单步DDPM带log-prob的过渡


class GRPOSampler:
    """
    Decoupled GRPO sampler for diffusion-based policies.

    Construct with a DDPM/DDIM scheduler and a predict_eps_fn(cnn_feature, i_it_py, c_it_py, py_ind, x_t, t_batch)
    that returns (eps_pred, L).
    """
    def __init__(self, scheduler, predict_eps_fn):
        self.scheduler = scheduler  # 调度器（DDPM/DDIM），提供时间步与相关参数
        self.predict_eps_fn = predict_eps_fn  # 噪声预测函数：返回 (eps_pred, L)

    @torch.no_grad()
    def sample_with_logprob(
        self,
        cnn_feature: torch.Tensor,  # CNN特征（条件）
        i_it_py: torch.Tensor,  # 初始点位（N,P,2）
        c_it_py: torch.Tensor,  # 额外条件（如分类/上下文）
        py_ind: torch.Tensor,  # 索引/批次中的样本映射
        steps: int = 50,  # 采样步数
        window_size: int = 0,  # 滑动窗口大小（0表示全程）
        window_range: Tuple[int, int] = (0, 0),  # 窗口起止范围限制
        generator: Optional[torch.Generator] = None,  # 随机数生成器
    ):
        N, P, _ = i_it_py.shape  # 批大小与点数
        device = i_it_py.device  # 设备
        x = torch.randn(N, P, 2, device=device)  # 初始噪声状态 x_T
        self.scheduler.set_timesteps(steps, device=device)  # 设置调度器的时间步序列
        timesteps = self.scheduler.timesteps  # 时间步张量/列表（通常为降序）
        Tn = len(timesteps)  # 时间步数量

        # 选择采样窗口 [s, e)，用于截取局部轨迹与log_prob
        if window_size and window_size > 0:
            start_min = int(window_range[0]) if isinstance(window_range, (tuple, list)) and len(window_range) > 0 else 0  # 起点下界
            end_max = int(window_range[1]) if isinstance(window_range, (tuple, list)) and len(window_range) > 1 else Tn  # 终点上界
            end_max = max(end_max, window_size + 1)  # 保证可容纳窗口
            if end_max <= start_min + window_size:
                # 当提供的范围过窄（如 window_range=(10,11), window_size=1）时，直接使用 start_min
                s = max(0, min(start_min, max(Tn - window_size, 0)))
            else:
                s = random.randint(start_min, end_max - window_size)  # 随机选择窗口起点
            e = min(s + window_size, Tn)  # 窗口终点（半开区间）
        else:
            s = 0  # 无窗口时，从起点开始
            e = max(0, Tn - 1)  # 到倒数第二个时间步（与下述逻辑一致）

        latents_seq = []  # 存储窗口内的潜变量序列（x 和 x_prev）
        log_probs = []  # 存储每步的log概率（窗口内）
        t_seq = []  # 存储对应的时间步（窗口内）
        x_ts = []  # 存储状态 x_t（窗口内）
        x_prevs = []  # 存储动作/转移后的 x_{t-1}（窗口内）

        for idx, t in enumerate(timesteps):  # 逐时间步迭代
            t_scalar = t if torch.is_tensor(t) else torch.tensor(int(t), device=device, dtype=torch.long)  # 标量时间步
            t_batch = torch.full((N,), int(t_scalar.item()), device=device, dtype=torch.long)  # 扩展到批次
            eps_pred, _ = self.predict_eps_fn(cnn_feature, i_it_py, c_it_py, py_ind, x, t_batch)  # 预测噪声 ε
            if idx == s:
                latents_seq.append(x)  # 进入窗口时，先记录当前 x_t
            x_prev, log_prob, _, _ = ddpm_step_with_logprob(self.scheduler, eps_pred, t_scalar, x, generator=generator, prev_sample=None)  # 进行一次DDPM步并得到log_prob
            if idx >= s and idx < e:  # 窗口范围内收集轨迹与分数
                latents_seq.append(x_prev)  # 记录转移后的 x_{t-1}
                log_probs.append(log_prob)  # 记录对应log概率
                t_seq.append(t_scalar if torch.is_tensor(t_scalar) else torch.tensor(int(t_scalar), device=device))  # 记录时间步
                # store state-action for this step in window
                x_ts.append(x)  # 记录状态 x_t
                x_prevs.append(x_prev)  # 记录下一状态 x_{t-1}
            x = x_prev  # 更新当前状态

        disp = x  # 最终位移（从初始点位到采样终点）
        py = i_it_py + disp  # 应用位移得到最终预测点位
        return {
            'latents': latents_seq,  # 窗口内的潜变量序列
            'log_probs': log_probs,  # 窗口内每步log概率
            'timesteps': t_seq,  # 对应的时间步
            'x_ts': x_ts,  # 窗口内状态 x_t
            'x_prevs': x_prevs,  # 窗口内状态 x_{t-1}
            'disp': disp,  # 末端位移
            'py': py,  # 末端预测点位
        }
