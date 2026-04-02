import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import DDPMScheduler
from typing import Tuple, Optional, Dict, Any, List
import random
import os

from .pretrain_evolution import DiffusionEvolution
from .ddpm_with_logprob import ddpm_step_with_logprob

class GRPOEvolution(DiffusionEvolution):
    """
    GRPO训练专用的扩散模型实现
    继承自预训练模型，添加GRPO特定逻辑
    """
    def __init__(
        self,
        state_dim: int = 128,
        feature_dim: int = 64,
        num_points: int = 128,
        num_timesteps: int = 1000,
        use_ddim_inference: bool = True,
        loss_weight: float = 1.0,
        loss_type: str = 'adaptive',
        gamma: float = 0.99,
        lam: float = 0.95,
        **kwargs
    ):
        super().__init__(
            state_dim=state_dim,
            feature_dim=feature_dim,
            num_points=num_points,
            num_timesteps=num_timesteps,
            use_ddim_inference=use_ddim_inference,
            loss_weight=loss_weight,
            loss_type=loss_type,
            **kwargs,
        )
        
        # GRPO 特定参数
        self.gamma = gamma
        self.lam = lam
        self.enable_grpo = True

    def sample_with_logprob(
        self,
        cnn_feature: torch.Tensor,
        i_it_py: torch.Tensor,
        c_it_py: torch.Tensor,
        py_ind: torch.Tensor,
        steps: int = 50,
        window_size: int = 0,
        window_range: Tuple[int, int] = (0, 0),
        generator: Optional[torch.Generator] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        GRPO 专用的带 logprob 的采样
        """
        N, P, _ = i_it_py.shape
        device = i_it_py.device
        x = torch.randn(N, P, 2, device=device)
        
        self.scheduler.set_timesteps(steps, device=device)
        timesteps = self.scheduler.timesteps
        Tn = len(timesteps)

        # 选择采样窗口 [s, e)
        if window_size and window_size > 0:
            start_min = int(window_range[0]) if isinstance(window_range, (tuple, list)) and len(window_range) > 0 else 0
            end_max = int(window_range[1]) if isinstance(window_range, (tuple, list)) and len(window_range) > 1 else Tn
            end_max = max(end_max, window_size + 1)
            
            if end_max <= start_min + window_size:
                s = max(0, Tn - window_size - 1)
            else:
                s = random.randint(start_min, end_max - window_size)
            e = min(s + window_size, Tn)
        else:
            s = 0
            e = max(0, Tn - 1)

        latents_seq = []  # 存储窗口内的潜变量序列
        log_probs = []    # 存储每步的log概率
        t_seq = []        # 存储对应的时间步
        x_ts = []         # 存储状态 x_t
        x_prevs = []      # 存储动作/转移后的 x_{t-1}

        for idx, t in enumerate(timesteps):
            t_scalar = t if torch.is_tensor(t) else torch.tensor(int(t), device=device, dtype=torch.long)
            t_batch = torch.full((N,), int(t_scalar.item()), device=device, dtype=torch.long)
            
            # 预测噪声
            eps_pred, _ = self.predict_eps(cnn_feature, i_it_py, c_it_py, py_ind, x, t_batch)
            
            # 记录进入窗口时的状态
            if idx == s:
                latents_seq.append(x)
                
            # 执行一步DDPM并获取log概率
            x_prev, log_prob, _, _ = ddpm_step_with_logprob(
                self.scheduler, eps_pred, t_scalar, x, 
                generator=generator, prev_sample=None
            )
            
            # 如果在窗口内，记录轨迹和概率
            if idx >= s and idx < e:
                latents_seq.append(x_prev)
                log_probs.append(log_prob)
                t_seq.append(t_scalar if torch.is_tensor(t_scalar) else torch.tensor(int(t_scalar), device=device))
                x_ts.append(x)
                x_prevs.append(x_prev)
                
            x = x_prev  # 更新状态

        disp = self.denormalize_disp(x)
        py = i_it_py + disp
        
        return {
            'latents': latents_seq,  # 窗口内的潜变量序列
            'log_probs': log_probs,  # 窗口内每步log概率
            'timesteps': t_seq,      # 对应的时间步
            'x_ts': x_ts,           # 窗口内状态 x_t
            'x_prevs': x_prevs,     # 窗口内状态 x_{t-1}
            'disp': disp,           # 末端位移(像素尺度)
            'py': py,               # 末端预测点位
        }

    def compute_advantages(
        self, 
        rewards: List[torch.Tensor], 
        values: List[torch.Tensor], 
        last_value: torch.Tensor,
        dones: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        计算GAE优势函数
        """
        advantages = []
        gae = 0
        next_value = last_value
        
        # 反向计算GAE
        for step in reversed(range(len(rewards))):
            if step == len(rewards) - 1:
                next_non_terminal = 1.0 - dones[step].float()
                next_value = last_value
            else:
                next_non_terminal = 1.0 - dones[step + 1].float()
                next_value = values[step + 1]
                
            delta = rewards[step] + self.gamma * next_value * next_non_terminal - values[step]
            gae = delta + self.gamma * self.lam * next_non_terminal * gae
            advantages.insert(0, gae)
            
        # 计算returns
        returns = torch.stack(advantages) + torch.stack(values)
        advantages = torch.stack(advantages)
        
        # 标准化优势
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return advantages, returns

    def update_policy(self, samples: Dict[str, List[torch.Tensor]], optimizer: torch.optim.Optimizer, clip_eps: float = 0.2):
        """
        GRPO策略更新
        """
        # 提取样本
        old_log_probs = torch.stack(samples['log_probs']).detach()
        states = samples['x_ts']
        actions = samples['x_prevs']
        advantages = samples['advantages']
        returns = samples['returns']
        
        # 计算新的动作概率
        new_log_probs = []
        for t in range(len(states)):
            # 这里需要根据你的模型结构实现计算新的log概率
            # 这只是一个示例，实际实现需要根据你的模型结构来调整
            _, log_prob, _, _ = ddpm_step_with_logprob(
                self.scheduler, 
                model_output=states[t],  # 这里需要替换为你的模型输出
                timestep=t,              # 需要提供时间步
                sample=actions[t],
                prev_sample=states[t] if t > 0 else None
            )
            new_log_probs.append(log_prob)
            
        new_log_probs = torch.stack(new_log_probs)
        
        # 计算概率比
        ratio = (new_log_probs - old_log_probs).exp()
        
        # 裁剪目标
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # 价值函数损失
        # 这里需要根据你的价值函数实现来计算
        value_loss = F.mse_loss(torch.stack(returns), torch.stack(samples['values']))
        
        # 总损失
        loss = policy_loss + 0.5 * value_loss
        
        # 更新参数
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 0.5)
        optimizer.step()
        
        return {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'total_loss': loss.item(),
            'approx_kl': (old_log_probs - new_log_probs).mean().item()
        }
