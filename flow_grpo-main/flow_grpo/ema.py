"""
参数指数滑动平均（EMA）工具（中文注释版）。

用途
- 在训练过程中维护一份模型参数的指数滑动平均副本，常用于评估时提升稳定性与性能；
- 提供便捷的参数交换接口：将 EMA 参数拷贝到模型进行评估，再恢复原训练参数继续训练；
- 支持按步数间隔更新、在不同设备/精度上保存 EMA 参数，减少显存占用。

注意
- `get_current_decay` 使用一个随步数升高的暖启动衰减，并与设定的 `decay` 取最小值；
- 当 `update_step_interval > 1` 时，仅在符合间隔的步数执行 EMA 更新；
- `copy_ema_to` 可选择性地把当前训练参数暂存到 CPU 以节省显存，`copy_temp_to` 用于恢复。
"""

from collections.abc import Iterable

import torch


class EMAModuleWrapper:
    """
    维护一组参数的 EMA 副本。

    参数
    - parameters: 需要做 EMA 的参数可迭代对象（如 `model.parameters()`）。
    - decay: EMA 衰减系数，越接近 1 代表平均越“慢”（历史占比更高）。
    - update_step_interval: 每多少个优化步更新一次 EMA（>=1）。
    - device: 将 EMA 参数存放在哪个设备上（可设为 CPU 以节省显存）。
    """
    def __init__(
            self,
            parameters: Iterable[torch.nn.Parameter],
            decay: float = 0.9999,
            update_step_interval: int = 1,
            device: torch.device | None = None,
    ):
        parameters = list(parameters)
        self.ema_parameters = [p.clone().detach().to(device) for p in parameters]

        self.temp_stored_parameters = None

        self.decay = decay
        self.update_step_interval = update_step_interval
        self.device = device

        # TODO: add an automatic decay calculation based on this formula:
        # The impact of the last n steps can be calculated as:
        #     impact = 1-(decay^n)
        # The number of steps needed to reach a specific impact is:
        #     n = log_decay(1-impact)
        # The decay needed to reach a specific impact after n steps is:
        #     decay = (1-impact)^(1/n)

    def get_current_decay(self, optimization_step) -> float:
        """
        计算当前步使用的衰减值：min((1+step)/(10+step), decay)。

        - 早期步数采用较小的衰减（更快跟随当前参数，暖启动）；
        - 随步数增长逐渐逼近 1，并被上限 `decay` 截断。
        """
        return min(
            (1 + optimization_step) / (10 + optimization_step),
            self.decay
        )

    @torch.no_grad()
    def step(self, parameters: Iterable[torch.nn.Parameter], optimization_step):
        """
        执行一次 EMA 更新。

        - 当 `(optimization_step + 1) % update_step_interval == 0` 时更新；
        - 公式：ema = ema + (1 - decay) * (param - ema)；
        - 若设备不同，使用临时张量在 EMA 所在设备上进行就地计算，降低显存峰值。
        """
        parameters = list(parameters)

        one_minus_decay = 1 - self.get_current_decay(optimization_step)

        if (optimization_step + 1) % self.update_step_interval == 0:
            for ema_parameter, parameter in zip(self.ema_parameters, parameters, strict=True):
                if parameter.requires_grad:
                    if ema_parameter.device == parameter.device:
                        ema_parameter.add_(one_minus_decay * (parameter - ema_parameter))
                    else:
                        # in place calculations to save memory
                        parameter_copy = parameter.detach().to(ema_parameter.device)
                        parameter_copy.sub_(ema_parameter)
                        parameter_copy.mul_(one_minus_decay)
                        ema_parameter.add_(parameter_copy)
                        del parameter_copy

    def to(self, device: torch.device = None, dtype: torch.dtype = None) -> None:
        """
        将 EMA 参数移动到指定设备/精度。非浮点参数仅移动设备不改 dtype。
        """
        self.device = device
        self.ema_parameters = [
            p.to(device=device, dtype=dtype) if p.is_floating_point() else p.to(device=device)
            for p in self.ema_parameters
        ]

    def copy_ema_to(self, parameters: Iterable[torch.nn.Parameter], store_temp: bool = True) -> None:
        """
        将 EMA 参数拷贝到传入参数（通常是模型参数）上。

        - store_temp=True 时，会先把原训练参数浅拷贝到 CPU 暂存，便于评估后恢复；
        - 典型用法：评估前 `copy_ema_to(model.parameters())`，评估后 `copy_temp_to(...)`。
        """
        if store_temp:
            self.temp_stored_parameters = [parameter.detach().cpu() for parameter in parameters]

        parameters = list(parameters)
        for ema_parameter, parameter in zip(self.ema_parameters, parameters, strict=True):
            parameter.data.copy_(ema_parameter.to(parameter.device).data)

    def copy_temp_to(self, parameters: Iterable[torch.nn.Parameter]) -> None:
        """
        将先前暂存的原训练参数恢复到模型上，并清空暂存。
        """
        for temp_parameter, parameter in zip(self.temp_stored_parameters, parameters, strict=True):
            parameter.data.copy_(temp_parameter.data)

        self.temp_stored_parameters = None

    def load_state_dict(self, state_dict: dict) -> None:
        """
        从 state_dict 加载 EMA 状态（decay 与 ema_parameters），并移动到当前 device。
        """
        self.decay = self.decay if self.decay else state_dict.get("decay", self.decay)
        self.ema_parameters = state_dict.get("ema_parameters")
        self.to(self.device)

    def state_dict(self) -> dict:
        """
        导出 EMA 状态字典，包含 `decay` 与 `ema_parameters`。
        """
        return {
            "decay": self.decay,
            "ema_parameters": self.ema_parameters,
        }
