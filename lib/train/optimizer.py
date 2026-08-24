"""Build the single optimizer used by supervised mainline training."""

import torch


def make_optimizer(cfg, net):
    optimizer_name = str(cfg.train.optim).strip().lower()
    if optimizer_name != "adamw":
        raise ValueError(
            f"the mainline supports only train.optim='adamw'; got {optimizer_name!r}"
        )
    learning_rate = float(cfg.train.lr)
    weight_decay = float(cfg.train.weight_decay)
    parameter_groups = [
        {
            "params": [parameter],
            "lr": learning_rate,
            "weight_decay": weight_decay,
        }
        for parameter in net.parameters()
        if parameter.requires_grad
    ]
    if not parameter_groups:
        raise ValueError("no trainable parameters were selected for AdamW")
    return torch.optim.AdamW(
        parameter_groups,
        lr=learning_rate,
        weight_decay=weight_decay,
    )
