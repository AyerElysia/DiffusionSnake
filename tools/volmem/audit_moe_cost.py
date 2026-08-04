#!/usr/bin/env python3
"""Report exact MemFlowDiT parameter storage and routed expert usage."""

import argparse
import json
import os
import pathlib
import sys


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg_file", required=True)
    parser.add_argument("--output", default="")
    return parser.parse_args()


ARGS = parse_args()
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.environ["CFG_FILE"] = ARGS.cfg_file
sys.argv = [sys.argv[0], "--cfg_file", ARGS.cfg_file]

from lib.config import cfg
from lib.networks import make_network
from lib.networks.diffusion.dit_denoiser_v4 import MoEFinalHead
from lib.networks.diffusion.prototype_phi_moe import PrototypePhiMoE
from lib.train.trainers.make_trainer import _wrapper_factory
from volmem.adapters import V46cContourAdapter, configure_single_slice_compatibility
from volmem.models import MemFlowDiTSnake


def count_parameters(module, trainable_only=False):
    return sum(
        parameter.numel()
        for parameter in module.parameters()
        if not trainable_only or parameter.requires_grad
    )


def parameter_bytes(module):
    return sum(parameter.numel() * parameter.element_size() for parameter in module.parameters())


def human_count(value):
    return round(float(value) / 1_000_000.0, 6)


def build_model():
    configure_single_slice_compatibility(cfg)
    base_network = make_network(cfg)
    slice_wrapper = _wrapper_factory(cfg, base_network)
    contour_adapter = V46cContourAdapter(slice_wrapper)
    return base_network, MemFlowDiTSnake(
        contour_adapter=contour_adapter,
        feature_dim=int(cfg.locate_feat_dim),
        memory_dim=int(cfg.volmem.memory_dim),
        memory_capacity=int(cfg.volmem.memory_capacity),
        memory_heads=int(cfg.volmem.memory_heads),
        mask_channels=int(cfg.volmem.mask_channels),
        memory_pool_size=int(cfg.volmem.memory_pool_size),
        dit_state_dim=int(cfg.dit_state_dim),
        distance_scale=float(cfg.volmem.relative_distance_scale),
        distance_mode=str(getattr(cfg.volmem, "relative_distance_mode", "signed")),
        memory_mask_fusion_mode=str(
            getattr(cfg.volmem, "memory_mask_fusion_mode", "concat")
        ),
        memory_mask_evidence_scale=float(
            getattr(cfg.volmem, "memory_mask_evidence_scale", 0.25)
        ),
        memory_position_in_values=bool(
            getattr(cfg.volmem, "memory_position_in_values", True)
        ),
        memory_global_pool_size=int(
            getattr(cfg.volmem, "memory_global_pool_size", 0)
        ),
    )


def module_report(model):
    reports = []
    conditional_inactive = 0
    for name, module in model.named_modules():
        if isinstance(module, PrototypePhiMoE):
            expert_counts = [count_parameters(expert) for expert in module.experts]
            per_expert = expert_counts[0] if expert_counts else 0
            active_experts_per_route = int(module.top_k)
            inactive = sum(expert_counts) - per_expert * active_experts_per_route
            conditional_inactive += inactive
            reports.append(
                {
                    "name": name,
                    "type": type(module).__name__,
                    "num_experts": int(module.num_experts),
                    "top_k": int(module.top_k),
                    "total_params": count_parameters(module),
                    "expert_params": sum(expert_counts),
                    "per_expert_params": per_expert,
                    "params_used_per_route": count_parameters(module) - inactive,
                    "implementation": "selected experts only; token compute is Top-K sparse",
                }
            )
        elif isinstance(module, MoEFinalHead):
            expert_params = sum(
                parameter.numel()
                for parameter_name, parameter in module.named_parameters(recurse=False)
                if parameter_name.startswith("expert_")
            )
            reports.append(
                {
                    "name": name,
                    "type": type(module).__name__,
                    "num_experts": int(module.num_experts),
                    "router_num_experts": int(module.router_num_experts),
                    "top_k": int(module.top_k),
                    "total_params": count_parameters(module),
                    "expert_params": expert_params,
                    "params_evaluated_per_route": count_parameters(module),
                    "implementation": (
                        "all routed experts are evaluated by dense einsum before Top-K gather"
                    ),
                }
            )
    return reports, conditional_inactive


base_network, model = build_model()
modules, conditional_inactive = module_report(model)
total = count_parameters(model)
payload = {
    "config": ARGS.cfg_file,
    "model": str(cfg.model),
    "total_params": total,
    "trainable_params": count_parameters(model, trainable_only=True),
    "total_params_million": human_count(total),
    "base_network_params": count_parameters(base_network),
    "base_network_params_million": human_count(count_parameters(base_network)),
    "memory_wrapper_extra_params": total - count_parameters(base_network),
    "parameter_storage_mib_current_dtype": parameter_bytes(model) / (1024.0 ** 2),
    "fp32_parameter_storage_mib": total * 4 / (1024.0 ** 2),
    "per_route_conditional_params": total - conditional_inactive,
    "per_route_conditional_params_million": human_count(total - conditional_inactive),
    "conditional_inactive_expert_params": conditional_inactive,
    "moe_modules": modules,
}

text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
if ARGS.output:
    output = pathlib.Path(ARGS.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8")
print(text, end="")
