from lib.utils.snake import snake_config
from .ct_snake import get_network as get_ro
from .pretrain_evolution import DiffusionEvolution
from .grpo_evolution import GRPOEvolution
from .dit_denoiser import DiTDenoiser
from .dit_denoiser_v2 import DiTDenoiserV2
from .dit_denoiser_v2_2 import DiTDenoiserV2_2

_network_factory = {
    'ro': get_ro
}

def make_evolution(use_grpo=False, **kwargs):
    """Factory function to create the appropriate evolution module.
    
    Args:
        use_grpo: Whether to use GRPO training mode
        **kwargs: Additional arguments to pass to the evolution module
        
    Returns:
        An instance of Evolution module
    """
    if kwargs.get('use_flow_matching', False) or kwargs.get('use_dit_v3_6', False):
        from .flow_matching_evolution import FlowMatchingEvolution
        return FlowMatchingEvolution(
            state_dim=kwargs.get('state_dim', 128),
            feature_dim=kwargs.get('feature_dim', 64),
            num_points=kwargs.get('num_points', 128),
            loss_weight=kwargs.get('loss_weight', 1.0),
            loss_type=kwargs.get('loss_type', 'adaptive'),
            dit_num_layers=kwargs.get('dit_num_layers', 6),
            dit_num_heads=kwargs.get('dit_num_heads', 8),
            dit_state_dim=kwargs.get('dit_state_dim', 256),
            ode_steps=kwargs.get('flow_ode_steps', 10)
        )
    if use_grpo:
        return GRPOEvolution(**kwargs)
    return DiffusionEvolution(**kwargs)

def get_network(cfg):
    arch = cfg.network
    heads = cfg.heads
    head_conv = cfg.head_conv
    num_layers = int(arch[arch.find('_') + 1:]) if '_' in arch else 0
    arch = arch[:arch.find('_')] if '_' in arch else arch
    get_model = _network_factory[arch]
    network = get_model(num_layers, heads, head_conv, snake_config.down_ratio, cfg.det_dir)
    return network
