"""Flow-matching contour evolution used by the official mainline."""

from .flow_matching_evolution import FlowMatchingEvolution

__all__ = ("FlowMatchingEvolution", "make_evolution")


def make_evolution(**kwargs):
    """Build the only evolution backend supported by this package."""
    if not kwargs.get("use_flow_matching", False):
        raise ValueError("the mainline package requires use_flow_matching=true")
    return FlowMatchingEvolution(
        state_dim=kwargs.get("state_dim", 128),
        feature_dim=kwargs.get("feature_dim", 64),
        num_points=kwargs.get("num_points", 128),
        loss_weight=kwargs.get("loss_weight", 1.0),
        loss_type=kwargs.get("loss_type", "adaptive"),
        dit_num_layers=kwargs.get("dit_num_layers", 6),
        dit_num_heads=kwargs.get("dit_num_heads", 8),
        dit_state_dim=kwargs.get("dit_state_dim", 256),
        ode_steps=kwargs.get("flow_ode_steps", 10),
    )
