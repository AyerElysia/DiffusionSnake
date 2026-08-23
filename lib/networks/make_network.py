"""Build the single network shipped by the mainline package."""

from lib.networks.mainline import get_network


def make_network(cfg):
    """Construct MoonViT-cache + Flow without importing legacy backends."""
    architecture = str(cfg.network)
    num_layers = int(architecture.rsplit("_", 1)[-1]) if "_" in architecture else 0
    return get_network(
        num_layers=num_layers,
        heads=dict(cfg.heads),
        head_conv=int(cfg.head_conv),
        down_ratio=int(getattr(cfg, "down_ratio", 4)),
        det_dir=str(getattr(cfg, "det_dir", "")),
    )
