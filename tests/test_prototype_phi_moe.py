import torch

from lib.networks.diffusion.dit_blocks_v3 import DiTBlockV3


def _bridge_dense_weights(dense_block, moe_block):
    source = dense_block.state_dict()
    target = moe_block.state_dict()
    compatible = {
        key: value
        for key, value in source.items()
        if key in target and target[key].shape == value.shape
    }
    marker = "prototype_phi_moe.experts."
    for target_key, target_value in target.items():
        if marker not in target_key:
            continue
        prefix, expert_suffix = target_key.split(marker, 1)
        _, tensor_suffix = expert_suffix.split(".", 1)
        source_key = "{}mlp.{}".format(prefix, tensor_suffix)
        if source_key in source and source[source_key].shape == target_value.shape:
            compatible[target_key] = source[source_key]
    moe_block.load_state_dict(compatible, strict=False)


def _make_blocks(top_k):
    torch.manual_seed(7)
    dense = DiTBlockV3(
        dim=64,
        num_heads=4,
        num_points=16,
    )
    torch.manual_seed(19)
    moe = DiTBlockV3(
        dim=64,
        num_heads=4,
        num_points=16,
        use_prototype_phi_moe=True,
        prototype_phi_num_experts=4,
        prototype_phi_top_k=top_k,
    )
    _bridge_dense_weights(dense, moe)
    return dense.eval(), moe.eval()


def test_dense_checkpoint_bridge_preserves_top1_ffn_output():
    dense, moe = _make_blocks(top_k=1)
    x = torch.randn(12, 16, 64)
    expected = dense.mlp(x)
    actual = moe.prototype_phi_moe(x)
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)
    assert moe.mlp is None
    assert not any(name.startswith("mlp.") for name, _ in moe.named_parameters())


def test_dense_checkpoint_bridge_preserves_top2_ffn_output():
    dense, moe = _make_blocks(top_k=2)
    x = torch.randn(12, 16, 64)
    expected = dense.mlp(x)
    actual = moe.prototype_phi_moe(x)
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)


if __name__ == "__main__":
    test_dense_checkpoint_bridge_preserves_top1_ffn_output()
    test_dense_checkpoint_bridge_preserves_top2_ffn_output()
    print("2 synthetic tests passed")
