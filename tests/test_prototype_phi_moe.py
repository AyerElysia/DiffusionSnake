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
    shared_marker = "prototype_phi_moe.shared_expert."
    shared_hidden = 0
    for target_key, target_value in target.items():
        if shared_marker not in target_key:
            continue
        tensor_suffix = target_key.split(shared_marker, 1)[1]
        source_key = "mlp.{}".format(tensor_suffix)
        bridged = torch.zeros_like(target_value)
        if tensor_suffix in ("w1.weight", "v.weight"):
            shared_hidden = min(target_value.size(0), source[source_key].size(0))
            bridged[:shared_hidden].copy_(source[source_key][:shared_hidden])
        elif tensor_suffix == "w2.weight":
            shared_hidden = min(target_value.size(1), source[source_key].size(1))
            bridged[:, :shared_hidden].copy_(source[source_key][:, :shared_hidden])
        compatible[target_key] = bridged
    for target_key, target_value in target.items():
        if marker not in target_key:
            continue
        prefix, expert_suffix = target_key.split(marker, 1)
        _, tensor_suffix = expert_suffix.split(".", 1)
        source_key = "{}mlp.{}".format(prefix, tensor_suffix)
        if source_key in source and source[source_key].shape == target_value.shape:
            compatible[target_key] = source[source_key]
        elif source_key in source and shared_hidden > 0:
            bridged = torch.zeros_like(target_value)
            if tensor_suffix in ("w1.weight", "v.weight"):
                count = min(target_value.size(0), source[source_key].size(0) - shared_hidden)
                bridged[:count].copy_(source[source_key][shared_hidden:shared_hidden + count])
            elif tensor_suffix == "w2.weight":
                count = min(target_value.size(1), source[source_key].size(1) - shared_hidden)
                bridged[:, :count].copy_(source[source_key][:, shared_hidden:shared_hidden + count])
            compatible[target_key] = bridged
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


def _make_shared_blocks(top_k=1):
    torch.manual_seed(7)
    dense = DiTBlockV3(dim=64, num_heads=4, num_points=16)
    torch.manual_seed(19)
    moe = DiTBlockV3(
        dim=64,
        num_heads=4,
        num_points=16,
        use_prototype_phi_moe=True,
        prototype_phi_num_experts=4,
        prototype_phi_top_k=top_k,
        prototype_phi_use_shared_expert=True,
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


def test_shared_dense_checkpoint_bridge_preserves_top1_ffn_output():
    dense, moe = _make_shared_blocks(top_k=1)
    x = torch.randn(12, 16, 64)
    expected = dense.mlp(x)
    actual = moe.prototype_phi_moe(x)
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)
    assert moe.prototype_phi_moe.shared_expert is not None
    assert moe.prototype_phi_moe.shared_hidden_dim == 128
    assert moe.prototype_phi_moe.expert_hidden_dim == 128


def test_shared_variant_reduces_stored_moe_parameters():
    _, routed_only = _make_blocks(top_k=1)
    _, shared = _make_shared_blocks(top_k=1)
    routed_params = sum(p.numel() for p in routed_only.prototype_phi_moe.parameters())
    shared_params = sum(p.numel() for p in shared.prototype_phi_moe.parameters())
    assert shared_params < routed_params


if __name__ == "__main__":
    test_dense_checkpoint_bridge_preserves_top1_ffn_output()
    test_dense_checkpoint_bridge_preserves_top2_ffn_output()
    test_shared_dense_checkpoint_bridge_preserves_top1_ffn_output()
    test_shared_variant_reduces_stored_moe_parameters()
    print("4 synthetic tests passed")
