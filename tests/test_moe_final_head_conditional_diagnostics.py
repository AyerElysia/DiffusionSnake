import torch

from lib.networks.diffusion.dit_denoiser_v4 import MoEFinalHead


def test_conditional_routing_capture_is_shape_safe_and_non_persistent():
    torch.manual_seed(11)
    head = MoEFinalHead(
        dim=64,
        out_dim=2,
        num_points=16,
        num_experts=4,
        top_k=2,
        use_point_embed=True,
        use_cyclic_router=True,
        use_shared_expert=True,
        expert_type="mlp",
        expert_hidden_dim=64,
    ).eval()
    x = torch.randn(3, 16, 64)
    t_emb = torch.randn(3, 64)

    head.enable_conditional_routing_capture(True)
    head.set_conditional_routing_context(
        diffusion_t=torch.tensor([0.0, 500.0, 1000.0]),
        contour_scale=torch.tensor([8.0, 16.0, 32.0]),
    )
    output = head(x, t_emb)
    events = head.drain_conditional_routing_events()

    assert output.shape == (3, 16, 2)
    assert len(events) == 1
    event = events[0]
    assert event["soft_sum"].shape == (3, 4)
    assert event["hard_sum"].shape == (3, 4)
    assert event["top1_sum"].shape == (3, 4)
    assert event["point_top1"].shape == (3, 8, 4)
    assert event["expert_delta_l2_sum"].shape == (3, 4)
    assert event["expert_delta_cross"].shape == (3, 4, 4)
    assert float(event["expert_delta_cross"].diagonal(dim1=1, dim2=2).sum()) > 0
    torch.testing.assert_close(
        event["top1_sum"].sum(dim=1),
        torch.full((3,), 16.0),
    )
    torch.testing.assert_close(
        event["hard_sum"].sum(dim=1),
        torch.full((3,), 32.0),
    )
    assert head.drain_conditional_routing_events() == []
    assert not any("conditional_routing" in key for key in head.state_dict())


def test_hard_phi_turns_hard_load_imbalance_into_router_gradient():
    head = MoEFinalHead(
        dim=32,
        out_dim=2,
        num_points=8,
        num_experts=4,
        top_k=2,
        balance_weight=1.0,
        balance_mode="hard_phi",
        hard_phi_ema_decay=0.0,
    )
    logits = torch.zeros(2, 8, 4, requires_grad=True)
    probs = torch.softmax(logits, dim=-1)
    selected_idx = torch.tensor([0, 1]).view(1, 1, 2).expand(2, 8, 2)

    loss = head._compute_balance_loss(probs, selected_idx)
    loss.backward()

    assert logits.grad is not None
    assert float(logits.grad.abs().sum()) > 0.0
    # Gradient descent must reduce the logits of the congested experts.
    assert float(logits.grad[..., :2].mean()) > 0.0
    assert float(logits.grad[..., 2:].mean()) < 0.0


if __name__ == "__main__":
    test_conditional_routing_capture_is_shape_safe_and_non_persistent()
    test_hard_phi_turns_hard_load_imbalance_into_router_gradient()
    print("conditional MoE diagnostic test passed")
