import torch

from lib.networks.diffusion.dit_denoiser_v4 import (
    DenseResidualFinalHead,
    ModernSparseResidualHead,
    SharedDenseSparseResidualHead,
)
from lib.networks.diffusion.dit_blocks_v2 import modulate


def _force_single_contour_route(head, x, t_emb):
    with torch.no_grad():
        shift, scale = head.adaLN(t_emb).chunk(2, dim=1)
        routed_x = modulate(head.norm(x), shift, scale)
        descriptor = routed_x.float().mean(dim=1)
        descriptor = torch.nn.functional.layer_norm(descriptor, (head.dim,))
        direction = torch.nn.functional.normalize(descriptor[0], dim=0)
        head.prototypes.copy_(-direction.repeat(head.num_experts, 1))
        head.prototypes[0].copy_(direction)
        if head.top_k > 1:
            head.prototypes[1].copy_(direction)
        head._prototypes_initialized.fill_(True)


def test_dense_control_matches_four_expert_parameter_pool():
    dense = DenseResidualFinalHead(dim=256, hidden_dim=1024)
    sparse = ModernSparseResidualHead(
        dim=256,
        num_experts=4,
        top_k=2,
        expert_hidden_dim=256,
    )
    dense_residual = sum(p.numel() for p in dense.residual_mlp.parameters())
    sparse_pool = sum(p.numel() for p in sparse.experts.parameters())
    assert abs(dense_residual - sparse_pool) <= 8


def test_top1_executes_only_selected_expert():
    torch.manual_seed(7)
    head = ModernSparseResidualHead(
        dim=32,
        num_experts=4,
        top_k=1,
        expert_hidden_dim=32,
    ).eval()
    x = torch.randn(1, 8, 32).repeat(3, 1, 1)
    t_emb = torch.zeros(3, 32)
    _force_single_contour_route(head, x, t_emb)

    calls = [0, 0, 0, 0]
    hooks = []
    for expert_id, expert in enumerate(head.experts):
        hooks.append(expert.register_forward_hook(
            lambda _module, _args, _out, expert_id=expert_id: calls.__setitem__(
                expert_id, calls[expert_id] + 1
            )
        ))
    output = head(x, t_emb)
    for hook in hooks:
        hook.remove()

    assert output.shape == (3, 8, 2)
    assert calls == [1, 0, 0, 0]


def test_top2_executes_two_selected_experts_not_full_pool():
    torch.manual_seed(13)
    head = ModernSparseResidualHead(
        dim=32,
        num_experts=4,
        top_k=2,
        expert_hidden_dim=32,
    ).eval()
    x = torch.randn(1, 8, 32).repeat(3, 1, 1)
    t_emb = torch.zeros(3, 32)
    _force_single_contour_route(head, x, t_emb)

    calls = [0, 0, 0, 0]
    hooks = []
    for expert_id, expert in enumerate(head.experts):
        hooks.append(expert.register_forward_hook(
            lambda _module, _args, _out, expert_id=expert_id: calls.__setitem__(
                expert_id, calls[expert_id] + 1
            )
        ))
    head(x, t_emb)
    for hook in hooks:
        hook.remove()

    assert sum(calls) == 2
    assert calls[0] == 1
    assert calls[1] == 1
    assert calls[2:] == [0, 0]


def test_shared_linear_path_is_exact_when_residuals_are_zero():
    torch.manual_seed(19)
    head = ModernSparseResidualHead(
        dim=32,
        num_experts=4,
        top_k=2,
        expert_hidden_dim=32,
    ).eval()
    with torch.no_grad():
        head.linear.weight.normal_(std=0.1)
        head.linear.bias.normal_(std=0.1)
        for expert in head.experts:
            expert[-1].weight.zero_()
            expert[-1].bias.zero_()
    x = torch.randn(5, 8, 32)
    t_emb = torch.randn(5, 32)
    shift, scale = head.adaLN(t_emb).chunk(2, dim=1)
    normalized = modulate(head.norm(x), shift, scale)
    expected = head.linear(normalized)
    actual = head(x, t_emb)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_router_and_selected_experts_receive_gradients():
    torch.manual_seed(23)
    head = ModernSparseResidualHead(
        dim=32,
        num_experts=4,
        top_k=2,
        expert_hidden_dim=32,
        balance_weight=1.0,
        phi_ema_decay=0.0,
        contrastive_weight=0.1,
    ).train()
    x = torch.randn(12, 8, 32, requires_grad=True)
    t_emb = torch.randn(12, 32)
    output = head(x, t_emb)
    loss = output.square().mean() + head.reg_loss()
    loss.backward()

    assert x.grad is not None and float(x.grad.abs().sum()) > 0
    assert head.prototypes.grad is not None
    assert float(head.prototypes.grad.abs().sum()) > 0
    expert_grad = sum(
        float(parameter.grad.abs().sum())
        for parameter in head.experts.parameters()
        if parameter.grad is not None
    )
    assert expert_grad > 0


def test_shared_sparse_is_exact_dense_function_at_initialization():
    torch.manual_seed(29)
    dense = DenseResidualFinalHead(dim=32, hidden_dim=64).eval()
    sparse = SharedDenseSparseResidualHead(
        dim=32,
        shared_hidden_dim=64,
        num_experts=4,
        expert_hidden_dim=16,
    ).eval()
    sparse.load_state_dict(dense.state_dict(), strict=False)
    x = torch.randn(7, 8, 32)
    t_emb = torch.randn(7, 32)
    torch.testing.assert_close(sparse(x, t_emb), dense(x, t_emb), rtol=0, atol=0)


def test_shared_sparse_top1_executes_only_selected_experts():
    torch.manual_seed(31)
    head = SharedDenseSparseResidualHead(
        dim=32,
        shared_hidden_dim=64,
        num_experts=4,
        expert_hidden_dim=16,
    ).eval()
    x = torch.randn(1, 8, 32).repeat(5, 1, 1)
    t_emb = torch.randn(5, 32)
    calls = [0, 0, 0, 0]
    hooks = [
        expert.register_forward_hook(
            lambda _module, _args, _out, expert_id=expert_id: calls.__setitem__(
                expert_id, calls[expert_id] + 1
            )
        )
        for expert_id, expert in enumerate(head.experts)
    ]
    head(x, t_emb)
    for hook in hooks:
        hook.remove()
    assert sum(calls) == 1


def test_shared_sparse_router_gets_task_gradient():
    torch.manual_seed(37)
    head = SharedDenseSparseResidualHead(
        dim=32,
        shared_hidden_dim=64,
        num_experts=4,
        expert_hidden_dim=16,
    ).train()
    with torch.no_grad():
        for expert in head.experts:
            expert[-1].weight.normal_(std=0.05)
    x = torch.randn(12, 8, 32, requires_grad=True)
    t_emb = torch.randn(12, 32)
    head(x, t_emb).square().mean().backward()
    assert head.router.weight.grad is not None
    assert float(head.router.weight.grad.abs().sum()) > 0


if __name__ == "__main__":
    test_dense_control_matches_four_expert_parameter_pool()
    test_top1_executes_only_selected_expert()
    test_top2_executes_two_selected_experts_not_full_pool()
    test_shared_linear_path_is_exact_when_residuals_are_zero()
    test_router_and_selected_experts_receive_gradients()
    test_shared_sparse_is_exact_dense_function_at_initialization()
    test_shared_sparse_top1_executes_only_selected_experts()
    test_shared_sparse_router_gets_task_gradient()
    print("modern output head tests passed")
