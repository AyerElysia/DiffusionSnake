import importlib.util
import math
from pathlib import Path

import torch

_MODULE_PATH = Path(__file__).parents[1] / "lib/train/per_point_ppo.py"
_SPEC = importlib.util.spec_from_file_location("per_point_ppo", _MODULE_PATH)
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
masked_pointwise_ppo_loss = _MODULE.masked_pointwise_ppo_loss
per_point_squashed_gaussian_logprob = _MODULE.per_point_squashed_gaussian_logprob

_POLICY_PATH = Path(__file__).parents[1] / "lib/train/per_point_fm_policy.py"
_POLICY_SPEC = importlib.util.spec_from_file_location("per_point_fm_policy", _POLICY_PATH)
_POLICY_MODULE = importlib.util.module_from_spec(_POLICY_SPEC)
_POLICY_SPEC.loader.exec_module(_POLICY_MODULE)
PerPointFMScalePolicy = _POLICY_MODULE.PerPointFMScalePolicy


def test_masked_pointwise_loss_gradient_assignment():
    current = torch.zeros(1, 6, requires_grad=True)
    old = torch.zeros_like(current)
    advantage = torch.tensor([[0.0, 1.0, -2.0, 0.5, 0.0, 0.0]])
    mask = torch.tensor([[False, True, True, True, False, False]])

    loss, _ = masked_pointwise_ppo_loss(current, old, advantage, mask, clip=0.2)
    loss.backward()

    grad = current.grad.detach()
    assert torch.equal(grad[~mask], torch.zeros_like(grad[~mask]))
    selected_grad = grad[mask]
    assert torch.all(selected_grad != 0)
    assert selected_grad.unique().numel() == selected_grad.numel()
    assert torch.sign(selected_grad[0]) != torch.sign(selected_grad[1])


def test_zero_mean_local_policy_has_local_only_gradient():
    policy = PerPointFMScalePolicy(
        outer_steps=1,
        feature_dim=4,
        feature_embed_dim=4,
        hidden_dim=8,
        zero_mean_local=True,
    )
    with torch.no_grad():
        policy.point_net[-1].weight.fill_(0.1)

    poly = torch.zeros(1, 6, 2)
    mean_action = torch.ones_like(poly)
    sampled_feat = torch.randn(1, 6, 4)
    mu, _ = policy(0, poly, poly, mean_action, sampled_feat, 1.0)
    assert torch.allclose(mu.mean(dim=1), torch.zeros(1), atol=1e-6)

    mu.retain_grad()
    mu[0, 2].backward()
    assert mu.grad[0, 2] == 1
    assert torch.equal(mu.grad[0, :2], torch.zeros(2))
    assert torch.equal(mu.grad[0, 3:], torch.zeros(3))


def test_squashed_gaussian_logprob_includes_scale_jacobian():
    raw = torch.zeros(1, 3)
    mu = torch.zeros_like(raw)
    logstd = torch.zeros(1, 1)
    max_scale = 0.25

    logprob = per_point_squashed_gaussian_logprob(raw, mu, logstd, max_scale)

    expected = -0.5 * math.log(2.0 * math.pi) - math.log(max_scale)
    torch.testing.assert_close(logprob, torch.full_like(raw, expected))


def test_masked_pointwise_ppo_loss_uses_sign_correct_clipping():
    current = torch.log(torch.tensor([[1.5, 0.5, 3.0]]))
    old = torch.zeros_like(current)
    advantage = torch.tensor([[1.0, -1.0, 10.0]])
    mask = torch.tensor([[True, True, False]])

    loss, selected_ratio = masked_pointwise_ppo_loss(
        current, old, advantage, mask, clip=0.2
    )

    torch.testing.assert_close(selected_ratio, torch.tensor([1.5, 0.5]))
    torch.testing.assert_close(loss, torch.tensor(-0.2))


if __name__ == "__main__":
    test_masked_pointwise_loss_gradient_assignment()
    test_zero_mean_local_policy_has_local_only_gradient()
    test_squashed_gaussian_logprob_includes_scale_jacobian()
    test_masked_pointwise_ppo_loss_uses_sign_correct_clipping()
