import pytest
import torch
import torch.nn.functional as F

from fixed_softmax_cross_entropy import (
    fixed_cross_entropy,
    fixed_log_softmax,
    fixed_softmax,
)


def test_paper_example_preserves_target_gradient():
    logits = torch.tensor(
        [[9.0, -9.0, -9.0]], dtype=torch.float32, requires_grad=True
    )
    targets = torch.tensor([0])

    loss = fixed_cross_entropy(logits, targets)
    (gradient,) = torch.autograd.grad(loss, logits)

    torch.testing.assert_close(
        loss, torch.tensor(3.046e-8), rtol=2e-3, atol=1e-11
    )
    torch.testing.assert_close(
        gradient[0],
        torch.tensor([-3.046e-8, 1.523e-8, 1.523e-8]),
        rtol=2e-3,
        atol=1e-11,
    )
    assert gradient[0, 0].item() < 0.0
    assert gradient[0, 0].item() != 0.0
    assert abs(gradient.sum().item()) < 1e-14


def test_target_gradient_stays_nonzero_past_float32_precision_cliff():
    margins = torch.tensor([16.5, 16.7, 20.0, 30.0, 80.0])
    logits = torch.stack((margins, torch.zeros_like(margins)), dim=-1)
    logits.requires_grad_()
    targets = torch.zeros(margins.shape, dtype=torch.long)

    loss = fixed_cross_entropy(logits, targets, reduction="sum")
    (gradient,) = torch.autograd.grad(loss, logits)

    expected_target_gradient = -torch.sigmoid(-margins)
    assert torch.all(gradient[:, 0] < 0)
    assert torch.all(gradient[:, 0] != 0)
    torch.testing.assert_close(
        gradient[:, 0], expected_target_gradient, rtol=2e-6, atol=0.0
    )

    # Keep the framework result as a diagnostic without assuming every version
    # still exhibits the original precision cliff.
    standard_logits = logits.detach().clone().requires_grad_()
    standard_loss = F.cross_entropy(
        standard_logits, targets, reduction="sum"
    )
    torch.autograd.grad(standard_loss, standard_logits)


@pytest.fixture(scope="module")
def scripted_fixed_cross_entropy():
    return torch.jit.script(fixed_cross_entropy)


@pytest.mark.parametrize("margin", [18.0, 30.0, 60.0])
@pytest.mark.parametrize("scripted", [False, True], ids=["eager", "torchscript"])
def test_target_gradient_is_nonzero_at_large_margins(
    margin, scripted, scripted_fixed_cross_entropy
):
    loss_fn = scripted_fixed_cross_entropy if scripted else fixed_cross_entropy
    logits = torch.tensor([[margin, 0.0]], dtype=torch.float32, requires_grad=True)
    targets = torch.tensor([0])

    loss = loss_fn(logits, targets)
    (gradient,) = torch.autograd.grad(loss, logits)
    expected = -torch.exp(torch.tensor(-margin, dtype=torch.float32))

    assert gradient[0, 0].item() < 0.0
    assert gradient[0, 0].item() != 0.0
    torch.testing.assert_close(gradient[0, 0], expected, rtol=2e-6, atol=0.0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_gradient_sums_to_zero(dtype):
    logits = torch.tensor(
        [[12.0, 0.5, -2.0, -3.0], [-7.0, 4.0, 1.0, -2.0]],
        dtype=dtype,
        requires_grad=True,
    )
    targets = torch.tensor([0, 2])

    loss = fixed_cross_entropy(logits, targets, reduction="sum")
    (gradient,) = torch.autograd.grad(loss, logits)

    error = gradient.sum(dim=-1).abs()
    scale = gradient.abs().amax(dim=-1)
    tolerance = torch.maximum(
        torch.full_like(scale, 1e-15), scale * 1e-5
    )
    assert torch.all(error <= tolerance)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_ordinary_logits_match_pytorch_cross_entropy(dtype):
    torch.manual_seed(7)
    logits = torch.randn(2, 3, 4, dtype=dtype, requires_grad=True)
    targets = torch.tensor([[0, 3, 1], [2, 1, 0]])

    fixed_loss = fixed_cross_entropy(logits, targets)
    standard_loss = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]), targets.reshape(-1)
    )
    (fixed_gradient,) = torch.autograd.grad(fixed_loss, logits, retain_graph=True)
    (standard_gradient,) = torch.autograd.grad(standard_loss, logits)

    tolerance = 2e-6 if dtype == torch.float32 else 1e-12
    torch.testing.assert_close(
        fixed_loss, standard_loss, rtol=tolerance, atol=tolerance
    )
    torch.testing.assert_close(
        fixed_gradient, standard_gradient, rtol=tolerance, atol=tolerance
    )


def test_hard_target_gradcheck_in_float64():
    torch.manual_seed(17)
    logits = torch.randn(2, 3, dtype=torch.float64, requires_grad=True)
    targets = torch.tensor([1, 2])

    assert torch.autograd.gradcheck(
        lambda value: fixed_cross_entropy(value, targets), (logits,)
    )


def test_hard_target_reductions():
    logits = torch.tensor([[0.2, 0.1, -0.3], [1.2, -0.5, 0.4], [0.0, 0.5, 1.0]])
    targets = torch.tensor([0, 2, 1])

    per_sample = fixed_cross_entropy(logits, targets, reduction="none")
    assert per_sample.shape == (3,)
    torch.testing.assert_close(
        fixed_cross_entropy(logits, targets, reduction="sum"), per_sample.sum()
    )
    torch.testing.assert_close(
        fixed_cross_entropy(logits, targets, reduction="mean"), per_sample.mean()
    )


def test_softmax_functions_match_pytorch_on_ordinary_logits():
    logits = torch.tensor([[0.2, -0.7, 1.1], [-1.3, 0.4, 0.0]])

    torch.testing.assert_close(
        fixed_log_softmax(logits), F.log_softmax(logits, dim=-1)
    )
    torch.testing.assert_close(fixed_softmax(logits), F.softmax(logits, dim=-1))


def test_log_softmax_and_softmax_keep_confident_class_gradient():
    logits = torch.tensor([[9.0, -9.0, -9.0]], requires_grad=True)

    log_probabilities = fixed_log_softmax(logits)
    (log_gradient,) = torch.autograd.grad(log_probabilities[0, 0], logits)
    assert log_probabilities[0, 0].item() < 0.0
    assert log_gradient[0, 0].item() > 0.0

    probabilities = fixed_softmax(logits)
    (probability_gradient,) = torch.autograd.grad(probabilities[0, 0], logits)
    assert probability_gradient[0, 0].item() > 0.0
    assert probability_gradient[0, 1:].lt(0).all()
    assert abs(probability_gradient.sum().item()) < 1e-14


def test_hard_target_loss_is_shift_invariant():
    logits = torch.tensor([[1.5, -0.2, 2.0], [-1.0, 0.4, 0.0]])
    targets = torch.tensor([0, 1])
    shift = 8.0

    baseline = logits.clone().requires_grad_()
    shifted = (logits + shift).requires_grad_()
    baseline_loss = fixed_cross_entropy(baseline, targets, reduction="sum")
    shifted_loss = fixed_cross_entropy(shifted, targets, reduction="sum")
    (baseline_gradient,) = torch.autograd.grad(baseline_loss, baseline)
    (shifted_gradient,) = torch.autograd.grad(shifted_loss, shifted)

    torch.testing.assert_close(shifted_loss, baseline_loss, rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(
        shifted_gradient, baseline_gradient, rtol=1e-6, atol=1e-7
    )


def test_bfloat16_target_gradient_when_supported():
    try:
        logits = torch.tensor([[6.0, 0.0]], dtype=torch.bfloat16, requires_grad=True)
        loss = fixed_cross_entropy(logits, torch.tensor([0]))
        (gradient,) = torch.autograd.grad(loss, logits)
    except (RuntimeError, NotImplementedError) as error:
        pytest.skip(f"bfloat16 operations are unavailable: {error}")

    assert gradient[0, 0].item() < 0.0
    assert gradient[0, 0].item() != 0.0


def test_class_vector_logits_accept_a_scalar_target():
    logits = torch.tensor([0.5, -0.2, 1.0], dtype=torch.float64)
    target = torch.tensor(2)

    loss = fixed_cross_entropy(logits, target)
    expected = F.cross_entropy(logits.unsqueeze(0), target.unsqueeze(0))

    torch.testing.assert_close(loss, expected)


def test_invalid_hard_target_arguments_raise_clear_errors():
    with pytest.raises(ValueError):
        fixed_log_softmax(torch.tensor(1.0))
    with pytest.raises(ValueError):
        fixed_log_softmax(torch.ones(2, 1))
    with pytest.raises(TypeError):
        fixed_log_softmax(torch.ones(2, dtype=torch.int64))
    with pytest.raises(ValueError):
        fixed_cross_entropy(torch.randn(2, 3), torch.tensor([[0], [1]]))
    with pytest.raises(TypeError):
        fixed_cross_entropy(torch.randn(2, 3), torch.tensor([0.0, 1.0]))
    with pytest.raises(ValueError):
        fixed_cross_entropy(torch.randn(2, 3), torch.tensor([0, 1]), "average")
