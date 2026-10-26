import pytest
import torch
import torch.nn.functional as F

from fixed_softmax_cross_entropy import fixed_soft_target_cross_entropy


def _double_reference_gradient(student, teacher):
    student64 = student.detach().to(torch.float64)
    teacher64 = teacher.detach().to(torch.float64)
    return torch.softmax(student64, dim=-1) - torch.softmax(teacher64, dim=-1)


def test_identical_student_and_teacher_have_zero_student_gradient():
    student = torch.tensor([[2.0, 1.0, 0.0]], requires_grad=True)
    teacher = student.detach().clone()
    teacher_probabilities = torch.softmax(teacher, dim=-1)
    expected_loss = -(
        teacher_probabilities * torch.log_softmax(teacher, dim=-1)
    ).sum(dim=-1).mean()

    loss = fixed_soft_target_cross_entropy(student, teacher)
    (gradient,) = torch.autograd.grad(loss, student)

    torch.testing.assert_close(loss, expected_loss, rtol=1e-6, atol=1e-7)
    assert loss.item() > 0.0
    assert torch.count_nonzero(gradient).item() == 0


def test_paper_perturbation_matches_float64_reference_better_than_standard():
    teacher = torch.tensor([[2.0, 1.0, 0.0]], dtype=torch.float32)
    perturbation = torch.tensor([[-1e-7, 1e-7, 0.0]], dtype=torch.float32)
    student = (teacher + perturbation).detach().requires_grad_()
    targets = teacher.clone()

    fixed_loss = fixed_soft_target_cross_entropy(student, targets)
    (fixed_gradient,) = torch.autograd.grad(fixed_loss, student)

    standard_student = student.detach().clone().requires_grad_()
    standard_loss = -(
        torch.softmax(targets, dim=-1)
        * torch.log_softmax(standard_student, dim=-1)
    ).sum(dim=-1).mean()
    (standard_gradient,) = torch.autograd.grad(standard_loss, standard_student)
    reference = _double_reference_gradient(student, targets)

    fixed_error = torch.linalg.vector_norm(fixed_gradient.double() - reference)
    standard_error = torch.linalg.vector_norm(standard_gradient.double() - reference)
    if standard_error.item() > 1e-10:
        assert fixed_error.item() < standard_error.item() * 0.1
    else:
        # A future PyTorch version may have fixed the same cancellation bug.
        torch.testing.assert_close(
            standard_gradient.double(), reference, rtol=5e-2, atol=2e-10
        )
    torch.testing.assert_close(
        fixed_gradient.double(), reference, rtol=5e-2, atol=2e-10
    )


def test_corrected_student_gradient_sums_to_zero():
    student = torch.tensor([[2.0, 1.0, 0.0]], requires_grad=True)
    teacher = torch.tensor([[1.9, 1.1, -0.1]])

    loss = fixed_soft_target_cross_entropy(student, teacher)
    (gradient,) = torch.autograd.grad(loss, student)

    scale = gradient.abs().amax().item()
    assert abs(gradient.sum().item()) <= max(1e-15, scale * 1e-5)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_ordinary_distillation_matches_standard_cross_entropy(dtype):
    torch.manual_seed(11)
    student = torch.randn(3, 4, dtype=dtype, requires_grad=True)
    teacher = torch.randn(3, 4, dtype=dtype)

    fixed_loss = fixed_soft_target_cross_entropy(student, teacher)
    standard_loss = -(
        torch.softmax(teacher, dim=-1)
        * torch.log_softmax(student, dim=-1)
    ).sum(dim=-1).mean()
    (fixed_gradient,) = torch.autograd.grad(fixed_loss, student, retain_graph=True)
    (standard_gradient,) = torch.autograd.grad(standard_loss, student)

    tolerance = 3e-6 if dtype == torch.float32 else 1e-12
    torch.testing.assert_close(
        fixed_loss, standard_loss, rtol=tolerance, atol=tolerance
    )
    torch.testing.assert_close(
        fixed_gradient, standard_gradient, rtol=tolerance, atol=tolerance
    )


@pytest.mark.parametrize("reduction", ["none", "mean", "sum"])
def test_batched_inputs_and_reductions(reduction):
    torch.manual_seed(13)
    student = torch.randn(2, 3, 4, requires_grad=True)
    teacher = torch.randn(2, 3, 4)

    loss = fixed_soft_target_cross_entropy(student, teacher, reduction=reduction)
    per_sample = -(
        torch.softmax(teacher, dim=-1) * torch.log_softmax(student, dim=-1)
    ).sum(dim=-1)
    expected = {
        "none": per_sample,
        "mean": per_sample.mean(),
        "sum": per_sample.sum(),
    }[reduction]

    torch.testing.assert_close(loss, expected, rtol=2e-6, atol=2e-6)


def test_teacher_receives_forward_loss_gradient_when_trainable():
    student = torch.tensor([[1.0, 0.2, -0.5]], requires_grad=True)
    teacher = torch.tensor([[0.8, 0.3, -0.2]], requires_grad=True)
    expected_teacher = teacher.detach().clone().requires_grad_()
    expected_loss = -(
        torch.softmax(expected_teacher, dim=-1)
        * torch.log_softmax(student.detach(), dim=-1)
    ).sum()
    (expected_gradient,) = torch.autograd.grad(expected_loss, expected_teacher)

    fixed_soft_target_cross_entropy(student, teacher).backward()

    torch.testing.assert_close(teacher.grad, expected_gradient, rtol=1e-6, atol=1e-7)
    assert student.grad is not None


def test_small_representable_perturbations_approach_zero_smoothly():
    teacher = torch.tensor([[2.0, 1.0, 0.0]], dtype=torch.float32)
    gradient_norms = []

    for magnitude in [1e-4, 1e-5, 1e-6, 1e-7]:
        perturbation = torch.tensor(
            [[-magnitude, magnitude, 0.0]], dtype=torch.float32
        )
        student = (teacher + perturbation).detach().requires_grad_()
        loss = fixed_soft_target_cross_entropy(student, teacher)
        (gradient,) = torch.autograd.grad(loss, student)
        reference = _double_reference_gradient(student, teacher)

        assert torch.count_nonzero(gradient).item() > 0
        torch.testing.assert_close(
            gradient.double(), reference, rtol=0.08, atol=2e-10
        )
        gradient_norms.append(torch.linalg.vector_norm(gradient).item())

    assert all(
        left > right for left, right in zip(gradient_norms, gradient_norms[1:])
    )


def test_bfloat16_soft_target_cancellation_matches_float64_reference():
    try:
        teacher = torch.tensor([[2.0, 1.0, 0.0]], dtype=torch.bfloat16)
        perturbation = torch.tensor(
            [[-0.015625, 0.0078125, 0.0]], dtype=torch.bfloat16
        )
        student = (teacher + perturbation).detach().requires_grad_()
        loss = fixed_soft_target_cross_entropy(student, teacher)
        (gradient,) = torch.autograd.grad(loss, student)
    except (RuntimeError, NotImplementedError) as error:
        pytest.skip(f"bfloat16 soft-target operations are unavailable: {error}")

    reference = _double_reference_gradient(student, teacher)
    assert torch.isfinite(loss).all()
    assert torch.count_nonzero(gradient).item() == gradient.numel()
    torch.testing.assert_close(gradient.double(), reference, rtol=0.02, atol=1e-6)


def test_large_log_ratio_fallback_stays_finite():
    student = torch.tensor([[-1000.0, 0.0]], requires_grad=True)
    teacher = torch.tensor([[0.0, -1000.0]])

    loss = fixed_soft_target_cross_entropy(student, teacher)
    (gradient,) = torch.autograd.grad(loss, student)

    assert torch.isfinite(loss).all()
    assert torch.isfinite(gradient).all()
    torch.testing.assert_close(gradient, torch.tensor([[-1.0, 1.0]]))


def test_invalid_soft_target_arguments_raise_clear_errors():
    with pytest.raises(ValueError):
        fixed_soft_target_cross_entropy(torch.randn(2, 3), torch.randn(2, 4))
    with pytest.raises(ValueError):
        fixed_soft_target_cross_entropy(
            torch.randn(2, 3), torch.randn(2, 3), reduction="average"
        )
