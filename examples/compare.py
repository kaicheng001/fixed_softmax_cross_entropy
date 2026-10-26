"""Print compact numerical comparisons for the two softmax fixes."""

import torch
import torch.nn.functional as F

from fixed_softmax_cross_entropy import (
    fixed_cross_entropy,
    fixed_soft_target_cross_entropy,
)


def _gradient(loss: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
    return torch.autograd.grad(loss, logits)[0]


def _relative_error(value: torch.Tensor, reference: torch.Tensor) -> float:
    denominator = torch.linalg.vector_norm(reference).clamp_min(
        torch.finfo(reference.dtype).tiny
    )
    return (torch.linalg.vector_norm(value - reference) / denominator).item()


def _show_collapse() -> None:
    logits = torch.tensor([[9.0, -9.0, -9.0]], dtype=torch.float32)
    targets = torch.tensor([0])

    standard_logits = logits.clone().requires_grad_()
    standard_loss = F.cross_entropy(standard_logits, targets, reduction="sum")
    standard_gradient = _gradient(standard_loss, standard_logits)

    fixed_logits = logits.clone().requires_grad_()
    fixed_loss = fixed_cross_entropy(fixed_logits, targets, reduction="sum")
    fixed_gradient = _gradient(fixed_loss, fixed_logits)

    reference_logits = logits.to(torch.float64).requires_grad_()
    reference_loss = fixed_cross_entropy(
        reference_logits, targets, reduction="sum"
    )
    reference_gradient = _gradient(reference_loss, reference_logits)

    print("Experiment 1: Softmax Collapse")
    for name, loss, gradient in (
        ("PyTorch CE", standard_loss, standard_gradient),
        ("Fixed CE", fixed_loss, fixed_gradient),
    ):
        print(f"  {name}")
        print(f"    loss:              {loss.item():.6e}")
        print(f"    target gradient:   {gradient[0, 0].item():.6e}")
        print(f"    non-target grads:  {gradient[0, 1:].tolist()}")
        print(f"    gradient sum:      {gradient.sum().item():.6e}")
        print(
            "    relative grad error vs float64: "
            f"{_relative_error(gradient.double(), reference_gradient):.6e}"
        )


def _show_cancellation() -> None:
    teacher = torch.tensor([[2.0, 1.0, 0.0]], dtype=torch.float32)
    perturbation = torch.tensor([[-1e-7, 1e-7, 0.0]], dtype=torch.float32)
    student_values = teacher + perturbation

    standard_student = student_values.clone().requires_grad_()
    standard_loss = -(
        torch.softmax(teacher, dim=-1)
        * torch.log_softmax(standard_student, dim=-1)
    ).sum()
    standard_gradient = _gradient(standard_loss, standard_student)

    fixed_student = student_values.clone().requires_grad_()
    fixed_loss = fixed_soft_target_cross_entropy(
        fixed_student, teacher, reduction="sum"
    )
    fixed_gradient = _gradient(fixed_loss, fixed_student)

    reference_gradient = (
        torch.softmax(student_values.double(), dim=-1)
        - torch.softmax(teacher.double(), dim=-1)
    )

    identical_student = teacher.clone().requires_grad_()
    identical_loss = fixed_soft_target_cross_entropy(
        identical_student, teacher, reduction="sum"
    )
    identical_gradient = _gradient(identical_loss, identical_student)

    print("Experiment 2: Softmax Cancellation")
    for name, gradient in (
        ("PyTorch soft-target CE", standard_gradient),
        ("Fixed soft-target CE", fixed_gradient),
    ):
        print(f"  {name}")
        print(f"    maximum absolute gradient: {gradient.abs().max().item():.6e}")
        print(f"    gradient sum:              {gradient.sum().item():.6e}")
        print(
            "    gradient error vs float64: "
            f"{_relative_error(gradient.double(), reference_gradient):.6e}"
        )
    print("  Identical student and teacher")
    print(f"    forward loss:              {identical_loss.item():.6e}")
    print(f"    maximum student gradient:  {identical_gradient.abs().max().item():.6e}")


def main() -> None:
    _show_collapse()
    print()
    _show_cancellation()


if __name__ == "__main__":
    main()
