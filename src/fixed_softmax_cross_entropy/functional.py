"""Numerically corrected softmax and cross-entropy functions."""

import torch
import torch.nn.functional as F


def _validate_logits(logits: torch.Tensor, name: str = "logits") -> None:
    if not isinstance(logits, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if logits.ndim == 0:
        raise ValueError(f"{name} must have a class dimension")
    if not logits.is_floating_point():
        raise TypeError(f"{name} must have a floating-point dtype")
    if logits.shape[-1] < 2:
        raise ValueError(f"{name} must contain at least two classes")


def _reduce(loss: torch.Tensor, reduction: str) -> torch.Tensor:
    if reduction == "none":
        return loss
    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    raise ValueError("reduction must be 'none', 'mean', or 'sum'")


def fixed_log_softmax(logits: torch.Tensor) -> torch.Tensor:
    """Return log-probabilities for logits shaped ``[..., C]``."""
    _validate_logits(logits)

    max_indices = logits.argmax(dim=-1, keepdim=True)
    max_logits = logits.gather(dim=-1, index=max_indices)
    is_max = torch.zeros_like(logits, dtype=torch.bool).scatter(
        dim=-1, index=max_indices, value=True
    )
    shifted = torch.where(is_max, torch.zeros_like(logits), logits - max_logits)
    others = torch.where(is_max, torch.full_like(logits, -torch.inf), shifted)
    margin = -torch.logsumexp(others, dim=-1, keepdim=True)
    return shifted + F.logsigmoid(margin)


def fixed_softmax(logits: torch.Tensor) -> torch.Tensor:
    """Return probabilities for logits shaped ``[..., C]``."""
    return fixed_log_softmax(logits).exp()


def fixed_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    """Compute cross-entropy for logits ``[..., C]`` and integer targets ``[...]``.

    ``reduction`` must be ``"none"``, ``"mean"``, or ``"sum"``.
    """
    _validate_logits(logits)
    if not isinstance(targets, torch.Tensor):
        raise TypeError("targets must be a torch.Tensor")
    if (
        targets.dtype == torch.bool
        or targets.is_floating_point()
        or targets.is_complex()
    ):
        raise TypeError("targets must have an integer dtype")
    if targets.shape != logits.shape[:-1]:
        raise ValueError("targets must have shape logits.shape[:-1]")
    if targets.device != logits.device:
        raise ValueError("targets and logits must be on the same device")

    target_indices = targets.to(dtype=torch.long)
    target_logits = logits.gather(-1, target_indices.unsqueeze(-1))
    others = (logits - target_logits).scatter(
        -1, target_indices.unsqueeze(-1), -torch.inf
    )
    margin = -torch.logsumexp(others, dim=-1)
    return _reduce(-F.logsigmoid(margin), reduction)


def fixed_soft_target_cross_entropy(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    """Compute soft-target CE for matching student and teacher logits ``[..., C]``.

    The corrected student gradient is first-order only. Teacher gradients
    follow the forward loss; detach teacher logits when the teacher is frozen.
    ``reduction`` must be ``"none"``, ``"mean"``, or ``"sum"``.
    """
    _validate_logits(student_logits, "student_logits")
    _validate_logits(teacher_logits, "teacher_logits")
    if student_logits.shape != teacher_logits.shape:
        raise ValueError("student_logits and teacher_logits must have the same shape")
    if student_logits.device != teacher_logits.device:
        raise ValueError("student_logits and teacher_logits must be on the same device")
    if student_logits.dtype != teacher_logits.dtype:
        raise ValueError("student_logits and teacher_logits must have the same dtype")

    log_q = fixed_log_softmax(student_logits)
    q = log_q.exp()
    p = fixed_softmax(teacher_logits)

    delta = student_logits - teacher_logits
    d = delta - (p * delta).sum(dim=-1, keepdim=True)
    s = d - torch.log1p((p * torch.expm1(d)).sum(dim=-1, keepdim=True))

    use_log_ratio = s.abs() < 1
    safe_s = torch.where(use_log_ratio, s, torch.zeros_like(s))
    u = torch.where(use_log_ratio, p * torch.expm1(safe_s), q - p)
    gradient = u - q * u.sum(dim=-1, keepdim=True)

    per_sample_loss = -(p * log_q.detach()).sum(dim=-1)
    gradient_only = (
        gradient.detach() * (student_logits - student_logits.detach())
    ).sum(dim=-1)
    return _reduce(per_sample_loss + gradient_only, reduction)
