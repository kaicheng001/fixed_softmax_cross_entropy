"""Generate the two-panel numerical comparison GIF."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import torch
import torch.nn.functional as F

from fixed_softmax_cross_entropy import (
    fixed_cross_entropy,
    fixed_soft_target_cross_entropy,
)


FRAMES = 48
FPS = 12
DPI = 90
STANDARD = "#C66B4E"
FIXED = "#168A83"
REFERENCE = "#68727D"


def _gradient(loss: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
    return torch.autograd.grad(loss, logits)[0]


def _collapse_data():
    margins = torch.linspace(0.0, 30.0, 100, dtype=torch.float32)
    logits = torch.stack((margins, torch.zeros_like(margins)), dim=-1)
    targets = torch.zeros(margins.shape, dtype=torch.long)

    standard_logits = logits.detach().clone().requires_grad_()
    standard_loss = F.cross_entropy(standard_logits, targets, reduction="sum")
    standard_gradient = _gradient(standard_loss, standard_logits)[:, 0]

    fixed_logits = logits.detach().clone().requires_grad_()
    fixed_loss = fixed_cross_entropy(fixed_logits, targets, reduction="sum")
    fixed_gradient = _gradient(fixed_loss, fixed_logits)[:, 0]

    reference_logits = logits.double().requires_grad_()
    reference_loss = fixed_cross_entropy(reference_logits, targets, reduction="sum")
    reference_gradient = _gradient(reference_loss, reference_logits)[:, 0]

    if fixed_gradient[-1].item() == 0 or reference_gradient[-1].item() == 0:
        raise AssertionError("The high-margin target gradient should remain representable")

    return (
        margins.tolist(),
        standard_gradient.tolist(),
        fixed_gradient.tolist(),
        reference_gradient.tolist(),
    )


def _cancellation_data():
    teacher = torch.tensor([[2.0, 1.0, 0.0]], dtype=torch.float32)
    epsilons = torch.logspace(-1, -9, 100, dtype=torch.float32)
    standard_errors = []
    fixed_errors = []
    identical_logits = []

    for epsilon in epsilons:
        perturbation = torch.tensor(
            [[-epsilon.item(), epsilon.item(), 0.0]], dtype=torch.float32
        )
        student_values = teacher + perturbation
        reference = (
            torch.softmax(student_values.double(), dim=-1)
            - torch.softmax(teacher.double(), dim=-1)
        )

        standard_student = student_values.detach().clone().requires_grad_()
        standard_loss = -(
            torch.softmax(teacher, dim=-1)
            * torch.log_softmax(standard_student, dim=-1)
        ).sum()
        standard_gradient = _gradient(standard_loss, standard_student)

        fixed_student = student_values.detach().clone().requires_grad_()
        fixed_loss = fixed_soft_target_cross_entropy(
            fixed_student, teacher, reduction="sum"
        )
        fixed_gradient = _gradient(fixed_loss, fixed_student)

        is_identical = torch.equal(student_values, teacher)
        if is_identical and torch.count_nonzero(fixed_gradient).item() != 0:
            raise AssertionError("Identical logits must have a zero fixed gradient")

        standard_errors.append(
            (standard_gradient.double() - reference).abs().max().item()
        )
        fixed_errors.append((fixed_gradient.double() - reference).abs().max().item())
        identical_logits.append(is_identical)

    return epsilons.tolist(), standard_errors, fixed_errors, identical_logits


def main() -> None:
    margins, standard_gradient, fixed_gradient, reference_gradient = _collapse_data()
    epsilons, standard_error, fixed_error, identical = _cancellation_data()

    collapse_positive = [
        abs(value)
        for values in (standard_gradient, fixed_gradient, reference_gradient)
        for value in values
        if value != 0
    ]
    cancellation_positive = [
        value
        for values in (standard_error, fixed_error)
        for value in values
        if value > 0
    ]
    collapse_floor = min(collapse_positive) / 5
    cancellation_floor = min(cancellation_positive) / 5 if cancellation_positive else 1e-16

    fig, (ax_collapse, ax_cancellation) = plt.subplots(
        1, 2, figsize=(11, 4.5), dpi=DPI
    )
    fig.patch.set_facecolor("white")
    for axis in (ax_collapse, ax_cancellation):
        axis.set_facecolor("white")
        axis.grid(True, which="major", color="#D9DEE3", linewidth=0.7)
        axis.grid(True, which="minor", color="#E8EBEE", linewidth=0.45, alpha=0.8)
        axis.set_axisbelow(True)
        axis.tick_params(labelsize=8)
        for spine in axis.spines.values():
            spine.set_color("#BFC6CC")

    standard_abs = [abs(value) for value in standard_gradient]
    fixed_abs = [abs(value) for value in fixed_gradient]
    reference_abs = [abs(value) for value in reference_gradient]
    standard_visible = [
        value if value > 0 else float("nan") for value in standard_abs
    ]
    zero_margins = [
        margin for margin, value in zip(margins, standard_gradient) if value == 0
    ]

    ax_collapse.plot(
        margins, standard_visible, color=STANDARD, linewidth=2.0, label="Standard PyTorch"
    )
    ax_collapse.plot(
        margins, fixed_abs, color=FIXED, linewidth=2.2, label="Fixed"
    )
    ax_collapse.plot(
        margins,
        reference_abs,
        color=REFERENCE,
        linewidth=2.0,
        linestyle=(0, (4, 2)),
        label="Float64 reference",
        zorder=4,
    )
    if zero_margins:
        ax_collapse.scatter(
            zero_margins,
            [collapse_floor] * len(zero_margins),
            color=STANDARD,
            edgecolors="white",
            linewidths=0.5,
            marker="v",
            s=25,
            label="Rounded to zero",
            zorder=5,
        )
        ax_collapse.text(
            0.97,
            0.055,
            "Rounded to zero",
            color=STANDARD,
            fontsize=7.5,
            ha="right",
            va="bottom",
            transform=ax_collapse.transAxes,
        )

    ax_collapse.set_yscale("log")
    ax_collapse.set_xlim(0.0, 30.0)
    ax_collapse.set_ylim(collapse_floor, 0.8)
    ax_collapse.set_xlabel("Logit margin", fontsize=9)
    ax_collapse.set_ylabel("Absolute target gradient", fontsize=9)
    ax_collapse.set_title(
        "(a) Softmax Collapse\nTarget gradient vs. logit margin",
        fontsize=10,
        pad=8,
    )
    ax_collapse.axvline(
        16.6, color=REFERENCE, linewidth=1.0, linestyle=(0, (3, 2)), zorder=1
    )
    ax_collapse.text(
        16.85,
        0.97,
        "FP32 precision cliff",
        rotation=90,
        color=REFERENCE,
        fontsize=7.5,
        va="top",
        ha="left",
        transform=ax_collapse.get_xaxis_transform(),
    )
    ax_collapse.legend(
        loc="lower left", frameon=False, fontsize=7.5, ncol=2, handlelength=2.0
    )

    identical_indices = [index for index, value in enumerate(identical) if value]
    if identical_indices:
        first_identical = identical_indices[0]
        ax_cancellation.axvspan(
            min(epsilons[first_identical], epsilons[-1]),
            max(epsilons[first_identical], epsilons[-1]),
            color="#CBD1D6",
            alpha=0.24,
            label="Identical float32 logits",
            zorder=0,
        )

    for errors, color, label in (
        (standard_error, STANDARD, "Standard PyTorch"),
        (fixed_error, FIXED, "Fixed"),
    ):
        visible_errors = [
            value if value > 0 else float("nan") for value in errors
        ]
        ax_cancellation.plot(
            epsilons, visible_errors, color=color, linewidth=2.0, label=label
        )
        zero_epsilons = [
            epsilon for epsilon, value in zip(epsilons, errors) if value == 0
        ]
        if zero_epsilons:
            ax_cancellation.scatter(
                zero_epsilons,
                [cancellation_floor] * len(zero_epsilons),
                color=color,
                edgecolors="white",
                linewidths=0.5,
                marker="v",
                s=25,
                zorder=5,
            )

    ax_cancellation.scatter(
        [],
        [],
        color=REFERENCE,
        marker="v",
        s=25,
        label="Zero error at lower boundary",
    )
    ax_cancellation.set_xscale("log")
    ax_cancellation.set_yscale("log")
    ax_cancellation.set_xlim(epsilons[0], epsilons[-1])
    max_error = max(standard_error + fixed_error)
    ax_cancellation.set_ylim(
        cancellation_floor, max(max_error * 3, cancellation_floor * 10)
    )
    ax_cancellation.set_xlabel("Perturbation scale ε", fontsize=9)
    ax_cancellation.set_ylabel("Max absolute gradient error", fontsize=9)
    ax_cancellation.set_title(
        "(b) Softmax Cancellation\nGradient error vs. perturbation",
        fontsize=10,
        pad=8,
    )
    ax_cancellation.legend(
        loc="lower left", frameon=False, fontsize=7.2, ncol=2, handlelength=2.0
    )

    standard_point_a, = ax_collapse.plot(
        [], [], marker="o", linestyle="none", markersize=4.5, color=STANDARD
    )
    fixed_point_a, = ax_collapse.plot(
        [], [], marker="o", linestyle="none", markersize=4.5, color=FIXED
    )
    reference_point_a, = ax_collapse.plot(
        [],
        [],
        marker="D",
        linestyle="none",
        markersize=4.0,
        markerfacecolor="white",
        markeredgecolor=REFERENCE,
    )
    info_a = ax_collapse.text(
        0.975,
        0.97,
        "",
        transform=ax_collapse.transAxes,
        ha="right",
        va="top",
        fontsize=7.5,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "alpha": 0.94},
    )

    standard_point_b, = ax_cancellation.plot(
        [], [], marker="o", linestyle="none", markersize=4.5, color=STANDARD
    )
    fixed_point_b, = ax_cancellation.plot(
        [], [], marker="o", linestyle="none", markersize=4.5, color=FIXED
    )
    info_b = ax_cancellation.text(
        0.5,
        0.97,
        "",
        transform=ax_cancellation.transAxes,
        ha="center",
        va="top",
        fontsize=7.0,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.94},
    )
    identical_label_b = ax_cancellation.text(
        0.97,
        0.74,
        "",
        transform=ax_cancellation.transAxes,
        ha="right",
        va="top",
        fontsize=7.0,
        color=REFERENCE,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.92},
    )

    def update(frame: int):
        index = round(frame * (len(margins) - 1) / (FRAMES - 1))
        margin = margins[index]
        standard_value = standard_gradient[index]
        fixed_value = fixed_gradient[index]
        reference_value = reference_gradient[index]
        standard_y = abs(standard_value) if standard_value != 0 else collapse_floor
        standard_point_a.set_data([margin], [standard_y])
        fixed_point_a.set_data([margin], [abs(fixed_value)])
        reference_point_a.set_data([margin], [abs(reference_value)])
        is_zero = standard_value == 0
        standard_point_a.set_markerfacecolor("white" if is_zero else STANDARD)
        info_a.set_text(
            f"margin = {margin:.2f}\n"
            f"standard gradient = {standard_value:.3e}"
            f"{' (rounded to zero)' if is_zero else ''}\n"
            f"fixed gradient = {fixed_value:.3e}"
        )
        info_a.get_bbox_patch().set_edgecolor(STANDARD if is_zero else "#D9DEE3")

        epsilon = epsilons[index]
        std_error = standard_error[index]
        fix_error = fixed_error[index]
        std_y = std_error if std_error > 0 else cancellation_floor
        fix_y = fix_error if fix_error > 0 else cancellation_floor
        standard_point_b.set_data([epsilon], [std_y])
        fixed_point_b.set_data([epsilon], [fix_y])
        standard_point_b.set_markerfacecolor("white" if std_error == 0 else STANDARD)
        fixed_point_b.set_markerfacecolor("white" if fix_error == 0 else FIXED)
        info_b.set_text(
            f"ε={epsilon:.1e} | std err={std_error:.1e} | fixed err={fix_error:.1e}"
        )
        identical_label_b.set_text(
            "Identical float32 logits" if identical[index] else ""
        )
        info_b.get_bbox_patch().set_edgecolor(
            FIXED if identical[index] else "#D9DEE3"
        )
        return (
            standard_point_a,
            fixed_point_a,
            reference_point_a,
            info_a,
            standard_point_b,
            fixed_point_b,
            info_b,
            identical_label_b,
        )

    animation = FuncAnimation(
        fig,
        update,
        frames=FRAMES,
        interval=1000 / FPS,
        blit=True,
        repeat=True,
    )
    output_path = (
        Path(__file__).resolve().parents[1] / "assets" / "softmax_fix.gif"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    animation.save(output_path, writer=PillowWriter(fps=FPS), dpi=DPI)
    plt.close(fig)

    size_kib = output_path.stat().st_size / 1024
    print(f"Saved {output_path} ({11 * DPI}x{4.5 * DPI:.0f}, {size_kib:.1f} KiB)")


if __name__ == "__main__":
    main()
