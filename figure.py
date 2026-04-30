"""Visualization utilities for bitflip experiments."""

import csv
import math
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


def _load_perplexity_series(csv_path: str) -> tuple[list[str], list[float]]:
    iterations: list[str] = []
    perplexities: list[float] = []

    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        raise ValueError(f"{csv_path} is empty.")

    # Extract initial perplexity from the first row's base_loss.
    initial_loss = float(rows[0]["base_loss"])
    iterations.append("0")
    perplexities.append(math.exp(initial_loss))

    # Extract perplexity after each flip.
    for i, row in enumerate(rows):
        loss_after_flip = float(row["loss_after_flip"])
        iterations.append(str(i + 1))
        perplexities.append(math.exp(loss_after_flip))

    return iterations, perplexities


def plot_perplexity_over_flips(
    csv_path: str = "bitflip_metadata.csv",
    output_path: str | None = None,
) -> None:
    """Plot perplexity (= exp(loss)) at each bitflip iteration.
    
    Reads the CSV output from gate_grad_bit_rank or gate_hess_bit_rank and plots
    the perplexity trajectory, starting with initial perplexity and showing the
    perplexity after each of the n bitflips.
    
    Args:
        csv_path: Path to the bitflip metadata CSV file.
        output_path: Output image path. If None, derives a unique name from csv_path.
    """
    try:
        iterations, perplexities = _load_perplexity_series(csv_path)
    except FileNotFoundError:
        print(f"Error: {csv_path} not found.")
        return
    except ValueError as exc:
        print(f"Error: {exc}")
        return
    
    # Create figure.
    fig_width = 500 / 72.27
    fig, ax = plt.subplots(figsize=(fig_width, fig_width * 0.4))
    
    x_pos = range(len(iterations))
    ax.plot(x_pos, perplexities, marker="o", linestyle="-", linewidth=2, markersize=8, color="blue")
    
    # Annotations.
    ax.set_xlabel("Bitflip Iteration", fontsize=9, labelpad=0.5)
    ax.set_ylabel("Perplexity", fontsize=9, labelpad=0.5)
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(mticker.LogLocator(base=10.0, subs=(1.0,)))
    ax.yaxis.set_minor_locator(mticker.NullLocator())
    ax.set_xticks(x_pos)
    ax.set_xticklabels(iterations, rotation=45, ha="right")
    ax.tick_params(axis="both", labelsize=8)
    ax.grid(True, which="major", alpha=0.3)
    
    # Annotate each point with the value.
    for i, (x, y) in enumerate(zip(x_pos, perplexities)):
        ax.annotate(f"{y:.2f}", (x, y), textcoords="offset points", xytext=(0, 10), ha="center", fontsize=9)
    
    plt.tight_layout()
    if output_path is None:
        stem = Path(csv_path).stem
        output_path = f"perplexity_over_flips_{stem}.pdf"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Figure saved to {output_path}")
    plt.show()


def plot_perplexity_side_by_side(
    olmoe_grad_csv_path: str = "olmoe_grad.csv",
    olmoe_hess_csv_path: str = "olmoe_hess.csv",
    qwen_grad_csv_path: str = "qwen_grad.csv",
    qwen_hess_csv_path: str = "qwen_hess.csv",
    output_path: str = "perplexity_side_by_side.pdf",
) -> None:
    """Plot OLMoE and Qwen side by side, with gradient and hessian lines in each subplot."""
    try:
        olmoe_grad_iterations, olmoe_grad_perplexities = _load_perplexity_series(olmoe_grad_csv_path)
        olmoe_hess_iterations, olmoe_hess_perplexities = _load_perplexity_series(olmoe_hess_csv_path)
        qwen_grad_iterations, qwen_grad_perplexities = _load_perplexity_series(qwen_grad_csv_path)
        qwen_hess_iterations, qwen_hess_perplexities = _load_perplexity_series(qwen_hess_csv_path)
    except FileNotFoundError as exc:
        print(f"Error: {exc}")
        return
    except ValueError as exc:
        print(f"Error: {exc}")
        return

    fig_width = 500 / 72.27
    fig, axes = plt.subplots(1, 2, figsize=(fig_width, fig_width * 0.45), sharey=True)
    fig.supxlabel("Bitflip Iteration", fontsize=10, y=0.08)
    fig.supylabel("Perplexity", fontsize=10, x=0.04)
    fig.subplots_adjust(left=0.12, right=0.99, bottom=0.22, top=0.86, wspace=0.0)

    # Keep colors consistent across both subplots.
    grad_color = "tab:blue"
    hess_color = "tab:orange"

    model_plots = [
        (
            axes[0],
            "OLMoE",
            olmoe_grad_iterations,
            olmoe_grad_perplexities,
            olmoe_hess_iterations,
            olmoe_hess_perplexities,
        ),
        (
            axes[1],
            "Qwen",
            qwen_grad_iterations,
            qwen_grad_perplexities,
            qwen_hess_iterations,
            qwen_hess_perplexities,
        ),
    ]

    for ax, model_name, grad_iterations, grad_perplexities, hess_iterations, hess_perplexities in model_plots:
        x_grad = list(range(len(grad_iterations)))
        x_hess = list(range(len(hess_iterations)))

        ax.plot(
            x_grad,
            grad_perplexities,
            marker="o",
            linestyle="-",
            linewidth=1.5,
            markersize=4,
            color=grad_color,
            label="Gradient",
        )
        ax.plot(
            x_hess,
            hess_perplexities,
            marker="s",
            linestyle="-",
            linewidth=1.5,
            markersize=4,
            color=hess_color,
            label="Hessian",
        )

        ax.set_title(f"{model_name} Perplexity Over Bitflips", fontsize=9, fontweight="bold")
        ax.set_yscale("log")
        ax.yaxis.set_major_locator(mticker.LogLocator(base=10.0, subs=(1.0,)))
        ax.yaxis.set_minor_locator(mticker.NullLocator())
        ax.set_xticks(x_grad)
        ax.set_xticklabels(grad_iterations, rotation=0, ha="center")
        ax.margins(x=0.08)
        ax.tick_params(axis="both", labelsize=8)
        ax.grid(True, which="major", alpha=0.3)

        if(ax == axes[0]):
            ax.annotate(
                f"Initial Perplexity: {grad_perplexities[0]:.2f}",
                (x_grad[0], grad_perplexities[0]),
                textcoords="offset points",
                xytext=(37.5, 125),
                ha="center",
                fontsize=8,
            )
        else:
            ax.annotate(
            f"Initial Perplexity: {grad_perplexities[0]:.2f}",
            (x_grad[0], grad_perplexities[0]),
            textcoords="offset points",
            xytext=(54, 122),
            ha="center",
            fontsize=8,
        )

    axes[1].legend(loc="upper right", frameon=False)

    # Keep the outside frame while removing only the shared seam.
    axes[0].spines["right"].set_visible(False)
    axes[1].spines["left"].set_visible(False)
    axes[1].tick_params(left=False, labelleft=False)

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Figure saved to {output_path}")
    plt.show()


if __name__ == "__main__":
    print("Plotting OLMoE and Qwen with gradient+hessian overlays...")
    plot_perplexity_side_by_side(
        olmoe_grad_csv_path="olmoe_grad.csv",
        olmoe_hess_csv_path="olmoe_hess.csv",
        qwen_grad_csv_path="qwen_grad.csv",
        qwen_hess_csv_path="qwen_hess.csv",
        output_path="perplexity_side_by_side.pdf",
    )
