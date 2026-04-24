"""Visualization utilities for bitflip experiments."""

import csv
import math
import matplotlib.pyplot as plt


def plot_perplexity_over_flips(csv_path: str = "bitflip_metadata.csv", title: str = "Perplexity Over Bitflips") -> None:
    """Plot perplexity (= exp(loss)) at each bitflip iteration.
    
    Reads the CSV output from gate_grad_bit_rank or gate_hess_bit_rank and plots
    the perplexity trajectory, starting with initial perplexity and showing the
    perplexity after each of the n bitflips.
    
    Args:
        csv_path: Path to the bitflip metadata CSV file.
        title: Title for the plot.
    """
    iterations = []
    perplexities = []
    
    try:
        with open(csv_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
    except FileNotFoundError:
        print(f"Error: {csv_path} not found.")
        return
    
    if not rows:
        print(f"Error: {csv_path} is empty.")
        return
    
    # Extract initial perplexity from the first row's base_loss.
    initial_loss = float(rows[0]["base_loss"])
    initial_ppl = math.exp(initial_loss)
    iterations.append("Initial")
    perplexities.append(initial_ppl)
    
    # Extract perplexity after each flip.
    for i, row in enumerate(rows):
        loss_after_flip = float(row["loss_after_flip"])
        ppl_after_flip = math.exp(loss_after_flip)
        iterations.append(f"Flip {i+1}")
        perplexities.append(ppl_after_flip)
    
    # Create figure.
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x_pos = range(len(iterations))
    ax.plot(x_pos, perplexities, marker="o", linestyle="-", linewidth=2, markersize=8, color="blue")
    
    # Annotations.
    ax.set_xlabel("Bitflip Iteration", fontsize=12)
    ax.set_ylabel("Perplexity", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(iterations, rotation=45, ha="right")
    ax.grid(True, alpha=0.3)
    
    # Annotate each point with the value.
    for i, (x, y) in enumerate(zip(x_pos, perplexities)):
        ax.annotate(f"{y:.2f}", (x, y), textcoords="offset points", xytext=(0, 10), ha="center", fontsize=9)
    
    plt.tight_layout()
    plt.savefig("perplexity_over_flips.png", dpi=150, bbox_inches="tight")
    print(f"Figure saved to perplexity_over_flips.png")
    plt.show()


if __name__ == "__main__":
    # Example usage: plot from the default CSV files.
    print("Plotting gradient-based GBR results...")
    plot_perplexity_over_flips("bitflip_metadata.csv", title="Perplexity Over Bitflips (Gradient-Based GBR)")
    
    print("\nPlotting Hessian-based GBR results...")
    plot_perplexity_over_flips("bitflip_metadata_hess.csv", title="Perplexity Over Bitflips (Hessian-Based GBR)")
