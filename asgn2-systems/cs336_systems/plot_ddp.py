from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


RESULTS_DIR = Path(__file__).resolve().parent / "results"
df = pd.read_csv(RESULTS_DIR / "allreduce.csv")

plots = [
    ("median_ms", "Median Latency (ms)", "latency.png"),
    ("throughput_gbps", "Effective Throughput (GB/s)", "throughput.png"),
]

for metric, ylabel, filename in plots:

    fig, ax = plt.subplots(figsize=(8, 5))

    for world_size, group in df.groupby("world_size"):

        group = group.sort_values("size_mb")

        ax.plot(
            group["size_mb"],
            group[metric],
            marker="o",
            label=f"{world_size} GPUs",
        )

    ax.set_xscale("log")
    if metric == "median_ms":
        ax.set_yscale("log")
    ax.set_xlabel("Tensor Size (MB)")
    ax.set_ylabel(ylabel)
    ax.set_title(f"All-Reduce {ylabel}")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(RESULTS_DIR / filename, dpi=200)
    plt.close(fig)

print("Plots saved to:", RESULTS_DIR)