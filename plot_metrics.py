"""
Plot metrics from a training log folder.

Usage:
    python plot_metrics.py <log_folder>

The folder should contain a metrics.jsonl file with one JSON object per line,
each having a 'step' key plus arbitrary metric keys. One PNG per metric is
saved into a plots/ subdirectory inside the folder.
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt


def load_metrics(folder: Path) -> dict[str, list]:
    metrics_file = folder / "metrics.jsonl"
    if not metrics_file.exists():
        sys.exit(f"No metrics.jsonl found in {folder}")

    data: dict[str, list] = {}
    with metrics_file.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            for key, value in obj.items():
                data.setdefault(key, []).append(value)
    return data


def plot_metrics(folder: Path) -> None:
    data = load_metrics(folder)

    if "step" not in data:
        sys.exit("metrics.jsonl has no 'step' field")

    steps = data["step"]
    metric_keys = [k for k in data if k != "step"]

    if not metric_keys:
        sys.exit("No metrics other than 'step' found")

    plots_dir = folder / "plots"
    plots_dir.mkdir(exist_ok=True)

    for key in metric_keys:
        values = data[key]
        n = min(len(steps), len(values))

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(steps[:n], values[:n], linewidth=1.2)
        ax.set_title(key, fontsize=11)
        ax.set_xlabel("step", fontsize=9)
        ax.tick_params(labelsize=8)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        filename = key.replace("/", "_") + ".png"
        out_path = plots_dir / filename
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {out_path}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(f"Usage: python {Path(__file__).name} <log_folder>")

    folder = Path(sys.argv[1])
    if not folder.is_dir():
        sys.exit(f"Not a directory: {folder}")

    plot_metrics(folder)
