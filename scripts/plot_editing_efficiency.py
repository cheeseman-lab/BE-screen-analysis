"""Plot sensor base editing efficiency distributions for the start/stop ABE screen.

Writes a single PNG with one panel per timepoint (D0, D12), each overlaying start- and stop-codon
guides.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "start_stop_ABE_screen_data"
OUT_DIR = ROOT / "figures"

COL = "sensor_mean_{day}_pct_successful_target_codon_mutation"
LIBRARIES = {"Start codon": ("start", "#2a78d6"), "Stop codon": ("stop", "#eb6834")}
BINS = np.arange(0, 102.5, 2.5)


DAYS = ("D0", "D12")


def plot_day(ax: plt.Axes, day: str, guides: dict[str, pd.DataFrame]) -> None:
    for label, (key, color) in LIBRARIES.items():
        vals = guides[key][COL.format(day=day)].dropna()
        weights = np.full(len(vals), 100 / len(vals))
        med = vals.median()
        ax.hist(
            vals,
            bins=BINS,
            weights=weights,
            histtype="stepfilled",
            color=color,
            alpha=0.25,
            linewidth=0,
        )
        ax.hist(
            vals,
            bins=BINS,
            weights=weights,
            histtype="step",
            color=color,
            linewidth=2,
            label=f"{label} (n={len(vals):,}, median {med:.1f}%)",
        )
        ax.axvline(med, color=color, linestyle="--", linewidth=1.2)

    ax.set_xlim(0, 100)
    ax.set_xlabel("Sensor editing efficiency (% successful target codon mutation)")
    ax.set_title(day, loc="left")
    ax.grid(axis="y", color="#e5e5e2", linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False, loc="upper left")


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    guides = {
        key: pd.read_csv(DATA_DIR / f"{key}_guide_summary.csv") for key, _ in LIBRARIES.values()
    }
    fig, axes = plt.subplots(1, len(DAYS), figsize=(12, 4.5), dpi=200, sharey=True)
    for ax, day in zip(axes, DAYS):
        plot_day(ax, day, guides)
    axes[0].set_ylabel("% of guides")
    fig.suptitle("ABE sensor editing efficiency", x=0.01, ha="left")
    fig.tight_layout()

    out = OUT_DIR / "editing_efficiency.png"
    fig.savefig(out)
    plt.close(fig)
    print(out)


if __name__ == "__main__":
    main()
