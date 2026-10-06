"""Figures for the be_s5_library (follow-up library candidates, no FDR filter).

Run scripts/be_s5_library_analysis.py first; this reads results/be_s5_library/.
  1. library_composition.png              guides per site type, split by essential subset / label
  2. guides_vs_s5_shakeoff_{start,stop}.png  guide LFC (BE) vs gene's 78h shake-off LFC (S5)
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from be_screen_analysis.plotting import GRAY, INK, MUTED, style

ROOT = Path(__file__).resolve().parents[2]
LIBRARY = ROOT / "results" / "be_s5_library" / "be_s5_library_guides.csv"
OUT_DIR = ROOT / "figures" / "be_s5_library"

TYPES = ["Annotated", "Extended", "Truncated", "Mixed", "Stop"]
# sequential blue steps for the essential subset (darker = stronger evidence), gray otherwise
GROUPS = [
    ("Essential, dropout_consistent", "#1c5cab"),
    ("Essential, no clear dropout", "#86b6ef"),
    ("Not in S1 essential subset", GRAY),
]


def _group(lib: pd.DataFrame) -> np.ndarray:
    return np.select(
        [
            lib["s5_in_s1_essential"] & (lib["s5_dropout_consistent"] == True),  # noqa: E712
            lib["s5_in_s1_essential"],
        ],
        [GROUPS[0][0], GROUPS[1][0]],
        default=GROUPS[2][0],
    )


def library_composition(lib: pd.DataFrame) -> Path:
    site_type = lib["site_type"].where(~lib["site_type"].str.contains(";"), "Mixed")
    group = _group(lib)
    counts = pd.crosstab(site_type, group).reindex(index=TYPES, columns=[g for g, _ in GROUPS])
    counts = counts.fillna(0).astype(int)

    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=200)
    x = np.arange(len(TYPES))
    bottom = np.zeros(len(TYPES))
    for name, color in GROUPS:
        vals = counts[name].to_numpy()
        ax.bar(
            x,
            vals,
            bottom=bottom,
            width=0.62,
            color=color,
            edgecolor="white",
            linewidth=1,
            label=name,
        )
        bottom += vals
    for xi, total in zip(x, bottom):
        ax.annotate(
            f"{int(total):,}",
            (xi, total),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            fontsize=8,
            color=INK,
        )
    ax.set_xticks(
        x,
        ["Annotated\nATG", "Extended\n(nc)", "Truncated\n(nc)", "Mixed\ntype", "Stop"],
        fontsize=8,
    )
    ax.set_ylabel("Guides (sensor efficiency ≥20% at D0 or D12)")
    ax.set_title(
        f"be_s5_library composition ({len(lib):,} guides, {lib['Hugo_Symbol'].nunique():,} genes)",
        loc="left",
        fontsize=10,
    )
    style(ax)
    ax.grid(axis="x", visible=False)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    fig.tight_layout()
    out = OUT_DIR / "library_composition.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def guides_vs_s5(lib: pd.DataFrame, screen: str) -> Path:
    g = lib[lib["screen"] == screen].assign(group=lambda d: _group(d))
    rho = g[["s5_shakeoff_lfc", "guide_lfc"]].corr(method="spearman").iloc[0, 1]

    fig, ax = plt.subplots(figsize=(6.5, 5), dpi=200)
    for name, color in reversed(GROUPS):  # gray first so the essential groups sit on top
        sub = g[g["group"] == name]
        ax.scatter(
            sub["s5_shakeoff_lfc"],
            sub["guide_lfc"],
            s=6,
            color=color,
            alpha=0.5 if color == GRAY else 0.7,
            linewidths=0,
            label=f"{name} (n={len(sub):,})",
        )
    ax.axhline(0, color=MUTED, linewidth=0.8)
    ax.axvline(0, color=MUTED, linewidth=0.8)
    ax.set_xlabel("S5 HeLa 78h shake-off vs remaining, log2FC (gene)\n(> 0: knockout cells detach)")
    ax.set_ylabel(f"ABE {screen} codon guide, log2FC D12 vs D0")
    ax.set_title(
        f"{screen.capitalize()} guides vs S5 shake-off "
        f"({g['Hugo_Symbol'].nunique():,} genes; Spearman ρ = {rho:.2f})",
        loc="left",
        fontsize=10,
    )
    style(ax)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles[::-1], labels[::-1], frameon=False, fontsize=8, loc="lower left", markerscale=2.5
    )
    fig.tight_layout()
    out = OUT_DIR / f"guides_vs_s5_shakeoff_{screen}.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    lib = pd.read_csv(LIBRARY)
    print(library_composition(lib))
    for screen in ("start", "stop"):
        print(guides_vs_s5(lib, screen))


if __name__ == "__main__":
    main()
