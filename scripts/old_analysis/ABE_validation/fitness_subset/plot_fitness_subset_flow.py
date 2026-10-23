"""River (alluvial) plot of the three 240-gene fitness subsets across S1, S5 and ABE start.

fitness_subset_flow.png  three columns, S1 D4 -> S5 78h -> ABE annotated start D4, each split into
  the screen's fitness subset (240 genes) / other genes in that screen / not in that screen.
  Ribbons are coloured by which fitness subsets a gene belongs to. Genes in none of the subsets
  are drawn at 1/COMPRESS of their height; the largest boxes (>= BREAK_MIN genes) carry a break
  mark; labels give true counts. ABE "other genes" = the gene's annotated start is targeted by
  Tyler's start library but the gene is not in the ABE fitness subset.

Genes shown: every gene in S1 or S5 (the ABE library is inside S5). Also writes
results/old_analysis/ABE_validation/fitness_subset/fitness_subset_flow_counts.csv (genes per path).
Run fitness_subset_analysis.py and scripts/abe_mageck_analysis.py first.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from be_screen_analysis.correlations import s5_gene_dropout
from be_screen_analysis.filtering import s1_published_gene_lfc
from be_screen_analysis.plotting import BLUE, GRAY, INK, ORANGE, alluvial

ROOT = Path(__file__).resolve().parents[4]
MAGECK_DIR = ROOT / "results" / "abe_mageck"
S1_MAGECK = ROOT / "data" / "shakeoff_data" / "table_s1" / "mageck.neg_HeLa.csv"
S5_FILE = ROOT / "data" / "shakeoff_data" / "table_s5" / "sgRNA_sequences_and_counts.csv"
LIST_DIR = ROOT / "figures" / "old_analysis" / "ABE_validation" / "fitness_subset"
OUT_DIR = ROOT / "results" / "old_analysis" / "ABE_validation" / "fitness_subset"

COMPRESS = 20  # display scale for genes in none of the fitness subsets
BREAK_MIN = 2000  # only boxes with at least this many genes get a break mark
SCREENS = {"S1 D4": "S1", "S5 78h": "S5", "ABE start D4": "ABE"}
GROUP_COLORS = {
    "all three": "#3d3d3a",
    "S1 + S5": "#7b4fb8",
    "S1 + ABE": "#a8862b",
    "S5 + ABE": "#2a9d9d",
    "S1 only": ORANGE,
    "S5 only": BLUE,
    "ABE only": "#2e9e6b",
    "none": GRAY,
}


def gene_paths() -> pd.DataFrame:
    """One row per gene: category in each screen, fitness-subset group and display weight."""
    read = lambda name: set(pd.read_csv(LIST_DIR / name, comment="#", index_col=0).index)  # noqa: E731
    guides = pd.read_csv(MAGECK_DIR / "start_guide_summary_mageck.csv")
    subset = {
        "S1": read("s1_d4_fitness_subset.csv"),
        "S5": read("s5_fitness_subset.csv"),
        "ABE": read("abe_start_fitness_subset.csv"),
    }
    library = {
        "S1": set(s1_published_gene_lfc(pd.read_csv(S1_MAGECK)).index),
        "S5": set(s5_gene_dropout(pd.read_csv(S5_FILE)).index),
        "ABE": set(guides.loc[guides["start_site_type"] == "Annotated", "Hugo_Symbol"]),
    }
    genes = pd.Index(sorted(set().union(*library.values())), name="Hugo_Symbol")
    paths = pd.DataFrame(index=genes)
    for column, screen in SCREENS.items():
        paths[column] = f"not in {screen}"
        paths.loc[genes.isin(library[screen]), column] = f"other {screen} genes"
        paths.loc[genes.isin(subset[screen]), column] = f"{screen} fitness subset"
    member = pd.DataFrame({s: genes.isin(subset[s]) for s in subset}, index=genes)
    names = member.apply(lambda r: [s for s in subset if r[s]], axis=1)
    paths["group"] = names.map(
        lambda m: "none" if not m else "all three" if len(m) == 3 else
        f"{m[0]} only" if len(m) == 1 else " + ".join(m)
    )  # fmt: skip
    paths["weight"] = 1.0
    paths.loc[paths["group"] == "none", "weight"] = 1 / COMPRESS
    return paths


def flow_plot(paths: pd.DataFrame) -> Path:
    stages = {
        column: [f"{s} fitness subset", f"other {s} genes", f"not in {s}"]
        for column, s in SCREENS.items()
    }
    fig, ax = plt.subplots(figsize=(11, 7.5), dpi=200)
    alluvial(ax, paths, stages, "group", GROUP_COLORS, weight="weight", break_min=BREAK_MIN)
    counts = paths["group"].value_counts()
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=c, alpha=0.55, label=f"{g} ({counts.get(g, 0):,})")
        for g, c in GROUP_COLORS.items()
    ]
    ax.legend(handles=handles, frameon=False, fontsize=8, loc="upper center", ncol=4,
              bbox_to_anchor=(0.45, -0.01), title="ribbon colour: fitness subsets the gene is in",
              title_fontsize=8)  # fmt: skip
    fig.suptitle(
        f"Fitness subsets (240 genes each) across S1, S5 and the ABE start screen "
        f"(n = {len(paths):,} genes in S1 or S5)\n"
        f"Grey ribbons (genes in no fitness subset) are drawn {COMPRESS}x compressed everywhere; "
        "the largest boxes are marked with a break; labels give true counts",
        x=0.01,
        ha="left",
        fontsize=9,
        color=INK,
    )
    fig.tight_layout()
    out = LIST_DIR / "fitness_subset_flow.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    paths = gene_paths()
    counts = paths.drop(columns="weight").value_counts().rename("n_genes").sort_index()
    counts.to_csv(OUT_DIR / "fitness_subset_flow_counts.csv")
    print(paths["group"].value_counts().to_string())
    print(flow_plot(paths))


if __name__ == "__main__":
    main()
