"""How much of each fitness subset is present, and shared, in the other screens.

fitness_subset_coverage.png  rows: the S1, S5 and ABE start fitness subsets (240 genes each). For
  each other screen, two columns: present in that screen (of the 240), then shared = in that
  screen's fitness subset (of those present); n/a = the subset's own screen. Present in ABE =
  annotated start targeted with >=20% editing (the pool the ABE subset is drawn from).

Also writes results/old_analysis/ABE_validation/fitness_subset/fitness_subset_presence.csv
(per gene: subset memberships and presence in each screen).

Run fitness_subset_analysis.py and scripts/abe_mageck_analysis.py first.
"""

import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from be_screen_analysis.correlations import abe_gene_lfc, s5_gene_dropout
from be_screen_analysis.filtering import s1_published_gene_lfc
from be_screen_analysis.plotting import INK, MUTED

ROOT = Path(__file__).resolve().parents[4]
MAGECK_DIR = ROOT / "results" / "abe_mageck"
S1_MAGECK = ROOT / "data" / "shakeoff_data" / "table_s1" / "mageck.neg_HeLa.csv"
S5_FILE = ROOT / "data" / "shakeoff_data" / "table_s5" / "sgRNA_sequences_and_counts.csv"
LIST_DIR = ROOT / "figures" / "old_analysis" / "ABE_validation" / "fitness_subset"
OUT_DIR = ROOT / "results" / "old_analysis" / "ABE_validation" / "fitness_subset"

DAY = "D4"


SCREENS = ["S1", "S5", "ABE"]
PRESENT = {"S1": "in S1", "S5": "in S5", "ABE": "in ABE\n(\u226520% editing)"}


def subsets() -> dict[str, set[str]]:
    """The three fitness subsets, keyed by screen."""
    read = lambda name: set(pd.read_csv(LIST_DIR / name, comment="#", index_col=0).index)  # noqa: E731
    return {
        "S1": read("s1_d4_fitness_subset.csv"),
        "S5": read("s5_fitness_subset.csv"),
        "ABE": read("abe_start_fitness_subset.csv"),
    }


def presence(sets: dict[str, set[str]], present: dict[str, set[str]]) -> pd.DataFrame:
    """Per gene of any subset: membership of each subset and presence in each condition."""
    genes = sorted(set().union(*sets.values()))
    table = pd.DataFrame(index=pd.Index(genes, name="Hugo_Symbol"))
    for k in sets:
        table[f"in_{k}_fitness_subset"] = table.index.isin(sets[k])
    for k in sets:
        table[f"present_in_{k}"] = table.index.isin(present[k])
    return table.rename(columns=lambda c: c.replace(" ", "_"))


def coverage(sets: dict[str, set[str]], pres: pd.DataFrame, labels: dict[str, str]) -> pd.DataFrame:
    """Per subset (rows) and other condition: present (of the subset), then shared (of present).

    ``labels`` gives the "present" column name of each condition.
    """
    key = lambda k: k.replace(" ", "_")  # noqa: E731
    cols, num, den = [], {}, {}
    for k in sets:
        cols += [labels[k], f"in {k} fitness\nsubset (of present)"]
    for row in sets:
        sub = pres.loc[sorted(sets[row])]
        n, d = [], []
        for k in sets:
            if k == row:  # a subset's own condition: present and in its subset by definition
                n += [np.nan, np.nan]
                d += [np.nan, np.nan]
                continue
            here = sub[f"present_in_{key(k)}"]
            n += [here.sum(), (here & sub[f"in_{key(k)}_fitness_subset"]).sum()]
            d += [len(sub), here.sum()]
        name = f"{row} fitness subset ({len(sub)})"
        num[name], den[name] = n, d
    n = pd.DataFrame(num, index=cols).T
    d = pd.DataFrame(den, index=cols).T
    table = n / d
    table.attrs["num"], table.attrs["den"] = n, d
    return table


def heatmap(ax: plt.Axes, table: pd.DataFrame, title: str, note: str = "") -> None:
    num, den = table.attrs["num"], table.attrs["den"]
    ax.imshow(table.to_numpy(), cmap="Blues", vmin=0, vmax=1, aspect="auto")
    for i in range(table.shape[0]):
        for j in range(table.shape[1]):
            v = table.iat[i, j]
            if np.isnan(v):
                ax.text(j, i, "n/a", ha="center", va="center", fontsize=8, color=MUTED)
                continue
            text = f"{v:.0%}\n{int(num.iat[i, j]):,}/{int(den.iat[i, j]):,}"
            ax.text(j, i, text, ha="center", va="center", fontsize=8,
                    color="white" if v > 0.6 else INK)  # fmt: skip
    for x in np.arange(1.5, table.shape[1] - 1, 2):  # separate each condition's column pair
        ax.axvline(x, color="white", linewidth=4)
    ax.set_xticks(range(table.shape[1]), table.columns, fontsize=8)
    ax.set_yticks(range(table.shape[0]), table.index, fontsize=9)
    ax.tick_params(length=0)
    for side in ax.spines.values():
        side.set_visible(False)
    if note:
        ax.text(0, -0.2, note, transform=ax.transAxes, fontsize=7, color=MUTED, va="top")
    ax.set_title(title, loc="left", fontsize=10)


def main() -> None:
    sets = subsets()
    lfc = {
        "S1": s1_published_gene_lfc(pd.read_csv(S1_MAGECK)),
        "S5": s5_gene_dropout(pd.read_csv(S5_FILE)),
        "ABE": abe_gene_lfc(MAGECK_DIR, "start"),
    }
    # present in ABE = annotated start targeted with >=20% editing (the ABE subset's pool)
    present = {
        "S1": set(lfc["S1"].index),
        "S5": set(lfc["S5"].index),
        "ABE": set(lfc["ABE"][f"ABE {DAY}"].dropna().index),
    }
    pres = presence(sets, present)
    pres.to_csv(OUT_DIR / "fitness_subset_presence.csv")
    table = coverage(sets, pres, PRESENT)
    print((table * 100).round(0).to_string())
    absent = list(pres.index[pres["in_S1_fitness_subset"] & ~pres["present_in_S5"]])
    note = (
        textwrap.fill(
            f"S1 fitness subset genes absent from S5 ({len(absent)}): " + ", ".join(absent), 120
        )
        + f"\nAll absent genes per subset and screen: {OUT_DIR.relative_to(ROOT)}/"
        "fitness_subset_presence.csv"
    )
    fig, ax = plt.subplots(figsize=(9.5, 5.4), dpi=200)
    heatmap(
        ax,
        table,
        "For each other screen: is the gene present, then of those present, is it shared\n"
        "(in that screen's fitness subset); n/a = the subset's own screen",
        note,
    )
    fig.tight_layout()
    out = LIST_DIR / "fitness_subset_coverage.png"
    fig.savefig(out)
    plt.close(fig)
    print(out)


if __name__ == "__main__":
    main()
