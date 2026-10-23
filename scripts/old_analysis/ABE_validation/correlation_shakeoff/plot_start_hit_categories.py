"""Tyler's start hit categories (|LFC| > 1) marked on the ABE start D4 vs S1 / S5 correlation.

Categories as in notebooks/start_scatter.ipynb (FDR ignored, |LFC| > 1 in either direction, site
LFC D12 vs D0): annotated only, non-canonical only, both, neither. Every gene of the D4
correlation is shown. Annotated side: |D12 site LFC| > 1 at any of the gene's annotated starts,
from the MAGeCK rerun (results/abe_mageck/start_guide_summary_mageck.csv; identical to the start
codon summary's annot_lfc, and also covers genes the summary has no pair for). Non-canonical
side, per figure: any of the gene's extended (or truncated) starts in the start codon summary
with |nc_lfc| > 1; a gene with no such start cannot be a non-canonical hit.

start_hit_categories_{extended,truncated}_d4.png  one figure per non-canonical type, two panels:
                             ABE annotated start D4 vs S1 D4 and vs S5 78h, every gene as in
                             abe_start_vs_s1_s5_d4_lfc.png (same points and fit), coloured by
                             category (the notebook's colours, yellow brightened, more opaque).

{extended,truncated}_start_vs_s1_s5_d4_lfc.png  the same, but the y axis is the extended (or
                             truncated) start's ABE D4 LFC (median of efficient guides at the
                             gene's starts of that type); genes with such a start only.

Also writes results/old_analysis/ABE_validation/correlation_shakeoff/start_hit_categories.csv
(one row per gene: annotated hit, extended / truncated hit and start presence, category per
type). Run scripts/abe_mageck_analysis.py first.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from be_screen_analysis.correlations import abe_gene_lfc, s5_gene_dropout
from be_screen_analysis.filtering import s1_published_gene_lfc
from be_screen_analysis.plotting import scatter_fit

ROOT = Path(__file__).resolve().parents[4]
MAGECK_DIR = ROOT / "results" / "abe_mageck"
SUMMARY = ROOT / "data" / "start_stop_ABE_screen_data" / "summaries" / "start_codon_summary.csv"
S1_DIR = ROOT / "data" / "shakeoff_data" / "table_s1"
S5_DIR = ROOT / "data" / "shakeoff_data" / "table_s5"
OUT_DIR = ROOT / "figures" / "old_analysis" / "ABE_validation" / "correlation_shakeoff"
RES_DIR = ROOT / "results" / "old_analysis" / "ABE_validation" / "correlation_shakeoff"

LFC_THRESH = 1
NC_TYPES = ["Extended", "Truncated"]
# notebook colours (start_scatter.ipynb); "neither" first so the hits are drawn on top
CATEGORIES = {
    "neither": "#CCCCCC",
    "annotated only": "#4477AA",
    "non-canonical only": "#F2D21B",  # notebook yellow #CCBB44, brightened
    "both": "#EE6677",
}
POINT_ALPHA = 0.8


def hit_categories(guides: pd.DataFrame, summary: pd.DataFrame) -> pd.DataFrame:
    """Per gene: annotated hit, per non-canonical type presence / hit, and the category per type."""
    annot = guides[guides["start_site_type"] == "Annotated"].drop_duplicates("start_site_target_id")
    out = pd.DataFrame(
        {"annot_hit": (annot["mageck_D12_site_lfc"].abs() > LFC_THRESH)
         .groupby(annot["Hugo_Symbol"]).any()}
    )  # fmt: skip
    for nc_type in NC_TYPES:
        nc = summary[summary["nc_start_site_type"] == nc_type]
        hit = (nc["nc_lfc"].abs() > LFC_THRESH).groupby(nc["Hugo_Symbol"]).any()
        key = nc_type.lower()
        out[f"has_{key}"] = out.index.isin(hit.index)
        out[f"{key}_hit"] = hit.reindex(out.index, fill_value=False).astype(bool)
        cat = pd.Series("neither", index=out.index)
        cat[out.annot_hit & ~out[f"{key}_hit"]] = "annotated only"
        cat[~out.annot_hit & out[f"{key}_hit"]] = "non-canonical only"
        cat[out.annot_hit & out[f"{key}_hit"]] = "both"
        out[f"category_{key}"] = cat
    return out.rename_axis("Hugo_Symbol")


def categories_on_correlation(
    cats: pd.DataFrame, abe: pd.Series, screens: dict, nc_type: str, start: str, out_name: str
) -> Path:
    """Two panels (S1, S5) for one non-canonical type, genes coloured by hit category.

    ``abe`` is the gene D4 LFC of the ``start`` sites ("annotated", "extended", "truncated").
    """
    cat = cats[f"category_{nc_type.lower()}"].rename("category")
    fig, axes = plt.subplots(1, 2, figsize=(12, 6.2), dpi=200)
    for ax, (screen, (ko, ko_day)) in zip(axes, screens.items()):
        d = pd.concat([ko.rename("ko"), abe.rename("abe"), cat], axis=1, join="inner").dropna()
        scatter_fit(ax, d["ko"], d["abe"], groups=d["category"], group_colors=CATEGORIES,
                    alpha=POINT_ALPHA)  # fmt: skip
        ax.legend(frameon=False, fontsize=7, loc="upper left", markerscale=1.5)
        ref = "published MAGeCK" if screen == "S1" else "intergenic-centred"
        ax.set_xlabel(f"{screen} CRISPR KO {ko_day}, log2FC ({ref})", fontsize=9)
        ax.set_ylabel(f"ABE {start} start D4, log2FC vs D0 (MAGeCK)", fontsize=9)
        ax.set_title(f"ABE {start} start D4 vs {screen} {ko_day} ({nc_type.lower()} categories)",
                     loc="left", fontsize=10)  # fmt: skip
        print(nc_type, screen, d["category"].value_counts().to_dict())
    fig.suptitle(
        f"Hit categories (|LFC| > {LFC_THRESH}, D12 site LFC), non-canonical side = "
        f"{nc_type.lower()} starts (genes without one cannot be a non-canonical hit); "
        "blue line: fit over all genes",
        x=0.01,
        ha="left",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = OUT_DIR / out_name
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    RES_DIR.mkdir(parents=True, exist_ok=True)
    guides = pd.read_csv(MAGECK_DIR / "start_guide_summary_mageck.csv")
    cats = hit_categories(guides, pd.read_csv(SUMMARY))
    cats.to_csv(RES_DIR / "start_hit_categories.csv")
    for nc_type in NC_TYPES:
        print(nc_type, cats[f"category_{nc_type.lower()}"].value_counts().to_dict())
    s1 = s1_published_gene_lfc(pd.read_csv(S1_DIR / "mageck.neg_HeLa.csv"))
    s5 = s5_gene_dropout(pd.read_csv(S5_DIR / "sgRNA_sequences_and_counts.csv"))
    screens = {"S1": (s1["D4"], "D4"), "S5": (s5["S5 78h"], "78h")}
    annot = abe_gene_lfc(MAGECK_DIR, "start")["ABE D4"]
    for nc_type in NC_TYPES:
        key = nc_type.lower()
        nc = abe_gene_lfc(MAGECK_DIR, "start", site_type=nc_type)["ABE D4"]
        print(categories_on_correlation(cats, annot, screens, nc_type, "annotated",
                                        f"start_hit_categories_{key}_d4.png"))  # fmt: skip
        print(categories_on_correlation(cats, nc, screens, nc_type, key,
                                        f"{key}_start_vs_s1_s5_d4_lfc.png"))  # fmt: skip


if __name__ == "__main__":
    main()
