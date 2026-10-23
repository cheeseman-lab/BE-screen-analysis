"""Annotate every gene in the ABE screen with DepMap CRISPR dependency (dependent cell lines).

Writes results/abe_depmap/:
  abe_genes_depmap.csv              one row per ABE gene: which ABE libraries target it (start,
                                    stop, in-screen KO) and depmap_n_dependent / _n_screened /
                                    _pct_dependent / _fraction ("732/808")
  {start,stop}_guide_summary_depmap.csv, {start,stop}_ko_guides_depmap.csv
                                    the MAGeCK guide tables with the same DepMap columns joined
                                    on by gene

Dependent = probability of dependency > 0.5 in data/reference/CRISPRGeneDependency.csv (the DepMap
portal's definition); genes are matched by exact HUGO symbol (DepMap column names minus the
Entrez ID). Run scripts/abe_mageck_analysis.py first; this reads results/abe_mageck/.
"""

from pathlib import Path

import pandas as pd

from be_screen_analysis.depmap import depmap_dependency

ROOT = Path(__file__).resolve().parents[1]
DEPMAP_FILE = ROOT / "data" / "reference" / "CRISPRGeneDependency.csv"
MAGECK_DIR = ROOT / "results" / "abe_mageck"
OUT_DIR = ROOT / "results" / "abe_depmap"
TABLES = {
    "start": "start_guide_summary_mageck.csv",
    "stop": "stop_guide_summary_mageck.csv",
    "start_ko": "start_ko_guides_mageck.csv",
    "stop_ko": "stop_ko_guides_mageck.csv",
}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    depmap = depmap_dependency(DEPMAP_FILE)
    counts = ["depmap_n_dependent", "depmap_n_screened"]
    depmap[counts] = depmap[counts].astype("Int64")  # stay whole numbers after NaN-padding joins
    print(f"DepMap: {len(depmap):,} genes, up to {depmap['depmap_n_screened'].max():,} lines")
    libraries = {}
    for name, file in TABLES.items():
        guides = pd.read_csv(MAGECK_DIR / file)
        out = guides.join(depmap, on="Hugo_Symbol")
        out.to_csv(OUT_DIR / file.replace("_mageck.csv", "_depmap.csv"), index=False)
        libraries[name] = set(guides["Hugo_Symbol"].dropna())

    genes = pd.Index(sorted(set().union(*libraries.values())), name="Hugo_Symbol")
    table = pd.DataFrame(
        {
            "in_start_library": genes.isin(libraries["start"]),
            "in_stop_library": genes.isin(libraries["stop"]),
            "in_ko_library": genes.isin(libraries["start_ko"] | libraries["stop_ko"]),
        },
        index=genes,
    ).join(depmap)
    table.round({"depmap_pct_dependent": 2}).to_csv(OUT_DIR / "abe_genes_depmap.csv")

    found = table["depmap_n_screened"].notna()
    print(f"ABE genes: {len(table):,}; matched in DepMap: {found.sum():,}")
    print("not in DepMap:", list(table.index[~found]))
    for lib in ("in_start_library", "in_stop_library", "in_ko_library"):
        pct = table.loc[table[lib] & found, "depmap_pct_dependent"]
        print(
            f"{lib}: {len(pct):,} genes, median % dependent {pct.median():.1f}, "
            f">=90% of lines: {(pct >= 90).sum():,}"
        )


if __name__ == "__main__":
    main()
