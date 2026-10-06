"""Shared plotting for the BE x shake-off screen comparisons (S1, S5)."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

BLUE, ORANGE, GRAY = "#2a78d6", "#eb6834", "#b4b3ad"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e5e5e2"


def style(ax: plt.Axes) -> None:
    """Recessive grid and no top/right spines."""
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def _label(
    ax: plt.Axes, df: pd.DataFrame, names: pd.Series, dx: float = 0.35, dy: float = 0.2
) -> None:
    # a label within (dx, dy) data units of an already placed one goes below-right instead
    placed: list[tuple[float, float]] = []
    for (x, y), name in zip(df[["x", "be"]].to_numpy(), names):
        crowded = any(abs(x - px) < dx and abs(y - py) < dy for px, py in placed)
        ax.annotate(
            name,
            (x, y),
            xytext=(4, -9) if crowded else (4, 2),
            textcoords="offset points",
            fontsize=7,
            color=INK,
        )
        placed.append((x, y))


def _finish(ax: plt.Axes, xlabel: str, ylabel: str, title: str, legend_loc: str) -> None:
    ax.axhline(0, color=MUTED, linewidth=0.8)
    ax.axvline(0, color=MUTED, linewidth=0.8)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", fontsize=10)
    style(ax)
    ax.legend(frameon=False, fontsize=8, loc=legend_loc)


def _save(fig: plt.Figure, out: Path) -> Path:
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out


def be_vs_screen_start(
    be_dir: Path,
    screen_lfc: pd.Series,
    hits: pd.DataFrame,
    xlabel: str,
    out: Path,
    n_labels: int = 8,
) -> Path:
    """Annotated ATG LFC (BE) vs a shake-off screen LFC, one point per gene."""
    start = pd.read_csv(be_dir / "start_codon_summary.csv")
    be = start.drop_duplicates("annot_start_site_target_id").groupby("Hugo_Symbol")["annot_lfc"]
    df = pd.concat([be.first().rename("be"), screen_lfc.rename("x")], axis=1, join="inner")
    df["hit"] = df.index.isin(hits.loc[hits["site_type"] == "Annotated", "Hugo_Symbol"])
    rho = df[["be", "x"]].corr(method="spearman").iloc[0, 1]

    fig, ax = plt.subplots(figsize=(6.5, 5), dpi=200)
    rest, hit = df[~df["hit"]], df[df["hit"]]
    ax.scatter(
        rest["x"],
        rest["be"],
        s=10,
        color=GRAY,
        alpha=0.6,
        linewidths=0,
        label=f"Genes in both screens (n={len(df):,})",
    )
    ax.scatter(
        hit["x"],
        hit["be"],
        s=28,
        color=BLUE,
        edgecolors="white",
        linewidths=0.8,
        label=f"Intersection, annotated ATG (n={len(hit)})",
    )
    top = hit.nlargest(n_labels, "x")
    _label(ax, top, top.index)
    _finish(
        ax,
        xlabel,
        "ABE annotated start codon, log2FC D12 vs D0",
        f"Start codon loss vs shake-off enrichment (Spearman ρ = {rho:.2f})",
        "lower left",
    )
    return _save(fig, out)


def be_vs_screen_stop(
    be_dir: Path,
    screen_lfc: pd.Series,
    hits: pd.DataFrame,
    be_filtered: pd.DataFrame,
    screen_name: str,
    xlabel: str,
    out: Path,
) -> Path:
    """Stop codon LFC (BE) vs a shake-off screen LFC, one point per stop site."""
    stop = pd.read_csv(be_dir / "stop_codon_summary.csv")
    df = stop[["Hugo_Symbol", "stop_site_target_id", "lfc"]].rename(columns={"lfc": "be"})
    df = df.merge(screen_lfc.rename("x"), left_on="Hugo_Symbol", right_index=True)
    sig = be_filtered.loc[be_filtered["screen"] == "stop", "site_id"]
    df["hit"] = df["stop_site_target_id"].isin(hits.loc[hits["screen"] == "stop", "site_id"])
    df["be_only"] = df["stop_site_target_id"].isin(sig) & ~df["hit"]
    rho = df[["be", "x"]].corr(method="spearman").iloc[0, 1]

    fig, ax = plt.subplots(figsize=(6.5, 5), dpi=200)
    rest = df[~df["hit"] & ~df["be_only"]]
    ax.scatter(
        rest["x"],
        rest["be"],
        s=10,
        color=GRAY,
        alpha=0.6,
        linewidths=0,
        label=f"Stop sites, gene in both screens (n={len(df):,})",
    )
    for mask, color, label in [
        (df["be_only"], ORANGE, f"ABE-significant, fails {screen_name} filter"),
        (df["hit"], BLUE, "Intersection"),
    ]:
        sub = df[mask]
        ax.scatter(
            sub["x"],
            sub["be"],
            s=28,
            color=color,
            edgecolors="white",
            linewidths=0.8,
            label=f"{label} (n={len(sub)})",
        )
        _label(ax, sub, sub["Hugo_Symbol"])
    _finish(
        ax,
        xlabel,
        "ABE stop codon, log2FC D12 vs D0",
        f"Stop codon loss vs shake-off enrichment (Spearman ρ = {rho:.2f})",
        "upper right",
    )
    return _save(fig, out)
