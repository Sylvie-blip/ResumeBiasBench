"""Figure 10 — forest plot of AI preference rate by occupational category.

Source: results/qual_audit/category_breakdown.csv (already computed:
cluster-robust rate and 95% CI per category, clustered on pair_id across all
7 available models' decisions - reused here rather than recomputed by a
different method, per the brief's CI ground rule).

BPO (17 pairs) and AUTOMOBILE (28 pairs) are underpowered (<30 pairs) and are
drawn with open/lighter markers rather than omitted, so their existence and
their imprecision are both visible.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import figstyle  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]

CAPTION = (
    "Figure 10. AI-polished preference rate by occupational category (24 "
    "categories), pooled across all 7 available models' decisions, "
    "cluster-robust 95% CI (clustered on pair_id) from results/qual_audit/"
    "category_breakdown.csv. Sorted by point estimate; dashed line marks "
    "parity (0.5). BPO (17 pairs) and AUTOMOBILE (28 pairs), open markers, "
    "are underpowered (<30 pairs) - shown rather than dropped so the reader "
    "sees both that they exist and that they are not being leaned on. Every "
    "one of the 22 adequately-powered categories has a 95% CI entirely "
    "above 0.5; the lowest, TEACHER at 0.737, is still comfortably clear of "
    "parity."
)


def main() -> None:
    figstyle.apply_style()

    # Source: results/qual_audit/category_breakdown.csv
    df = pd.read_csv(REPO_ROOT / "results" / "qual_audit" / "category_breakdown.csv")
    df = df.sort_values("rate").reset_index(drop=True)

    fig, (ax, ax_n) = plt.subplots(
        1, 2, figsize=(figstyle.FULL_WIDTH, 7.0),
        gridspec_kw={"width_ratios": [5, 1]},
    )

    y_pos = range(len(df))
    for y, row in zip(y_pos, df.itertuples()):
        underpowered = bool(row.underpowered)
        if underpowered:
            ax.plot([row.ci_lo, row.ci_hi], [y, y], color=figstyle.AI_COLOR,
                     linewidth=1.4, alpha=0.55, zorder=2, solid_capstyle="round")
            ax.plot([row.rate], [y], marker=figstyle.AI_MARKER, linestyle="none",
                     markerfacecolor="white", markeredgecolor=figstyle.AI_COLOR,
                     markeredgewidth=1.3, markersize=figstyle.MARKER_SIZE, zorder=3)
        else:
            ax.plot([row.ci_lo, row.ci_hi], [y, y], color=figstyle.AI_COLOR,
                     linewidth=2.0, zorder=2, solid_capstyle="round")
            ax.plot([row.rate], [y], marker=figstyle.AI_MARKER, linestyle="none",
                     zorder=3, **figstyle.ring_kwargs(figstyle.AI_COLOR))

    ax.axvline(0.5, color=figstyle.PARITY_COLOR, linewidth=1.2, linestyle=(0, (4, 3)), zorder=1)
    ax.text(0.5, -0.65, "parity", color=figstyle.PARITY_COLOR, fontsize=7.5,
             ha="center", va="top", clip_on=False)

    ax.set_yticks(list(y_pos))
    labels = [
        f"{cat}{' *' if under else ''}"
        for cat, under in zip(df["category"], df["underpowered"])
    ]
    ax.set_yticklabels(labels)
    ax.set_ylim(-0.8, len(df) - 0.1)
    ax.set_xlim(0.45, 0.90)
    ax.set_xlabel("AI-polished preference rate")
    figstyle.style_grid_x(ax)

    for y, row in zip(y_pos, df.itertuples()):
        ax_n.text(0, y, f"{row.n_pairs}", va="center", ha="left", fontsize=7.5,
                   color=figstyle.INK_SECONDARY)
    ax_n.set_xlim(0, 1)
    ax_n.set_ylim(ax.get_ylim())
    ax_n.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    for spine in ax_n.spines.values():
        spine.set_visible(False)
    ax_n.set_title("n pairs", fontsize=7.5, color=figstyle.INK_SECONDARY, loc="left")

    fig.text(0.02, 0.005, "* underpowered (<30 pairs): BPO, AUTOMOBILE",
              fontsize=7, color=figstyle.INK_SECONDARY, ha="left")

    fig.tight_layout(rect=(0, 0.02, 1, 1))
    out_path = REPO_ROOT / "data" / "figures" / "category_forest"
    figstyle.save_fig(fig, out_path)
    print(f"Saved {out_path}.png / .pdf")
    print(df[["category", "n_pairs", "rate", "ci_lo", "ci_hi", "underpowered"]].to_string())


if __name__ == "__main__":
    main()
