"""Figure 12 — per-pair length change (AI-polished minus human), tokens.

Source: results/qual_audit/length_check.csv, column delta_tokens (n=2,109
pairs; paper's own tokenizer, \\b[a-zA-Z']+\\b, lowercased). Median and mean
are computed directly from this column, not read from the paper's prose.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import figstyle  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]

CLIP_LOW = -1200
CLIP_HIGH = 250

CAPTION_TEMPLATE = (
    "Figure 12. Per-pair length change in tokens (AI-polished minus human), "
    "n=2,109 pairs, from results/qual_audit/length_check.csv (delta_tokens; "
    "paper's own tokenizer). Zero is marked with a vertical rule; the "
    "positive tail (AI-polished longer than human, {pos_pct:.1f}% of pairs) "
    "is shaded. Median {median:.1f}, mean {mean:.1f} tokens - the "
    "AI-polished version is shorter on average, which runs against the "
    "objection that screeners simply reward added length. The x-axis is "
    "clipped to [{lo}, {hi}] tokens for legibility; {n_clipped} pairs "
    "({clipped_pct:.1f}%) fall below the left edge (all clipped pairs are "
    "large negative deltas, i.e. sharply shortened, not omitted from any "
    "statistic above) and are not individually visible."
)


def main() -> None:
    figstyle.apply_style()

    # Source: results/qual_audit/length_check.csv
    df = pd.read_csv(REPO_ROOT / "results" / "qual_audit" / "length_check.csv")
    delta = df["delta_tokens"]

    n = len(delta)
    median = delta.median()
    mean = delta.mean()
    pos_pct = (delta > 0).mean() * 100
    n_clipped = int((delta < CLIP_LOW).sum())
    clipped_pct = n_clipped / n * 100

    caption = CAPTION_TEMPLATE.format(
        pos_pct=pos_pct, median=median, mean=mean, lo=CLIP_LOW, hi=CLIP_HIGH,
        n_clipped=n_clipped, clipped_pct=clipped_pct,
    )
    print(caption)

    clipped = delta.clip(lower=CLIP_LOW)  # pairs beyond the edge pile into the leftmost bin

    fig, ax = plt.subplots(figsize=(figstyle.SINGLE_COL_WIDTH, 3.2))

    bins = np.linspace(CLIP_LOW, CLIP_HIGH, 46)
    counts, bin_edges, patches = ax.hist(
        clipped, bins=bins, color=figstyle.HUMAN_COLOR, alpha=0.85,
        edgecolor="white", linewidth=0.4, zorder=2,
    )

    for count, left, right, patch in zip(counts, bin_edges[:-1], bin_edges[1:], patches):
        if left >= 0:
            patch.set_facecolor(figstyle.AI_COLOR)

    # The leftmost bin absorbs every pair clipped at CLIP_LOW - mark it as a
    # pileup artifact, not a real mode, so it isn't misread as one.
    patches[0].set_hatch("///")
    patches[0].set_edgecolor(figstyle.INK_SECONDARY)
    ax.annotate(
        f"≤{CLIP_LOW}\npileup\n(n={n_clipped})",
        xy=(CLIP_LOW, counts[0]), xytext=(10, -4), textcoords="offset points",
        ha="left", va="top", fontsize=6.3, color=figstyle.INK_SECONDARY,
    )

    ax.axvline(0, color=figstyle.INK_SECONDARY, linewidth=1.0, zorder=3)
    ax.axvline(median, color=figstyle.INK_PRIMARY, linewidth=1.2, linestyle=(0, (3, 2)), zorder=3)
    ax.axvline(mean, color=figstyle.INK_SECONDARY, linewidth=1.2, linestyle=(0, (1, 1.5)), zorder=3)

    ymax = counts.max()
    ax.annotate(
        f"median {median:.1f}", xy=(median, ymax * 0.97), xytext=(-6, 0),
        textcoords="offset points", ha="right", va="top", fontsize=7,
        color=figstyle.INK_PRIMARY,
    )
    ax.annotate(
        f"mean {mean:.1f}", xy=(mean, ymax * 0.83), xytext=(6, 0),
        textcoords="offset points", ha="left", va="top", fontsize=7,
        color=figstyle.INK_SECONDARY,
    )
    ax.annotate(
        f"positive tail\n{pos_pct:.1f}% of pairs",
        xy=(CLIP_HIGH * 0.55, ymax * 0.55), ha="center", va="center",
        fontsize=6.8, color=figstyle.AI_COLOR,
    )

    ax.set_xlim(CLIP_LOW, CLIP_HIGH)
    ax.set_xlabel("Length change, AI-polished minus human (tokens)")
    ax.set_ylabel("Pairs")
    figstyle.style_grid_y(ax)

    fig.tight_layout()
    out_path = REPO_ROOT / "data" / "figures" / "length_change_distribution"
    figstyle.save_fig(fig, out_path)
    print(f"Saved {out_path}.png / .pdf")


if __name__ == "__main__":
    main()
