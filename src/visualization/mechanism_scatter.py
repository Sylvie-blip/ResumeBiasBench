"""Figure 13 — mechanism scatter: separation effect size vs. win-prediction coefficient.

X: Mann-Whitney rank-biserial effect size (AI vs. Human) per raw feature,
from results/mannwhitney_effect_sizes.csv (written by src/scripts/
compare_metrics.py on data/results/new_merged.csv - the additive CSV output
added for this figure; console output is unchanged).

Y: standardized delta-regression coefficient predicting which version wins
the screening decision, from results/qual_audit/feature_regression_coefficients.csv
(already computed; not recomputed here by a different method).

perplexity is excluded from this plot: it has an effect size but no
regression coefficient (dropped there for collinearity with entropy, see
FIGURE_REPORT.md), so the two sources only share 8 features.

No trend line, no correlation coefficient: the figure exists to show the
absence of a diagonal relationship between "separates AI from human text"
and "predicts which one a screener picks."
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import figstyle  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]

DISPLAY_NAME = {
    "entropy": "Token entropy",
    "rolling_entropy": "Sliding-window entropy",
    "sentence_entropy_variance": "Sentence entropy variance",
    "lexical_density": "Lexical density",
    "pattern_regularity": "Pattern regularity",
    "repetition": "Repetition",
    "sentence_evenness": "Sentence evenness",
    "tonal_stability": "Tonal stability",
}

CAPTION = (
    "Figure 13. Mann-Whitney effect size separating AI-edited from human "
    "resumes (x; results/mannwhitney_effect_sizes.csv, from "
    "src/scripts/compare_metrics.py on data/results/new_merged.csv) against "
    "each feature's standardized coefficient predicting which version a "
    "screener picks (y; results/qual_audit/feature_regression_coefficients.csv). "
    "perplexity is excluded (present on the x-axis source but dropped from "
    "the y-axis regression for collinearity with entropy - the two sources "
    "share 8 features, not 9). No trend line, no correlation coefficient: "
    "the point of the figure is the absence of a diagonal. Lexical density "
    "and repetition sit far out on the separation axis but near zero on "
    "the prediction axis; token entropy does the reverse - what "
    "distinguishes AI-edited from human text is not what a screener "
    "rewards."
)


def main() -> None:
    figstyle.apply_style()

    # Source: results/mannwhitney_effect_sizes.csv (written by the modified compare_metrics.py)
    eff = pd.read_csv(REPO_ROOT / "results" / "mannwhitney_effect_sizes.csv")
    eff = eff.set_index("metric")["r"]

    # Source: results/qual_audit/feature_regression_coefficients.csv
    coef = pd.read_csv(
        REPO_ROOT / "results" / "qual_audit" / "feature_regression_coefficients.csv",
        index_col=0,
    )["coef"]
    coef.index = [i.replace("delta_", "") for i in coef.index]
    coef = coef[coef.index.isin(DISPLAY_NAME)]

    features = sorted(DISPLAY_NAME, key=lambda f: DISPLAY_NAME[f])
    xs = eff.loc[features]
    ys = coef.loc[features]

    fig, ax = plt.subplots(figsize=(figstyle.SINGLE_COL_WIDTH, 3.6))

    ax.axhline(0, color=figstyle.INK_SECONDARY, linewidth=0.8, zorder=1)
    ax.axvline(0, color=figstyle.INK_SECONDARY, linewidth=0.8, zorder=1)

    ax.plot(xs, ys, marker=figstyle.AI_MARKER, linestyle="none", zorder=3,
             **figstyle.ring_kwargs(figstyle.AI_COLOR))

    # Four features (rolling_entropy, sentence_entropy_variance,
    # sentence_evenness, pattern_regularity) sit almost on top of each
    # other near (0.3-0.45, -0.015 to 0.011) - stack their labels in a
    # column to the right with thin leader lines back to each point,
    # rather than letting the text collide.
    isolated_offsets = {
        "entropy": (8, 8),
        "lexical_density": (10, 10),
        "repetition": (10, 4),
        "tonal_stability": (0, -16),
    }
    for f, (dx, dy) in isolated_offsets.items():
        ha = "left" if dx >= 0 else "right"
        va = "center" if dy != -16 else "top"
        ax.annotate(
            DISPLAY_NAME[f], xy=(xs[f], ys[f]), xytext=(dx, dy),
            textcoords="offset points", fontsize=6.8, ha=ha, va=va,
            color=figstyle.INK_PRIMARY,
        )

    cluster = ["rolling_entropy", "sentence_entropy_variance", "sentence_evenness", "pattern_regularity"]
    label_x_data = 0.62
    label_ys_data = [0.115, 0.075, 0.035, -0.005]
    for f, ly in zip(cluster, label_ys_data):
        ax.annotate(
            DISPLAY_NAME[f], xy=(xs[f], ys[f]), xytext=(label_x_data, ly),
            textcoords="data", fontsize=6.8, ha="left", va="center",
            color=figstyle.INK_PRIMARY,
            arrowprops=dict(arrowstyle="-", color=figstyle.INK_SECONDARY, linewidth=0.6),
        )

    ax.set_xlabel("Mann-Whitney effect size (AI vs. Human separation)")
    ax.set_ylabel("Standardized coefficient (predicts screener win)")
    figstyle.style_grid_x(ax)
    figstyle.style_grid_y(ax)

    ax.set_xlim(-0.85, 1.05)
    ax.set_ylim(-0.24, 0.21)

    fig.tight_layout()
    out_path = REPO_ROOT / "data" / "figures" / "mechanism_scatter"
    figstyle.save_fig(fig, out_path)
    print(f"Saved {out_path}.png / .pdf")
    print(pd.DataFrame({"effect_size_r": xs, "coef": ys}))


if __name__ == "__main__":
    main()
