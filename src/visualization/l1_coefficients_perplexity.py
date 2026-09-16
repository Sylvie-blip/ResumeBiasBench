"""Figure 5 — L1 logistic-regression coefficients, with and without perplexity.

Runs `src/classifiers/logistic_regression.py` (this repo, after the additive
`--include-perplexity` flag added for this figure) twice and parses its
"Learned Metric Weights" block from stdout, rather than hardcoding numbers,
so the figure stays reproducible from source. Underlying data:
data/results/new_merged.csv (2,109 Human + 2,109 AI rows, 9 style features).

Ground-truth check performed before plotting (see FIGURE_REPORT.md): the
with-perplexity run must reproduce the paper's Table 4 entropy/perplexity
pair (+5.990 / -6.270). It does (5.990217 / -6.270441), so the classifier is
unchanged since Table 4 was produced and this figure proceeds.
"""

import re
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import figstyle  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
CLASSIFIER_SCRIPT = REPO_ROOT / "src" / "classifiers" / "logistic_regression.py"
# logistic_regression.py hardcodes this absolute input path (pre-existing
# behavior, not touched by the additive flag change) - stage the real data
# there if it is not already present.
HARDCODED_CSV = Path("/Users/sylviadong/Documents/new_merged.csv")
SOURCE_CSV = REPO_ROOT / "data" / "results" / "new_merged.csv"

DISPLAY_NAME = {
    "entropy": "Token entropy",
    "perplexity": "Perplexity",
    "rolling_entropy": "Sliding-window entropy",
    "sentence_entropy_variance": "Sentence entropy variance",
    "lexical_density": "Lexical density",
    "pattern_regularity": "Pattern regularity",
    "repetition": "Repetition",
    "sentence_evenness": "Sentence evenness",
    "tonal_stability": "Tonal stability",
}

CAPTION = (
    "Figure 5. L1-penalized logistic regression coefficients for classifying "
    "a resume as AI-polished (positive, orange) vs. human-written (negative, "
    "blue), from src/classifiers/logistic_regression.py on data/results/"
    "new_merged.csv (n=4,218 texts; fixed random_state=1234; C=1/0.0001). "
    "Left: perplexity included, reproducing the paper's Table 4 collinearity "
    "artifact (token entropy +5.99 vs. perplexity -6.27 - perplexity is a "
    "deterministic transform of entropy in this codebase's Metrics.py, so "
    "the pair is one signal split into two opposite-signed coefficients, "
    "not two independent findings). Right: perplexity dropped. Rows share "
    "one order in both panels, sorted by |coefficient| in the right "
    "(without-perplexity) panel; perplexity has no such rank and is inserted "
    "next to entropy on the left only. Both panels share one x-scale so the "
    "collapse in magnitude is directly comparable. Note: recomputed "
    "without-perplexity coefficients do not match the paper's description "
    "of this panel (\"only entropy and lexical density survive\") - lexical "
    "density is the largest coefficient, but entropy is one of the smallest "
    "survivors, well below repetition, pattern regularity and sliding-window "
    "entropy. See FIGURE_REPORT.md."
)


def ensure_hardcoded_csv() -> None:
    if not HARDCODED_CSV.exists():
        HARDCODED_CSV.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(SOURCE_CSV, HARDCODED_CSV)


def run_classifier(include_perplexity: bool) -> dict:
    cmd = [sys.executable, str(CLASSIFIER_SCRIPT)]
    if include_perplexity:
        cmd.append("--include-perplexity")
    proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
    out = proc.stdout

    block_match = re.search(
        r"Learned Metric Weights.*?\n(.*?)\n\s*Intercept", out, re.DOTALL
    )
    if not block_match:
        raise RuntimeError("Could not find coefficient block in classifier output")

    coefs = {}
    for line in block_match.group(1).splitlines():
        line = line.strip()
        if not line:
            continue
        m = re.match(r"^([a-zA-Z_]+):\s*(-?\d+\.\d+)$", line)
        if m:
            coefs[m.group(1)] = float(m.group(2))
    return coefs


def main() -> None:
    figstyle.apply_style()
    ensure_hardcoded_csv()

    without = run_classifier(include_perplexity=False)
    with_pp = run_classifier(include_perplexity=True)

    # Confirm the with-perplexity run reproduces Table 4 before plotting.
    entropy_wp = with_pp.get("entropy")
    perplexity_wp = with_pp.get("perplexity")
    table4_ok = (
        entropy_wp is not None
        and perplexity_wp is not None
        and abs(entropy_wp - 5.990) < 0.05
        and abs(perplexity_wp - (-6.270)) < 0.05
    )
    print(f"With-perplexity entropy={entropy_wp:.6f}, perplexity={perplexity_wp:.6f}")
    print(f"Matches paper Table 4 (+5.990 / -6.270)? {table4_ok}")
    if not table4_ok:
        print(
            "STOPPING: regenerated with-perplexity coefficients do not match "
            "Table 4. Not plotting Figure 5. Something else changed in the "
            "classifier or data since the paper was written.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Row order: sorted by |coef| in the WITHOUT-perplexity panel.
    row_order = sorted(without.keys(), key=lambda f: -abs(without[f]))
    entropy_pos = row_order.index("entropy")
    with_row_order = row_order[: entropy_pos + 1] + ["perplexity"] + row_order[entropy_pos + 1 :]

    fig, (ax_with, ax_without) = plt.subplots(
        1, 2, figsize=(figstyle.FULL_WIDTH, 3.2), sharex=True
    )

    def plot_panel(ax, order, coefs, title):
        y_pos = list(range(len(order)))[::-1]
        for y, feat in zip(y_pos, order):
            val = coefs[feat]
            color = figstyle.AI_COLOR if val >= 0 else figstyle.HUMAN_COLOR
            marker = figstyle.AI_MARKER if val >= 0 else figstyle.HUMAN_MARKER
            ax.plot([0, val], [y, y], color=color, linewidth=1.4, zorder=2, solid_capstyle="round")
            ax.plot(
                [val], [y], marker=marker, linestyle="none", zorder=3,
                **figstyle.ring_kwargs(color),
            )
        ax.set_yticks(y_pos)
        ax.set_yticklabels([DISPLAY_NAME[f] for f in order])
        ax.axvline(0, color=figstyle.INK_SECONDARY, linewidth=0.8, zorder=1)
        figstyle.style_grid_x(ax)
        ax.set_xlabel("L1 logistic regression coefficient")
        ax.set_title(title, fontsize=8.5, color=figstyle.INK_PRIMARY)

    plot_panel(ax_with, with_row_order, with_pp, "With perplexity")
    plot_panel(ax_without, row_order, without, "Without perplexity")

    xmax = max(abs(v) for v in with_pp.values()) * 1.15
    ax_with.set_xlim(-xmax, xmax)

    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D([0], [0], marker=figstyle.AI_MARKER, color=figstyle.AI_COLOR,
               linestyle="none", **figstyle.ring_kwargs(figstyle.AI_COLOR),
               label="Positive (favors AI-polished class)"),
        Line2D([0], [0], marker=figstyle.HUMAN_MARKER, color=figstyle.HUMAN_COLOR,
               linestyle="none", **figstyle.ring_kwargs(figstyle.HUMAN_COLOR),
               label="Negative (favors human-written class)"),
    ]
    fig.legend(
        handles=legend_handles, loc="lower center", ncol=2,
        bbox_to_anchor=(0.5, -0.06), frameon=False,
    )

    fig.tight_layout()
    out_path = REPO_ROOT / "data" / "figures" / "l1_coefficients_perplexity"
    figstyle.save_fig(fig, out_path)
    print(f"Saved {out_path}.png / .pdf")


if __name__ == "__main__":
    main()
