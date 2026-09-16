"""Figure 6 — per-model AI preference rate with 95% CI, all 8 models.

Rates are recomputed from the raw per-decision result files at the repo
root (results_*.txt for 7 models; grep '^WINNER:' counts), not read out of
the paper's Table 1. granite4:3b has no per-decision file anywhere in this
repo (see results/qual_audit/audit_summary.md, 'Model coverage') - its point
is the paper's own Table 1 aggregate rate (0.785), shown with an open marker,
no error bar, and no n, per the brief's explicit exception for this model.

Tie handling (ground rules): pick_winner() in run_screening.py breaks a tie
on (total, experience, technical) in AI's favor. The raw result files only
record the WINNER's score triple, never the loser's, for every one of the
2,109 x 7 = 14,763 decisions - so whether any given decision was an exact
tie cannot be reconstructed from stored data (confirmed by inspection; see
FIGURE_REPORT.md). The tie rate is therefore unknown, not just unreported;
plotted rates should be read with that caveat rather than as a certain
upper bound of a known size.
"""

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import figstyle  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]

# Source: results_*.txt at the repo root (raw per-pair decisions, one row
# per PAIR/WINNER block).
RESULT_FILES = {
    "qwen2.5:3b": "results_qwen2.53b.txt",
    "qwen2.5:14b": "results_qwen2.5_14b.txt",
    "qwen2.5:32b": "results_qwen2.5_32b.txt",
    "gemma3:4b": "results_gemma3_4b.txt",
    "gemma3:12b": "results_gemma3_12b.txt",
    "mistral-small3.2:latest": "results_mistral-small3.2_latest.txt",
    "cogito:3b": "results_cogito_3b.txt",
}

GRANITE_MODEL = "granite4:3b"
# Source: paper's Table 1 (no raw decisions exist for this model anywhere in
# the repo - see results/qual_audit/audit_summary.md 'Model coverage'). This
# is the one explicit, brief-sanctioned exception to "never read a rate out
# of the paper's tables."
GRANITE_TABLE1_RATE = 0.785

_WINNER_RX = re.compile(r"^WINNER:\s*(AI|HUMAN):", re.MULTILINE)


def compute_model_rate(fname: str) -> tuple[int, int, float]:
    text = (REPO_ROOT / fname).read_text(encoding="utf-8", errors="replace")
    winners = _WINNER_RX.findall(text)
    n = len(winners)
    k = sum(1 for w in winners if w == "AI")
    return k, n, k / n


CAPTION = (
    "Figure 6. Per-model AI-preference rate across all 2,109 resume pairs, "
    "recomputed from the raw per-decision result files (95% Wilson CI). "
    "granite4:3b (open marker) has no surviving per-decision file in this "
    "repo; its point is the paper's Table 1 aggregate rate only, with no "
    "n and no interval available. No interval among the 7 directly "
    "computed models touches the 0.5 parity line. Ties in the underlying "
    "scoring rule are broken in AI's favor (pick_winner, run_screening.py), "
    "but the stored result files retain only the winning score triple, so "
    "the true tie rate cannot be recovered from this repo's data (see "
    "FIGURE_REPORT.md) - rates should be read with that caveat."
)


def main() -> None:
    figstyle.apply_style()

    rows = []
    for model, fname in RESULT_FILES.items():
        k, n, rate = compute_model_rate(fname)
        lo, hi = figstyle.wilson_ci(k, n)
        rows.append({"model": model, "k": k, "n": n, "rate": rate, "lo": lo, "hi": hi})

    rows.append(
        {"model": GRANITE_MODEL, "k": None, "n": None, "rate": GRANITE_TABLE1_RATE,
         "lo": None, "hi": None}
    )

    rows.sort(key=lambda r: r["rate"])

    fig, ax = plt.subplots(figsize=(figstyle.SINGLE_COL_WIDTH, 3.4))
    y_pos = list(range(len(rows)))

    for y, r in zip(y_pos, rows):
        if r["lo"] is not None:
            ax.plot([r["lo"], r["hi"]], [y, y], color=figstyle.AI_COLOR,
                     linewidth=2.0, zorder=2, solid_capstyle="round")
            ax.plot([r["rate"]], [y], marker=figstyle.AI_MARKER, linestyle="none",
                     zorder=3, **figstyle.ring_kwargs(figstyle.AI_COLOR))
            label = f"{r['rate']:.3f}"
        else:
            ax.plot([r["rate"]], [y], marker=figstyle.AI_MARKER, linestyle="none",
                     markerfacecolor="white", markeredgecolor=figstyle.AI_COLOR,
                     markeredgewidth=1.4, markersize=figstyle.MARKER_SIZE, zorder=3)
            label = f"{r['rate']:.3f}*"
        ax.annotate(
            label,
            xy=(max(r["rate"], r["hi"] if r["hi"] else r["rate"]) + 0.012, y),
            va="center", ha="left", fontsize=7, color=figstyle.INK_SECONDARY,
        )

    ax.axvline(0.5, color=figstyle.PARITY_COLOR, linewidth=1.2, linestyle=(0, (4, 3)), zorder=1)
    ax.text(0.5, len(rows) - 0.55, "parity", color=figstyle.PARITY_COLOR,
             fontsize=7, ha="center", va="bottom")

    ax.set_yticks(y_pos)
    ax.set_yticklabels([r["model"] for r in rows])
    ax.set_xlim(0.4, 1.0)
    ax.set_ylim(-0.8, len(rows) - 0.05)
    ax.set_xlabel("AI-polished preference rate")
    ax.set_title(
        "n = 2,109 pairs per model,\n*granite4:3b: Table 1 value, n & CI unknown",
        fontsize=6.8, color=figstyle.INK_SECONDARY, loc="left", pad=8,
    )
    figstyle.style_grid_x(ax)

    fig.tight_layout()
    out_path = REPO_ROOT / "data" / "figures" / "preference_by_model"
    figstyle.save_fig(fig, out_path)
    print(f"Saved {out_path}.png / .pdf")
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
