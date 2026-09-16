"""Figure 7 — qwen2.5 preference rate vs. parameter count (3B/14B/32B).

Counts recomputed directly from the raw per-decision files at the repo
root: results_qwen2.53b.txt, results_qwen2.5_14b.txt, results_qwen2.5_32b.txt
(grep '^WINNER:' counts), matching the brief's stated 1269/2109, 1874/2109,
1821/2109.

This is three points from one model family. The line connecting them is
visual guidance only, not a fit - no trend line is drawn, and none should be
read into the plot. The series rises (3B -> 14B) then falls (14B -> 32B):
non-monotonic, suggestive only, not a scaling law.
"""

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import figstyle  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]

# Source: results_qwen2.53b.txt, results_qwen2.5_14b.txt, results_qwen2.5_32b.txt
POINTS = [
    ("qwen2.5:3b", 3, "results_qwen2.53b.txt"),
    ("qwen2.5:14b", 14, "results_qwen2.5_14b.txt"),
    ("qwen2.5:32b", 32, "results_qwen2.5_32b.txt"),
]

_WINNER_RX = re.compile(r"^WINNER:\s*(AI|HUMAN):", re.MULTILINE)

CAPTION = (
    "Figure 7. qwen2.5 AI-polished preference rate at three parameter "
    "counts (log x-axis), 95% Wilson CIs, recomputed from the raw per-pair "
    "decision files (n=2,109 pairs at each size). Three points, one model "
    "family: the connecting line is a visual guide between the three "
    "measured sizes, not a fitted trend - none is drawn or implied. The "
    "rate rises from 3B to 14B and then falls at 32B: non-monotonic, "
    "suggestive only, not evidence of a scaling law."
)


def compute_rate(fname: str) -> tuple[int, int, float]:
    text = (REPO_ROOT / fname).read_text(encoding="utf-8", errors="replace")
    winners = _WINNER_RX.findall(text)
    n = len(winners)
    k = sum(1 for w in winners if w == "AI")
    return k, n, k / n


def main() -> None:
    figstyle.apply_style()

    data = []
    for model, params_b, fname in POINTS:
        k, n, rate = compute_rate(fname)
        lo, hi = figstyle.wilson_ci(k, n)
        data.append({"model": model, "params_b": params_b, "k": k, "n": n,
                      "rate": rate, "lo": lo, "hi": hi})
        print(f"{model}: {k}/{n} = {rate:.6f}  CI=[{lo:.4f}, {hi:.4f}]")

    fig, ax = plt.subplots(figsize=(figstyle.SINGLE_COL_WIDTH, 3.2))

    xs = [d["params_b"] for d in data]
    ys = [d["rate"] for d in data]

    # Thin connecting line: a visual guide between measured points, not a fit.
    ax.plot(xs, ys, color=figstyle.AI_COLOR, linewidth=1.4, zorder=2, alpha=0.8)

    for d in data:
        ax.plot([d["params_b"], d["params_b"]], [d["lo"], d["hi"]],
                 color=figstyle.AI_COLOR, linewidth=2.0, zorder=3, solid_capstyle="round")
        ax.plot([d["params_b"]], [d["rate"]], marker=figstyle.AI_MARKER, linestyle="none",
                 zorder=4, **figstyle.ring_kwargs(figstyle.AI_COLOR))
        ax.annotate(
            f"{d['model']}\n{d['rate']:.3f}",
            xy=(d["params_b"], d["hi"]), xytext=(0, 6), textcoords="offset points",
            ha="center", va="bottom", fontsize=6.8, color=figstyle.INK_SECONDARY,
        )

    ax.axhline(0.5, color=figstyle.PARITY_COLOR, linewidth=1.2, linestyle=(0, (4, 3)), zorder=1)
    ax.text(3, 0.515, "parity", color=figstyle.PARITY_COLOR, fontsize=7, ha="left", va="bottom")

    ax.set_xscale("log")
    ax.set_xticks(xs)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}B"))
    ax.set_xlim(2.3, 42)
    ax.set_ylim(0.45, 1.0)
    ax.set_xlabel("Parameter count (log scale)")
    ax.set_ylabel("AI-polished preference rate")
    figstyle.style_grid_x(ax)

    fig.tight_layout()
    out_path = REPO_ROOT / "data" / "figures" / "qwen_scale"
    figstyle.save_fig(fig, out_path)
    print(f"Saved {out_path}.png / .pdf")


if __name__ == "__main__":
    main()
