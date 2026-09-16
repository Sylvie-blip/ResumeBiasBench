"""Figure 11 — dumbbell plot: full-corpus rate vs. preserved-subset rate (t=0.90).

Source: results/qual_audit/preserved_subset_winrates.csv, the t=0.90
per-model table (subset_k/n/rate with its cluster-robust 95% CI already
computed there - reused rather than recomputed by a different method,
per the CI ground rule). The full-corpus ("local_baseline") CI is not in
that file; since each model contributes exactly one decision per pair (no
repeated measurement to cluster), a Wilson interval on local_baseline_k/n
is the appropriate simple-proportion CI and is computed here.

Sorted so the outlier is not buried in the middle: qwen2.5:3b already has
the lowest full-corpus rate of the 7 models, so sorting by full-corpus rate
naturally puts it at one end rather than requiring a special-cased sort.
"""

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import figstyle  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
# Source: results/qual_audit/preserved_subset_winrates.csv
SRC_CSV = REPO_ROOT / "results" / "qual_audit" / "preserved_subset_winrates.csv"

CAPTION = (
    "Figure 11. Full-corpus AI-preference rate (blue) vs. the preserved-"
    "subset rate at t=0.90 (orange) - pairs where dictionary-scoped "
    "qualification keywords are provably retained - per model, with 95% "
    "CIs. Source: results/qual_audit/preserved_subset_winrates.csv "
    "(subset CI: cluster-robust, clustered on pair_id, already computed "
    "there; full-corpus CI: Wilson, computed here, since each model "
    "contributes one decision per pair with nothing to cluster). Seven "
    "models' pairs sit almost on top of each other - the preference "
    "survives holding qualifications constant. qwen2.5:3b is the "
    "exception: its preserved-subset rate drops to 0.536, with a CI that "
    "includes parity (0.5), unlike its own full-corpus rate (0.602) or any "
    "other model's subset rate."
)


def parse_t090_table(path: Path) -> list[dict]:
    lines = path.read_text(encoding="utf-8").splitlines()
    start = next(i for i, l in enumerate(lines) if "Threshold t=0.90" in l)
    header_idx = next(
        i for i in range(start, len(lines))
        if lines[i].startswith("model,subset_k")
    )
    rows = []
    reader = csv.DictReader(lines[header_idx:])
    for row in reader:
        # DictReader silently skips the blank separator line, so the next
        # section's "=== Threshold ... ===" / "pooled_k,..." lines would
        # otherwise be mis-parsed as bogus data rows under the same header.
        # Every real model name contains ':' (e.g. "cogito:3b"); nothing
        # else in this file does.
        model = row.get("model") or ""
        if ":" not in model:
            break
        rows.append(row)
    return rows


def main() -> None:
    figstyle.apply_style()

    raw_rows = parse_t090_table(SRC_CSV)

    data = []
    for r in raw_rows:
        full_k, full_n = int(r["local_baseline_k"]), int(r["local_baseline_n"])
        full_rate = full_k / full_n
        full_lo, full_hi = figstyle.wilson_ci(full_k, full_n)
        sub_rate = float(r["subset_rate"])
        sub_lo, sub_hi = float(r["cluster_ci_95_lo"]), float(r["cluster_ci_95_hi"])
        data.append({
            "model": r["model"],
            "full_rate": full_rate, "full_lo": full_lo, "full_hi": full_hi,
            "full_n": full_n,
            "sub_rate": sub_rate, "sub_lo": sub_lo, "sub_hi": sub_hi,
            "sub_n": int(r["subset_n"]),
        })

    data.sort(key=lambda d: d["full_rate"])

    fig, ax = plt.subplots(figsize=(figstyle.SINGLE_COL_WIDTH, 3.8))
    y_pos = list(range(len(data)))

    for y, d in zip(y_pos, data):
        ax.plot([d["full_rate"], d["sub_rate"]], [y, y],
                 color=figstyle.INK_SECONDARY, linewidth=1.2, alpha=0.5, zorder=2)
        ax.plot([d["full_lo"], d["full_hi"]], [y, y], color=figstyle.HUMAN_COLOR,
                 linewidth=1.6, alpha=0.55, zorder=2, solid_capstyle="round")
        ax.plot([d["sub_lo"], d["sub_hi"]], [y, y], color=figstyle.AI_COLOR,
                 linewidth=1.6, alpha=0.55, zorder=2, solid_capstyle="round")
        ax.plot([d["full_rate"]], [y], marker=figstyle.HUMAN_MARKER, linestyle="none",
                 zorder=3, **figstyle.ring_kwargs(figstyle.HUMAN_COLOR))
        ax.plot([d["sub_rate"]], [y], marker=figstyle.AI_MARKER, linestyle="none",
                 zorder=3, **figstyle.ring_kwargs(figstyle.AI_COLOR))

    ax.axvline(0.5, color=figstyle.PARITY_COLOR, linewidth=1.2, linestyle=(0, (4, 3)), zorder=1)
    ax.text(0.5, -0.9, "parity", color=figstyle.PARITY_COLOR, fontsize=7.5,
             ha="center", va="top", clip_on=False)

    ax.set_yticks(y_pos)
    ax.set_yticklabels([d["model"] for d in data])
    ax.set_xlim(0.4, 1.0)
    ax.set_ylim(-0.6, len(data) + 0.6)
    ax.set_xlabel("AI-polished preference rate")
    figstyle.style_grid_x(ax)

    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], marker=figstyle.HUMAN_MARKER, color=figstyle.HUMAN_COLOR,
               linestyle="none", **figstyle.ring_kwargs(figstyle.HUMAN_COLOR),
               label="Full corpus (n=2,109)"),
        Line2D([0], [0], marker=figstyle.AI_MARKER, color=figstyle.AI_COLOR,
               linestyle="none", **figstyle.ring_kwargs(figstyle.AI_COLOR),
               label="Preserved subset, t=0.90 (n=360)"),
    ]
    ax.legend(handles=handles, loc="upper left", fontsize=6.8, frameon=False)

    fig.tight_layout()
    out_path = REPO_ROOT / "data" / "figures" / "preserved_vs_full"
    figstyle.save_fig(fig, out_path)
    print(f"Saved {out_path}.png / .pdf")
    for d in data:
        print(d)


if __name__ == "__main__":
    main()
