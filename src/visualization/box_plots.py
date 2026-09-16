"""Create box plots per metric comparing three datasets:

- `winning` CSV (e.g. human-winning rows)
- `grouped` CSV which contains both Human and AI rows labeled by a group column

The script splits `grouped` by the group column into the Human and AI
datasets, finds numeric metric columns common to all three datasets, and
creates box plots for each metric comparing the three distributions.

Example:
  python box_plots.py \
    --winning /path/to/winning.csv \
    --grouped /path/to/grouped.csv \
    --group-col group --human Human --ai AI \
    --out Bias_code/Graphing/boxplots_compare.png
"""

from __future__ import annotations

import argparse
from math import ceil
from pathlib import Path
from typing import Sequence

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare boxplots across three datasets (winning, Human, AI)")
    p.add_argument("--winning", required=True, type=Path, help="CSV file with winning rows (e.g. human-winning)")
    p.add_argument("--grouped", required=True, type=Path, help="CSV containing both Human and AI labelled by group column")
    p.add_argument("--group-col", default="group", help="Column name in grouped CSV that contains the group labels (default: 'group')")
    p.add_argument("--human", default="Human", help="Label value representing human rows in grouped CSV (default: 'Human')")
    p.add_argument("--ai", default="AI", help="Label value representing AI rows in grouped CSV (default: 'AI')")
    p.add_argument("--out", type=Path, default=Path(__file__).parent / "boxplots_compare.png", help="Output PNG path")
    p.add_argument("--ncols", type=int, default=3, help="Number of columns in the subplot grid (default: 3)")
    p.add_argument("--dpi", type=int, default=200, help="Output image DPI")
    return p.parse_args(argv)


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")
    return pd.read_csv(path)


def numeric_common_columns(dfs: Sequence[pd.DataFrame]) -> list[str]:
    cols = set(dfs[0].columns)
    for df in dfs[1:]:
        cols &= set(df.columns)
    if not cols:
        return []
    numeric = []
    for c in sorted(cols):
        try:
            combined = pd.concat([df[c].dropna() for df in dfs], ignore_index=True)
            pd.to_numeric(combined)
            numeric.append(c)
        except Exception:
            continue
    return numeric


def make_boxplots(winning: pd.DataFrame, grouped: pd.DataFrame, group_col: str, human_label: str, ai_label: str, out_path: Path, ncols: int = 3, dpi: int = 200) -> None:
    if group_col not in grouped.columns:
        raise KeyError(f"group column '{group_col}' not found in grouped CSV. Available columns: {list(grouped.columns)}")

    human_df = grouped[grouped[group_col] == human_label]
    ai_df = grouped[grouped[group_col] == ai_label]

    if human_df.empty:
        raise ValueError(f"No rows found with {group_col} == {human_label}")
    if ai_df.empty:
        raise ValueError(f"No rows found with {group_col} == {ai_label}")

    metrics = numeric_common_columns([winning, human_df, ai_df])
    if not metrics:
        raise ValueError("No numeric metric columns found in common across the three datasets.")

    n = len(metrics)
    ncols = max(1, int(ncols))
    nrows = ceil(n / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows), squeeze=False)

    colors = ["#8da0cb", "#fc8d62", "#66c2a5"]
    labels = ["Winning", human_label, ai_label]

    for idx, metric in enumerate(metrics):
        r = idx // ncols
        c = idx % ncols
        ax = axes[r][c]

        data_w = pd.to_numeric(winning[metric], errors="coerce").dropna()
        data_h = pd.to_numeric(human_df[metric], errors="coerce").dropna()
        data_a = pd.to_numeric(ai_df[metric], errors="coerce").dropna()

        # boxplot with slightly narrower widths to bring boxes closer
        bp = ax.boxplot([data_w, data_h, data_a], labels=labels, patch_artist=True, widths=0.5)

        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)

        ax.set_title(metric)
        ax.grid(axis="y", linestyle="--", alpha=0.4)

    # hide unused axes
    total = nrows * ncols
    for extra in range(n, total):
        r = extra // ncols
        c = extra % ncols
        axes[r][c].axis("off")

    # legend (color key)
    legend_patches = [mpatches.Patch(color=colors[i], label=labels[i]) for i in range(len(labels))]
    fig.legend(handles=legend_patches, loc="upper right")

    fig.suptitle("Per-metric comparison: Winning vs Human vs AI", fontsize=14)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    winning = load_csv(args.winning)
    grouped = load_csv(args.grouped)

    try:
        make_boxplots(winning, grouped, args.group_col, args.human, args.ai, args.out, ncols=args.ncols, dpi=args.dpi)
    except Exception as e:
        print(f"Error: {e}")
        return 2

    print(f"Saved boxplots to: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
"""Create box plots per metric comparing three datasets:

- `winning_csv` (e.g. human-winning rows)
- `grouped_csv` which contains both Human and AI rows with a group column

The script splits `grouped_csv` by the group column into the Human and AI
datasets, finds numeric metric columns common to all three datasets, and
creates box plots for each metric comparing the three distributions.

Example:
  python box_plots.py \
    --winning /path/to/winning.csv \
    --grouped /path/to/grouped.csv \
    --group-col group --human Human --ai AI \
    --out Bias_code/Graphing/boxplots_compare.png
"""

from __future__ import annotations

import argparse
from math import ceil
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare boxplots across three datasets (winning, Human, AI)")
    p.add_argument("--winning", required=True, type=Path, help="CSV file with winning rows (e.g. human-winning)")
    p.add_argument("--grouped", required=True, type=Path, help="CSV containing both Human and AI labelled by group column")
    p.add_argument("--group-col", default="group", help="Column name in grouped CSV that contains the group labels (default: 'group')")
    p.add_argument("--human", default="Human", help="Label value representing human rows in grouped CSV (default: 'Human')")
    p.add_argument("--ai", default="AI", help="Label value representing AI rows in grouped CSV (default: 'AI')")
    p.add_argument("--out", type=Path, default=Path(__file__).parent / "boxplots_compare.png", help="Output PNG path")
    p.add_argument("--ncols", type=int, default=3, help="Number of columns in the subplot grid (default: 3)")
    p.add_argument("--dpi", type=int, default=200, help="Output image DPI")
    return p.parse_args(argv)


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")
    return pd.read_csv(path)


def numeric_common_columns(dfs: Sequence[pd.DataFrame]) -> list[str]:
    # intersect columns
    cols = set(dfs[0].columns)
    for df in dfs[1:]:
        cols &= set(df.columns)
    if not cols:
        return []
    # pick numeric ones
    numeric = []
    for c in sorted(cols):
        try:
            # Use pandas to_numeric to test convertibility
            combined = pd.concat([df[c].dropna() for df in dfs], ignore_index=True)
            pd.to_numeric(combined)
            numeric.append(c)
        except Exception:
            continue
    return numeric


def make_boxplots(winning: pd.DataFrame, grouped: pd.DataFrame, group_col: str, human_label: str, ai_label: str, out_path: Path, ncols: int = 3, dpi: int = 200) -> None:
    if group_col not in grouped.columns:
        raise KeyError(f"group column '{group_col}' not found in grouped CSV. Available columns: {list(grouped.columns)}")

    human_df = grouped[grouped[group_col] == human_label]
    ai_df = grouped[grouped[group_col] == ai_label]

    if human_df.empty:
        raise ValueError(f"No rows found with {group_col} == {human_label}")
    if ai_df.empty:
        raise ValueError(f"No rows found with {group_col} == {ai_label}")

    metrics = numeric_common_columns([winning, human_df, ai_df])
    if not metrics:
        raise ValueError("No numeric metric columns found in common across the three datasets.")

    n = len(metrics)
    ncols = max(1, int(ncols))
    nrows = ceil(n / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows))
    if nrows * ncols == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for i, metric in enumerate(metrics):
        ax = axes[i]
        data_w = pd.to_numeric(winning[metric].dropna(), errors="coerce").dropna()
        data_h = pd.to_numeric(human_df[metric].dropna(), errors="coerce").dropna()
        data_a = pd.to_numeric(ai_df[metric].dropna(), errors="coerce").dropna()

        parts = ax.boxplot([data_w, data_h, data_a], labels=["Winning", human_label, ai_label], patch_artist=True)

        # color boxes
        colors = ["#8da0cb", "#fc8d62", "#66c2a5"]
        for patch, color in zip(parts["boxes"], colors):
            patch.set_facecolor(color)

        ax.set_title(metric)
        ax.grid(axis="y", linestyle="--", alpha=0.4)

    # hide unused axes
    for j in range(n, len(axes)):
        fig.delaxes(axes[j])

    fig.suptitle("Per-metric comparison: Winning vs Human vs AI", fontsize=14)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    winning = load_csv(args.winning)
    grouped = load_csv(args.grouped)

    try:
        make_boxplots(winning, grouped, args.group_col, args.human, args.ai, args.out, ncols=args.ncols, dpi=args.dpi)
    except Exception as e:
        print(f"Error: {e}")
        return 2

    print(f"Saved boxplots to: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
        ax.set_xticks([1, 2, 3])
        ax.set_xticklabels(labels, rotation=20)
        ax.set_title(col)
        ax.grid(axis="y", linestyle="--", alpha=0.3)

    # hide any unused subplots
    total_plots = nrows * ncols
    for idx in range(n, total_plots):
        r = idx // ncols
        c = idx % ncols
        axes[r][c].axis("off")

    # legend (color key)
    legend_patches = [mpatches.Patch(color=colors[i], label=labels[i]) for i in range(len(labels))]
    fig.legend(handles=legend_patches, loc="upper right")
    fig.suptitle("Per-metric boxplots — dataset comparison", fontsize=14)
    fig.tight_layout(rect=[0, 0, 0.95, 0.95])
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi)
    plt.close(fig)


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Create boxplots comparing three CSV datasets per metric column")
    parser.add_argument("--csv1", required=True, type=Path)
    parser.add_argument("--csv2", required=True, type=Path)
    parser.add_argument("--csv3", required=True, type=Path)
    parser.add_argument("--labels", default="A,B,C", help="Comma-separated labels for the three datasets")
    parser.add_argument("--cols", default=None, help="Optional comma-separated list of columns to plot (must exist in all CSVs)")
    parser.add_argument("--out", default=Path(__file__).parent / "box_plots.png", type=Path, help="Output PNG path")
    parser.add_argument("--dpi", default=200, type=int)

    args = parser.parse_args(argv)

    dfs = [read_csv(args.csv1), read_csv(args.csv2), read_csv(args.csv3)]
    labels = [s.strip() for s in args.labels.split(",")]
    if len(labels) != 3:
        print("--labels must provide exactly 3 comma-separated labels")
        return 2

    if args.cols:
        cols = [c.strip() for c in args.cols.split(",") if c.strip()]
    else:
        cols = numeric_columns_intersection(dfs)

    if not cols:
        print("No numeric columns found in the intersection of the three CSVs.")
        return 0

    print(f"Plotting {len(cols)} columns to {args.out}")
    draw_boxplots(dfs, labels, cols, Path(args.out), dpi=args.dpi)
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
