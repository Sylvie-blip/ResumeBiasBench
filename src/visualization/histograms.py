"""Compare metric distributions across three CSV datasets.

Reads three CSV files that share the same metric column names. For each
numeric metric column present in all three files, writes a PNG histogram
overlaying the three distributions.

Usage (example):
  python3 /Users/sylviadong/Documents/Resume_Bias/Bias_code/Graphing/histograms.py \
    --csvs /Users/sylviadong/Documents/new_merged_AI2.csv /Users/sylviadong/Documents/new_merged_winning.csv /Users/sylviadong/Documents/new_merged_losing.csv \
    --labels "AI-Modified" "AI Favored" "AI Disfavored" \
    --out-dir /Users/sylviadong/Documents/Resume_Bias/Bias_code/Graphing/Histograms --bins 40
    --rename "entropy" "Token Entropy" \
           "perplexity" "Perplexity" \
           "rolling_entropy" "Sliding-Window Entropy" \
           "sentence_entropy_variance" "Sentence Entropy Variance" \
           "lexical_density" "Lexical Density" \
           "pattern_regularity" "Pattern Regularity" \
           "repetition" "Type-Token Ratio" \
           "sentence_evenness" "Sentence Length Dispersion" \
           "tonal_stability" "Sentiment Variance" 

from __future__ import annotations

import argparse
import os
import sys
from typing import List

try:
    import pandas as pd
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception as e:
    print("Missing required plotting libraries. Please install 'pandas numpy matplotlib'.\n", e)
    sys.exit(1)


def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def common_columns(dfs: List[pd.DataFrame]) -> List[str]:
    cols = set(dfs[0].columns)
    for df in dfs[1:]:
        cols &= set(df.columns)
    return sorted(list(cols))


def is_numeric_series(s: pd.Series) -> bool:
    # try coercion to numeric and see if any non-null values remain
    coerced = pd.to_numeric(s, errors="coerce")
    return coerced.notnull().sum() > 0


def plot_metric_histograms(csv_paths: List[str], labels: List[str], out_dir: str, bins: int = 30,
                           normalize: bool = False, log_scale: bool = False, rename_map: dict | None = None) -> None:
    os.makedirs(out_dir, exist_ok=True)

    if rename_map is None:
        rename_map = {}

    dfs = [load_csv(p) for p in csv_paths]
    cols = common_columns(dfs)
    if not cols:
        print("No common columns found across CSVs.")
        return

    # choose numeric columns only
    numeric_cols = [c for c in cols if any(is_numeric_series(df[c]) for df in dfs)]
    if not numeric_cols:
        print("No numeric columns found to plot.")
        return

    for col in numeric_cols:
        # gather numeric arrays for this column
        arrays = []
        for df in dfs:
            arr = pd.to_numeric(df[col], errors="coerce").dropna().values
            arrays.append(arr)

        # skip if all arrays are empty
        if all(len(a) == 0 for a in arrays):
            print(f"Skipping '{col}' — no numeric data in any CSV.")
            continue

        # compute shared bin range from combined data
        stacked = np.concatenate([a for a in arrays if len(a) > 0])
        vmin, vmax = np.nanmin(stacked), np.nanmax(stacked)
        if np.isfinite(vmin) and np.isfinite(vmax) and vmin < vmax:
            bins_edges = np.linspace(vmin, vmax, bins + 1)
        else:
            bins_edges = bins

        # use renamed label if provided, else original
        display_col = rename_map.get(col, col)

        plt.figure(figsize=(8, 5))
        for arr, label in zip(arrays, labels):
            if len(arr) == 0:
                continue
            plt.hist(arr, bins=bins_edges, alpha=0.5, label=label, density=normalize)

        plt.title(f"Histogram — {display_col}")
        plt.xlabel(display_col)
        plt.ylabel("Density" if normalize else "Count")
        if log_scale:
            plt.yscale("log")
        plt.legend()
        safe_name = "".join(c if c.isalnum() or c in ("_", "-") else "_" for c in display_col)
        out_path = os.path.join(out_dir, f"metric_hist_{safe_name}.png")
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"Wrote {out_path}")


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot per-metric histograms comparing CSV datasets")
    p.add_argument("--csvs", nargs=3, required=True, help="Three CSV files to compare", metavar=("A", "B", "C"))
    p.add_argument("--labels", nargs=3, default=None, help="Labels for the three datasets (provide 3)")
    p.add_argument("--out-dir", default="metric_hist_plots", help="Output directory for PNGs")
    p.add_argument("--bins", type=int, default=30, help="Number of histogram bins")
    p.add_argument("--normalize", action="store_true", help="Normalize histograms to density")
    p.add_argument("--log-scale", action="store_true", help="Log scale the y axis")
    p.add_argument("--rename", nargs="+", default=[], metavar=("OLD", "NEW"),
                   help="Rename metrics: --rename old_name new_name [old_name2 new_name2 ...]")
    return p.parse_args(argv)


def main(argv: List[str] | None = None) -> None:
    args = parse_args(argv)
    labels = args.labels if args.labels is not None else [os.path.splitext(os.path.basename(p))[0] for p in args.csvs]
    if len(labels) != 3:
        print("Please supply exactly 3 labels when using --labels")
        sys.exit(1)

    # Parse rename pairs: --rename old1 new1 old2 new2 ...
    rename_map = {}
    if len(args.rename) % 2 != 0:
        print("Error: --rename requires pairs of (old_name new_name)")
        sys.exit(1)
    for i in range(0, len(args.rename), 2):
        rename_map[args.rename[i]] = args.rename[i + 1]

    plot_metric_histograms(args.csvs, labels, args.out_dir, bins=args.bins, normalize=args.normalize,
                           log_scale=args.log_scale, rename_map=rename_map)


if __name__ == "__main__":
    main()
"""





import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def compare_groups_high_res(file_path, rename_dict, group_col='group', output_dir='/Users/sylviadong/Documents/human_vs_ai_hist'):
    # Create the directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load the dataset
    df = pd.read_csv(file_path)
    
    # Filter groups
    human_data = df[df[group_col].str.lower() == 'human']
    ai_data = df[df[group_col].str.lower() == 'ai']

    # 1. Increase general text size globally using 'context'
    # Options: "paper", "notebook", "talk", "poster" (talk is usually best for presentations)
    sns.set_theme(style="whitegrid")
    sns.set_context("talk", font_scale=1.2) 

    numeric_cols = df.select_dtypes(include=['number']).columns

    for col in numeric_cols:
        display_name = rename_dict.get(col, col)
        
        # Increased figure size slightly to accommodate larger text
        plt.figure(figsize=(12, 7))
        
        # Plotting
        sns.histplot(human_data[col], color="skyblue", label="Human", kde=True, stat="count", alpha=0.5)
        sns.histplot(ai_data[col], color="orange", label="AI", kde=True, stat="count", alpha=0.5)
        
        # 2. Customizing specific text elements for extra clarity
        plt.title(f'{display_name}', fontsize=26, pad=20, fontweight='bold')
        plt.xlabel(display_name, fontsize=22, labelpad=15)
        plt.ylabel('Count', fontsize=22, labelpad=15)
        plt.legend(fontsize=20, frameon=True)
        
        # Clean filename
        filename = f"compare_{display_name.replace(' ', '_')}.png"
        filepath = os.path.join(output_dir, filename)
        
        # 3. Increase quality to 600 DPI (Print Quality)
        plt.tight_layout()
        plt.savefig(filepath, dpi=600, bbox_inches='tight')
        plt.close()
        
        print(f"Exported high-quality plot: {filename}")

if __name__ == "__main__":
    my_renames = {
        "entropy": "Token Entropy",
        'avg_sent_len': 'Sentence Complexity',
        'flesch_kincaid': 'Readability Score',
        "perplexity": "Perplexity",
        "rolling_entropy": "Sliding-Window Entropy",
        "sentence_entropy_variance": "Sentence Entropy Variance",
        "lexical_density": "Lexical Density",
        "pattern_regularity": "Pattern Regularity",
        "repetition": "Type-Token Ratio",
        "sentence_evenness": "Sentence Length Dispersion",
        "tonal_stability": "Sentiment Variance"
    }
    
    compare_groups_high_res('/Users/sylviadong/Documents/new_merged.csv', my_renames)