"""Small utility: plot number of files per subset (subdirectory) in a directory.

Subdirectories are sorted alphabetically, then split into two halves,
each saved as a separate bar chart.

Usage (CLI):
  python /Users/sylviadong/Documents/Resume_Bias/Bias_code/Graphing/job_category_bar.py --dir '/Users/sylviadong/Downloads/sorted_data' --out resume_categories_counts_new.png

Two output files will be created:
  - resume_categories_counts_part1.png
  - resume_categories_counts_part2.png
"""

from __future__ import annotations

import argparse
import os
from typing import Dict, Iterable, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _iter_files_in_dir(path: str, recursive: bool = False, include_hidden: bool = False) -> Iterable[str]:
    """Yield full file paths for files under `path`."""
    if recursive:
        for root, dirs, files in os.walk(path, topdown=True):
            if not include_hidden:
                dirs[:] = [d for d in dirs if not d.startswith('.')]
            for fn in files:
                if not include_hidden and fn.startswith('.'):
                    continue
                yield os.path.join(root, fn)
    else:
        try:
            for entry in os.scandir(path):
                if entry.is_file():
                    if not include_hidden and entry.name.startswith('.'):
                        continue
                    yield entry.path
        except FileNotFoundError:
            return


def _save_plot(
    subsets: Dict[str, int],
    out_path: str,
    title: str,
) -> None:
    """Save a bar chart for a given subset dict."""
    labels = list(subsets.keys())
    values = [subsets[k] for k in labels]

    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 0.5), 4))
    bars = ax.bar(labels, values, color="C0")
    ax.set_ylabel("Resume count")
    ax.set_title(title)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")

    max_val = max(values) if values else 0
    ax.set_ylim(0, max_val + 30)

    for bar, val in zip(bars, values):
        ax.annotate(
            str(val),
            xy=(bar.get_x() + bar.get_width() / 2, val),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center", va="bottom", fontsize=8,
        )

    plt.tight_layout()

    out_dir = os.path.dirname(os.path.abspath(out_path))
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_files_per_subset(
    parent_dir: str,
    out_path: str,
    extensions: Optional[Iterable[str]] = None,
    include_hidden: bool = False,
    recursive: bool = False,
    title: Optional[str] = None,
) -> Dict[str, int]:
    """Count files in each immediate subdirectory, split into two halves, save two charts.

    Output files are derived from out_path:
      e.g. out.png -> out_part1.png and out_part2.png
    """
    parent_dir = os.path.abspath(parent_dir)
    if not os.path.isdir(parent_dir):
        raise ValueError(f"parent_dir is not a directory: {parent_dir}")

    exts: Optional[List[str]] = None
    if extensions:
        exts = [e.lower() if e.startswith('.') else f'.{e.lower()}' for e in extensions]

    # Collect all subdirectories sorted alphabetically
    entries = sorted(
        [e for e in os.scandir(parent_dir) if e.is_dir()],
        key=lambda e: e.name,
    )

    if not entries:
        # No subdirectories: treat the parent itself as one group
        entries_to_use = [(os.path.basename(parent_dir) or parent_dir, parent_dir)]
    else:
        entries_to_use = [
            (e.name, e.path) for e in entries
            if include_hidden or not e.name.startswith('.')
        ]

    # Count files per subdirectory
    all_subsets: Dict[str, int] = {}
    for name, path in entries_to_use:
        cnt = 0
        for fp in _iter_files_in_dir(path, recursive=recursive, include_hidden=include_hidden):
            if exts and os.path.splitext(fp)[1].lower() not in exts:
                continue
            cnt += 1
        all_subsets[name] = cnt

    # Split into two halves
    keys = list(all_subsets.keys())
    mid = len(keys) // 2
    half1 = {k: all_subsets[k] for k in keys[:mid]}
    half2 = {k: all_subsets[k] for k in keys[mid:]}

    # Derive output paths: insert _part1 / _part2 before the extension
    base, ext = os.path.splitext(out_path)
    out_path1 = f"{base}_part1{ext}"
    out_path2 = f"{base}_part2{ext}"

    base_title = title or "Resumes per Subset"
    _save_plot(half1, out_path1, title=f"{base_title} (Part 1 of 2)")
    _save_plot(half2, out_path2, title=f"{base_title} (Part 2 of 2)")

    return all_subsets


def _parse_extensions(s: Optional[str]) -> Optional[List[str]]:
    if not s:
        return None
    parts = [p.strip() for p in s.split(',') if p.strip()]
    return parts or None


def _main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Plot file counts per immediate subdirectory (split into two halves)")
    parser.add_argument('--dir', required=True, help='Parent directory containing subsets (subdirectories)')
    parser.add_argument('--out', required=True, help='Output image base path (png/pdf) — _part1 and _part2 will be appended')
    parser.add_argument('--ext', help='Comma-separated list of extensions to include (e.g. "txt,pdf")')
    parser.add_argument('--recursive', action='store_true', help='Count files recursively inside each subset')
    parser.add_argument('--include-hidden', action='store_true', help='Include hidden files and directories')
    args = parser.parse_args(argv)

    exts = _parse_extensions(args.ext)
    try:
        counts = plot_files_per_subset(
            args.dir, args.out,
            extensions=exts,
            include_hidden=args.include_hidden,
            recursive=args.recursive,
        )
    except Exception as e:
        print(f"Error: {e}")
        return 2

    base, ext = os.path.splitext(args.out)
    total = sum(counts.values())
    keys = list(counts.keys())
    mid = len(keys) // 2

    print(f"Wrote {base}_part1{ext} and {base}_part2{ext}")
    print(f"{len(counts)} subsets total, {total} files total")
    print(f"\nPart 1 ({mid} subsets):")
    for k in keys[:mid]:
        print(f"  {k}: {counts[k]}")
    print(f"\nPart 2 ({len(keys) - mid} subsets):")
    for k in keys[mid:]:
        print(f"  {k}: {counts[k]}")

    return 0


if __name__ == '__main__':
    raise SystemExit(_main())