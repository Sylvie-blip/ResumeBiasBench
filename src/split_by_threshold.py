#!/usr/bin/env python3
"""
split_by_threshold.py — Split a metrics CSV into two groups based on
a minimum human_win_count threshold from a separate wins CSV.

Usage:
    python split_by_threshold.py wins.csv metrics.csv --threshold 3

Arguments:
    wins_csv      CSV with at least 'resume_id' (int) and 'human_win_count' columns
    metrics_csv   CSV with a filename column like 'resume_<id>.txt' and metric columns
    --threshold   Minimum human_win_count to be in the "meets" group (default: 1)
    --id-col      Column name in metrics CSV that holds the filename (default: auto-detect)
    --out-meets   Output filename for rows meeting threshold (default: meets_threshold.csv)
    --out-below   Output filename for rows below threshold  (default: below_threshold.csv)
"""

import argparse
import re
import sys
import pandas as pd


def extract_resume_id(filename: str) -> int | None:
    """Pull the first integer out of a filename string, e.g. 'resume_42.txt' → 42."""
    m = re.search(r"\d+", str(filename))
    return int(m.group()) if m else None


def detect_filename_col(df: pd.DataFrame) -> str:
    """Heuristically find the column that holds filenames like 'resume_<id>.txt'."""
    for col in df.columns:
        sample = df[col].dropna().astype(str)
        # Column looks like filenames if most values contain digits and a dot
        has_ext = sample.str.contains(r"\d+.*\.", regex=True)
        if has_ext.mean() > 0.5:
            return col
    raise ValueError(
        "Could not auto-detect the filename column in the metrics CSV. "
        "Use --id-col to specify it explicitly."
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Split a metrics CSV by human_win_count threshold."
    )
    parser.add_argument("wins_csv", help="CSV containing resume_id and human_win_count")
    parser.add_argument("metrics_csv", help="CSV containing filenames and metrics")
    parser.add_argument(
        "--threshold", type=float, default=1,
        help="Minimum human_win_count to be in the 'meets' group (default: 1)"
    )
    parser.add_argument(
        "--id-col", default=None,
        help="Column in metrics CSV that holds the filename (auto-detected if omitted)"
    )
    parser.add_argument(
        "--out-meets", default="meets_threshold.csv",
        help="Output path for rows meeting the threshold (default: meets_threshold.csv)"
    )
    parser.add_argument(
        "--out-below", default="below_threshold.csv",
        help="Output path for rows below the threshold (default: below_threshold.csv)"
    )
    args = parser.parse_args()

    # ── load ─────────────────────────────────────────────────────────────────
    wins = pd.read_csv(args.wins_csv)
    metrics = pd.read_csv(args.metrics_csv)

    # Validate wins CSV
    for col in ("resume_id", "human_win_count"):
        if col not in wins.columns:
            print(f"ERROR: '{col}' column not found in {args.wins_csv}.", file=sys.stderr)
            print(f"  Available columns: {wins.columns.tolist()}", file=sys.stderr)
            sys.exit(1)

    # ── filter to Human-labeled rows only ───────────────────────────────────
    if "group" not in metrics.columns:
        print(f"ERROR: 'group' column not found in {args.metrics_csv}.", file=sys.stderr)
        print(f"  Available columns: {metrics.columns.tolist()}", file=sys.stderr)
        sys.exit(1)

    total_rows = len(metrics)
    metrics = metrics[metrics["group"].str.strip().str.upper() == "HUMAN"].copy()
    print(f"Filtered to Human rows: {len(metrics)} of {total_rows} rows kept.")

    if len(metrics) == 0:
        print("ERROR: no rows with group='Human' found.", file=sys.stderr)
        sys.exit(1)

    # ── detect filename column ───────────────────────────────────────────────
    id_col = args.id_col or detect_filename_col(metrics)
    print(f"Using '{id_col}' as the filename column in {args.metrics_csv}.")

    # ── parse resume_id from filenames ───────────────────────────────────────
    metrics = metrics.copy()
    metrics["_resume_id"] = metrics[id_col].apply(extract_resume_id)

    missing = metrics["_resume_id"].isna().sum()
    if missing:
        print(f"Warning: {missing} row(s) had no parseable ID in '{id_col}' — they will go to below_threshold.")

    # ── build threshold lookup ───────────────────────────────────────────────
    wins_clean = wins[["resume_id", "human_win_count"]].drop_duplicates("resume_id")
    threshold_map = wins_clean.set_index("resume_id")["human_win_count"]

    metrics["_win_count"] = metrics["_resume_id"].map(threshold_map)

    unmatched = metrics["_win_count"].isna().sum()
    if unmatched:
        print(f"Warning: {unmatched} row(s) in metrics had no matching resume_id in wins CSV — they will go to below_threshold.")

    # ── split ────────────────────────────────────────────────────────────────
    meets_mask = metrics["_win_count"] >= args.threshold
    meets = metrics[meets_mask].drop(columns=["_resume_id", "_win_count"])
    below = metrics[~meets_mask].drop(columns=["_resume_id", "_win_count"])

    # ── output ───────────────────────────────────────────────────────────────
    meets.to_csv(args.out_meets, index=False)
    below.to_csv(args.out_below, index=False)

    print(f"\nThreshold : human_win_count >= {args.threshold}")
    print(f"  Meets   : {len(meets):>5} rows  →  {args.out_meets}")
    print(f"  Below   : {len(below):>5} rows  →  {args.out_below}")
    print(f"  Total   : {len(metrics):>5} rows")


if __name__ == "__main__":
    main()