#!/usr/bin/env python3
"""
compare_metrics.py — Mann-Whitney U test across metric columns in CSV(s).

Usage:
    # 1 CSV with a 'group' column (values: 'AI' / 'Human'):
    python compare_metrics.py data.csv

    # 2 CSVs (AI-favored vs AI-disfavored):
    python compare_metrics.py ai_favored.csv ai_disfavored.csv
"""

import sys
import argparse
import csv
from pathlib import Path
import pandas as pd
from scipy.stats import mannwhitneyu

_REPO_ROOT = Path(__file__).resolve().parents[2]
EFFECT_SIZES_OUT_PATH = _REPO_ROOT / "results" / "mannwhitney_effect_sizes.csv"


def write_effect_sizes_csv(results: list[dict], label_a: str, label_b: str, out_path: Path) -> None:
    """Write one row per feature with its Mann-Whitney effect size (additive;
    console output above is unchanged)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["metric", f"n_{label_a}", f"n_{label_b}", "U", "p", "r", "sig", "label_a", "label_b"]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            row = {k: r.get(k) for k in fieldnames if k in r}
            row["label_a"] = label_a
            row["label_b"] = label_b
            writer.writerow(row)
    print(f"Wrote {out_path}")


# ── helpers ──────────────────────────────────────────────────────────────────

def effect_size_r(stat: float, n1: int, n2: int) -> float:
    """Rank-biserial correlation r = 1 - 2U / (n1 * n2)."""
    return 1 - (2 * stat) / (n1 * n2)


def stars(p: float) -> str:
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def numeric_metrics(df: pd.DataFrame, exclude: list[str] | None = None) -> list[str]:
    """Return numeric columns, skipping any in the exclude list."""
    exclude = set(exclude or [])
    return [c for c in df.select_dtypes("number").columns if c not in exclude]


def run_test(col: str, a: pd.Series, b: pd.Series, label_a: str, label_b: str) -> dict:
    """Run Mann-Whitney U and return a result dict."""
    a = a.dropna()
    b = b.dropna()
    if len(a) == 0 or len(b) == 0:
        return {
            "metric": col,
            f"n_{label_a}": len(a),
            f"n_{label_b}": len(b),
            "U": None,
            "p": None,
            "r": None,
            "sig": "—",
            "note": "insufficient data",
        }
    stat, p = mannwhitneyu(a, b, alternative="two-sided")
    r = effect_size_r(stat, len(a), len(b))
    return {
        "metric": col,
        f"n_{label_a}": len(a),
        f"n_{label_b}": len(b),
        "U": round(stat, 4),
        "p": round(p, 6),
        "r": round(r, 4),
        "sig": stars(p),
        "note": "",
    }


def print_results(results: list[dict], label_a: str, label_b: str) -> None:
    col_w = max(len(r["metric"]) for r in results) + 2
    key_a = f"n_{label_a}"
    key_b = f"n_{label_b}"
    header = (
        f"{'Metric':<{col_w}} {key_a:>8} {key_b:>8} {'U':>12} {'p':>10} {'r':>8} {'sig':>5}"
    )
    sep = "─" * len(header)
    print(f"\n  {label_a}  vs  {label_b}")
    print(sep)
    print(header)
    print(sep)
    for r in results:
        U_str = f"{r['U']:.4f}" if r["U"] is not None else "—"
        p_str = f"{r['p']:.6f}" if r["p"] is not None else "—"
        r_str = f"{r['r']:.4f}" if r["r"] is not None else "—"
        note = f"  ← {r['note']}" if r["note"] else ""
        print(
            f"{r['metric']:<{col_w}} {r[key_a]:>8} {r[key_b]:>8} "
            f"{U_str:>12} {p_str:>10} {r_str:>8} {r['sig']:>5}{note}"
        )
    print(sep)
    print("Significance: *** p<0.001  ** p<0.01  * p<0.05  ns = not significant")
    print("r: rank-biserial correlation (effect size); positive → higher in first group\n")


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Mann-Whitney U metric comparison.")
    parser.add_argument("csvs", nargs="+", metavar="CSV", help="1 or 2 CSV files")
    args = parser.parse_args()

    if len(args.csvs) == 1:
        # ── single-CSV mode ──────────────────────────────────────────────────
        df = pd.read_csv(args.csvs[0])

        if "group" not in df.columns:
            print(
                "ERROR: single-CSV mode requires a 'group' column with values 'AI' / 'Human'.",
                file=sys.stderr,
            )
            sys.exit(1)

        groups = df["group"].unique().tolist()
        print(f"Groups found: {groups}")

        ai_mask = df["group"].str.strip().str.upper() == "AI"
        human_mask = df["group"].str.strip().str.upper() == "HUMAN"

        if not ai_mask.any():
            print("ERROR: no rows with group='AI' found.", file=sys.stderr)
            sys.exit(1)
        if not human_mask.any():
            print("ERROR: no rows with group='Human' found.", file=sys.stderr)
            sys.exit(1)

        ai_df = df[ai_mask]
        human_df = df[human_mask]
        metrics = numeric_metrics(df, exclude=["group"])

        print(f"\nLoaded '{args.csvs[0]}': {len(df)} rows, {len(metrics)} numeric metrics.")
        print(f"  AI rows   : {len(ai_df)}")
        print(f"  Human rows: {len(human_df)}")

        results = [
            run_test(col, ai_df[col], human_df[col], "AI", "Human")
            for col in metrics
        ]
        print_results(results, "AI", "Human")
        write_effect_sizes_csv(results, "AI", "Human", EFFECT_SIZES_OUT_PATH)

    elif len(args.csvs) == 2:
        # ── two-CSV mode ─────────────────────────────────────────────────────
        df_a = pd.read_csv(args.csvs[0])
        df_b = pd.read_csv(args.csvs[1])

        metrics_a = set(numeric_metrics(df_a))
        metrics_b = set(numeric_metrics(df_b))
        shared = sorted(metrics_a & metrics_b)

        only_a = metrics_a - metrics_b
        only_b = metrics_b - metrics_a
        if only_a:
            print(f"Note: columns only in '{args.csvs[0]}' (skipped): {sorted(only_a)}")
        if only_b:
            print(f"Note: columns only in '{args.csvs[1]}' (skipped): {sorted(only_b)}")

        label_a = args.csvs[0].removesuffix(".csv")
        label_b = args.csvs[1].removesuffix(".csv")

        print(
            f"\nLoaded '{args.csvs[0]}': {len(df_a)} rows  |  "
            f"'{args.csvs[1]}': {len(df_b)} rows"
        )
        print(f"Shared numeric metrics: {len(shared)}")

        results = [
            run_test(col, df_a[col], df_b[col], label_a, label_b)
            for col in shared
        ]
        print_results(results, label_a, label_b)
        write_effect_sizes_csv(results, label_a, label_b, EFFECT_SIZES_OUT_PATH)

    else:
        parser.error("Pass exactly 1 or 2 CSV files.")


if __name__ == "__main__":
    main()