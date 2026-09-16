"""Priority 2 (reviewer's main ask, not in the original audit spec): does the
AI-polished preference hold across all 24 occupational categories, or does it
concentrate/reverse somewhere?

Groups ALL plain-text screening decisions (7 models x however many pairs are
in that category - NOT restricted to the preserved subset) by category.
Reports n pairs, n decisions, cluster-robust rate and 95% CI per category
(clustered on pair_id, same methodology as the main audit). Per spec
guidance: report CIs, not per-category significance tests, to sidestep a
multiple-comparisons objection across 24 categories.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analysis  # noqa: E402

UNDERPOWERED_THRESHOLD_PAIRS = 30  # BPO (17) and AUTOMOBILE (28) fall below this


def build_category_table(screening_rows: list[dict], preservation_rows: list[dict]) -> list[dict]:
    pair_to_category = {r["pair_id"]: r["category"] for r in preservation_rows}
    categories = sorted(set(pair_to_category.values()))

    results = []
    for cat in categories:
        cat_pair_ids = {pid for pid, c in pair_to_category.items() if c == cat}
        cat_decisions = [r for r in screening_rows if r["pair_id"] in cat_pair_ids]
        rate_result = analysis.cluster_robust_rate(cat_decisions)
        k, n, _ = analysis.rate(cat_decisions)
        results.append({
            "category": cat,
            "n_pairs": len(cat_pair_ids),
            "n_decisions": n,
            "ai_wins": k,
            "rate": rate_result["mean"] if rate_result else None,
            "ci_95": rate_result["ci_95"] if rate_result else (float("nan"), float("nan")),
            "p_vs_0.5": rate_result["p_vs_0.5"] if rate_result else None,
            "underpowered": len(cat_pair_ids) < UNDERPOWERED_THRESHOLD_PAIRS,
        })
    return sorted(results, key=lambda r: r["rate"] if r["rate"] is not None else -1)


def check_directional_consistency(category_table: list[dict]) -> dict:
    adequate = [r for r in category_table if not r["underpowered"]]
    below_half = [r for r in adequate if r["ci_95"][1] is not None and r["ci_95"][1] < 0.5]
    above_half_or_straddle = [r for r in adequate if r not in below_half]
    return {
        "n_adequate_categories": len(adequate),
        "n_below_half_ci": len(below_half),
        "reversed_categories": [r["category"] for r in below_half],
        "all_consistent": len(below_half) == 0,
    }


if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parents[2]
    screening_rows = analysis.load_screening_outcomes(
        str(repo_root / "results" / "qual_audit" / "screening_outcomes.csv")
    )
    preservation_rows = analysis.load_preservation_scores(
        str(repo_root / "results" / "qual_audit" / "preservation_scores.csv")
    )

    table = build_category_table(screening_rows, preservation_rows)
    consistency = check_directional_consistency(table)

    print(f"{'Category':<20} {'n_pairs':>8} {'n_dec':>8} {'rate':>7} {'95% CI':>18} {'underpowered':>13}")
    for r in table:
        ci_str = f"[{r['ci_95'][0]:.3f}, {r['ci_95'][1]:.3f}]"
        flag = " <-- UNDERPOWERED" if r["underpowered"] else ""
        print(f"{r['category']:<20} {r['n_pairs']:>8} {r['n_decisions']:>8} "
              f"{r['rate']:.3f} {ci_str:>18} {str(r['underpowered']):>13}{flag}")

    print()
    print(f"Adequately-powered categories (n_pairs >= {UNDERPOWERED_THRESHOLD_PAIRS}): {consistency['n_adequate_categories']}")
    print(f"Categories with CI entirely below 0.5 (reversed): {consistency['reversed_categories']}")
    print(f"Directionally consistent across adequately-powered categories: {consistency['all_consistent']}")

    import csv
    out_path = repo_root / "results" / "qual_audit" / "category_breakdown.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["category", "n_pairs", "n_decisions", "ai_wins", "rate", "ci_lo", "ci_hi", "p_vs_0.5", "underpowered"])
        for r in table:
            writer.writerow([r["category"], r["n_pairs"], r["n_decisions"], r["ai_wins"],
                              r["rate"], r["ci_95"][0], r["ci_95"][1], r["p_vs_0.5"], r["underpowered"]])
    print(f"\nWrote {out_path}")
