"""Priority 5 (writing prompt, but verified against real data rather than
taken on faith): does linguistic style predict which HUMAN-written resume
wins, among human-written resumes only (i.e. holding "is this AI-polished"
constant at "no" and asking whether style still matters)?

The instructions asserted specific numbers already existed (|r| <= 0.173,
token entropy reversed direction). Verification process, since the claim
could not be taken on faith:

1. data/results/new_metrics_results_winning / _losing (the only precomputed
   split in this repo) only has 5 of 9 features and, critically, does NOT
   include token entropy at all - the specific claim about entropy cannot be
   checked against those files.
2. Its winning/losing membership (917 vs 1,192) comes from
   data/results/humanresumes_winning / _losing, built by split_by_threshold.py
   against the OLD, incomplete human_win_tally.csv (missing granite4:3b, only
   3-7 of 8 models represented depending on the pair). That tally's 1,667
   rows are exactly "human won at least once" - a lenient definition.
3. Recomputing that same "human won >=1 of 7 models" definition from the
   current, complete, validated screening_outcomes.csv gives 1,667 won /
   442 lost pairs - an exact match to the old tally's row count, confirming
   this is the historical definition - and, joined against the real
   data/results/new_merged.csv features (all 9, including entropy), reproduces
   the claimed numbers closely: every |r| <= 0.173 (entropy the largest in
   magnitude at ~-0.12), entropy's direction reversed exactly as claimed.
   This is the analysis reported as primary below.
4. As a robustness check (not the headline), a stricter "human won a MAJORITY
   of 7 models" definition (114 won / 1,995 lost - a much smaller, rarer
   group) is also reported: qualitatively the same story (entropy reversed,
   still the largest effect) but numerically noisier and larger in magnitude
   (entropy r ~ -0.32) given the much smaller n_won. Both are shown so the
   choice of win-definition threshold is visible rather than picked silently.
"""

import sys
from pathlib import Path

import pandas as pd
from scipy.stats import mannwhitneyu

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import analysis  # noqa: E402

FEATURES = [
    "entropy", "perplexity", "rolling_entropy", "sentence_entropy_variance",
    "lexical_density", "pattern_regularity", "repetition", "sentence_evenness",
    "tonal_stability",
]


def effect_size_r(stat: float, n1: int, n2: int) -> float:
    return 1 - (2 * stat) / (n1 * n2)


def build_human_win_membership(screening_rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(screening_rows)
    df["is_human_win"] = (df["winner"] == "HUMAN").astype(int)
    agg = df.groupby("pair_id")["is_human_win"].agg(["sum", "count"]).reset_index()
    agg = agg.rename(columns={"sum": "human_wins", "count": "n_models"})
    agg["human_won_any"] = agg["human_wins"] >= 1
    agg["human_won_majority"] = agg["human_wins"] > (agg["n_models"] / 2)
    return agg


def _compare(joined: pd.DataFrame, won_mask: pd.Series) -> list[dict]:
    won = joined[won_mask]
    lost = joined[~won_mask]
    results = []
    for feat in FEATURES:
        a = won[feat].dropna()
        b = lost[feat].dropna()
        stat, p = mannwhitneyu(a, b, alternative="two-sided")
        r = effect_size_r(stat, len(a), len(b))
        results.append({"feature": feat, "n_won": len(a), "n_lost": len(b),
                         "U": stat, "p": p, "r": r})
    return results


def run(repo_root: Path) -> dict:
    screening_rows = analysis.load_screening_outcomes(
        str(repo_root / "results" / "qual_audit" / "screening_outcomes.csv")
    )
    membership = build_human_win_membership(screening_rows)

    merged_features = pd.read_csv(repo_root / "data" / "results" / "new_merged.csv")
    merged_features["pair_id"] = (
        merged_features["file"].str.replace("_AI", "", regex=False).str.replace(".txt", "", regex=False)
    )
    human_features = merged_features[merged_features["group"] == "Human"]
    joined = human_features.merge(membership, on="pair_id", how="inner")

    any_results = _compare(joined, joined["human_won_any"])
    majority_results = _compare(joined, joined["human_won_majority"])

    return {
        "any": {
            "n_won": int(joined["human_won_any"].sum()),
            "n_lost": int((~joined["human_won_any"]).sum()),
            "results": any_results,
        },
        "majority": {
            "n_won": int(joined["human_won_majority"].sum()),
            "n_lost": int((~joined["human_won_majority"]).sum()),
            "results": majority_results,
        },
    }


def _print_table(label: str, block: dict):
    print(f"--- {label}: human-won={block['n_won']}, human-lost={block['n_lost']} ---")
    print(f"{'Feature':<28} {'r':>8} {'p':>12}")
    for row in sorted(block["results"], key=lambda x: -abs(x["r"])):
        print(f"{row['feature']:<28} {row['r']:>8.4f} {row['p']:>12.3e}")
    print()


if __name__ == "__main__":
    result = run(_REPO_ROOT)
    _print_table("PRIMARY: human won >=1 of 7 models (matches historical definition)", result["any"])
    _print_table("ROBUSTNESS CHECK: human won a majority (>=4) of 7 models", result["majority"])

    import csv
    out_path = _REPO_ROOT / "results" / "qual_audit" / "winning_losing_style.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["definition", "feature", "n_won", "n_lost", "U", "p", "r"])
        for label, block in result.items():
            for row in block["results"]:
                w.writerow([label, row["feature"], row["n_won"], row["n_lost"], row["U"], row["p"], row["r"]])
    print(f"Wrote {out_path}")
