"""Priority 3 (reviewer follow-up, not in the original audit spec): does the
AI polisher simply writing more explain the preference, independent of any
qualification content?

Token length uses the paper's own tokenizer (re.findall(r"\\b[a-zA-Z']+\\b",
text.lower()), per the paper's "Data Preprocessing" section) rather than a
naive .split() word count, so the comparison is apples-to-apples with the
paper's own normalization pipeline.

No LLM calls; reuses the already-parsed screening_outcomes.csv for outcomes.
"""

import csv
import re
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
from scipy import stats as sp_stats

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import analysis  # noqa: E402

_TOKEN_RX = re.compile(r"\b[a-zA-Z']+\b")


def token_count(text: str) -> int:
    return len(_TOKEN_RX.findall(text.lower()))


def build_length_table(human_dir: Path, ai_dir: Path) -> pd.DataFrame:
    rows = []
    human_ids = {p.stem for p in human_dir.glob("*.txt")}
    ai_ids = {p.stem.removesuffix("_AI") for p in ai_dir.glob("*_AI.txt")}
    for pid in sorted(human_ids & ai_ids, key=int):
        orig_text = (human_dir / f"{pid}.txt").read_text(encoding="utf-8", errors="replace")
        pol_text = (ai_dir / f"{pid}_AI.txt").read_text(encoding="utf-8", errors="replace")
        orig_n = token_count(orig_text)
        pol_n = token_count(pol_text)
        rows.append({"pair_id": pid, "orig_tokens": orig_n, "pol_tokens": pol_n,
                      "delta_tokens": pol_n - orig_n})
    return pd.DataFrame(rows)


def pair_ai_win_fraction(screening_rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(screening_rows)
    df["is_ai_win"] = (df["winner"] == "AI").astype(float)
    agg = df.groupby("pair_id")["is_ai_win"].agg(["mean", "count"]).reset_index()
    agg = agg.rename(columns={"mean": "ai_win_fraction", "count": "n_models"})
    return agg


def run(repo_root: Path) -> dict:
    length_df = build_length_table(repo_root / "data" / "humanresumes", repo_root / "data" / "airesumes")

    screening_rows = analysis.load_screening_outcomes(
        str(repo_root / "results" / "qual_audit" / "screening_outcomes.csv")
    )
    win_frac_df = pair_ai_win_fraction(screening_rows)

    merged = length_df.merge(win_frac_df, on="pair_id", how="inner")

    pearson_r, pearson_p = sp_stats.pearsonr(merged["delta_tokens"], merged["ai_win_fraction"])
    spearman_r, spearman_p = sp_stats.spearmanr(merged["delta_tokens"], merged["ai_win_fraction"])

    not_longer_ids = set(merged.loc[merged["delta_tokens"] <= 0, "pair_id"])
    not_longer_decisions = [r for r in screening_rows if r["pair_id"] in not_longer_ids]
    nl_k, nl_n, nl_rate = analysis.rate(not_longer_decisions)
    nl_test = analysis.cluster_robust_group_test(
        screening_rows, not_longer_ids
    ) if not_longer_ids and len(not_longer_ids) < len(merged) else None

    return {
        "length_df": merged,
        "mean_delta": merged["delta_tokens"].mean(),
        "median_delta": merged["delta_tokens"].median(),
        "std_delta": merged["delta_tokens"].std(),
        "pct_longer": (merged["delta_tokens"] > 0).mean(),
        "pearson_r": pearson_r, "pearson_p": pearson_p,
        "spearman_r": spearman_r, "spearman_p": spearman_p,
        "not_longer_pair_count": len(not_longer_ids),
        "not_longer_k": nl_k, "not_longer_n": nl_n, "not_longer_rate": nl_rate,
        "not_longer_test": nl_test,
    }


if __name__ == "__main__":
    result = run(_REPO_ROOT)
    df = result["length_df"]
    print(f"Pairs analyzed: {len(df)}")
    print(f"Mean token delta (polished - original): {result['mean_delta']:.1f}")
    print(f"Median token delta: {result['median_delta']:.1f}")
    print(f"Std token delta: {result['std_delta']:.1f}")
    print(f"% pairs where polished is longer: {result['pct_longer']:.1%}")
    print()
    print(f"Pearson r (delta_tokens vs ai_win_fraction): {result['pearson_r']:.4f}, p={result['pearson_p']:.3e}")
    print(f"Spearman r: {result['spearman_r']:.4f}, p={result['spearman_p']:.3e}")
    print()
    print(f"Pairs where polished <= original length: {result['not_longer_pair_count']}")
    print(f"  AI-win rate in this subset: {result['not_longer_k']}/{result['not_longer_n']} = {result['not_longer_rate']:.3f}")
    if result["not_longer_test"]:
        ct = result["not_longer_test"]
        print(f"  cluster-robust 95% CI: [{ct['ci_95'][0]:.3f}, {ct['ci_95'][1]:.3f}]")
        print(f"  p vs 0.5 (clustered): {ct['p_preserved_vs_0.5']:.3e}")

    out_path = _REPO_ROOT / "results" / "qual_audit" / "length_check.csv"
    df.to_csv(out_path, index=False)
    print(f"\nWrote {out_path}")
