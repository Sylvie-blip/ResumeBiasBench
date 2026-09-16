"""Step 4: join screening outcomes to preservation labels and recompute the
AI-polished win rate within the preserved subset.

Methodology notes (see plan for full rationale):
- 7 of the paper's 8 models now have a complete (2,109-pair) raw-decision file
  at the repo root (see parse_screeners.py); only granite4:3b is unavailable.
  This module still always reports N alongside every rate rather than
  assuming full coverage.
- "Pooled" rate = every (pair, model) decision row is one Bernoulli trial, so
  a pair with multiple available model decisions contributes multiple trials.
  This is stated explicitly wherever a pooled N is printed.

Inference (PRIORITY 0 FIX): pooled decisions are nested within pairs - each
pair is judged by up to 7 models, and within-pair outcomes are correlated (a
pair that reads as strongly edited tends to win across models). Treating the
2,688-ish pooled decisions as independent observations understates the true
variance. All pooled and per-model tests below are computed via
`cluster_robust_group_test`: an OLS regression of the binary AI-win indicator
on a preserved/non-preserved indicator, with standard errors clustered on
pair_id (Cameron & Miller 2015 cluster-robust inference; t-distribution with
n_clusters - 1 degrees of freedom). This has two benefits over the earlier
Wilson-CI + two-proportion-z-test approach:
  1. It correctly widens the interval to reflect within-pair correlation.
  2. It compares the preserved subset against its true complement (the
     non-preserved pairs) rather than against a "local baseline" that
     partially contains the subset itself - the old design's comparison group
     overlapped with the subset being tested, which biases toward finding no
     difference for a reason unrelated to clustering. Per-model comparisons
     (one decision per pair per model, so no repeated-measurement clustering
     within a single model) still benefit from this same complement-based
     redesign, which is why it is applied uniformly rather than only where
     clustering strictly applies.
"""

import csv
import sys
from pathlib import Path

import pandas as pd
import statsmodels.api as sm
from scipy import stats as sp_stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from preservation import THRESHOLDS  # noqa: E402

# Paper's Table 1 win rate per model. Ordering/values as given in the audit
# spec, matching the README's 8-model list (qwen2.5:3b/14b/32b, gemma3:4b/12b,
# mistral-small3.2:latest, granite4:3b, cogito:3b) -> (0.602, 0.888, 0.863,
# 0.716, 0.790, 0.793, 0.785, 0.880). granite4:3b (0.785) has no raw decisions
# available and so never appears as a key here.
PAPER_TABLE1_RATE = {
    "qwen2.5:3b": 0.602,
    "qwen2.5:14b": 0.888,
    "qwen2.5:32b": 0.863,
    "gemma3:4b": 0.716,
    "gemma3:12b": 0.790,
    "mistral-small3.2:latest": 0.793,
    "cogito:3b": 0.880,
}
PAPER_FULL_CORPUS_POOLED_RATE = 0.790


def load_csv(path: str) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_screening_outcomes(path: str) -> list[dict]:
    return load_csv(path)


def load_preservation_scores(path: str) -> list[dict]:
    rows = load_csv(path)
    for r in rows:
        r["undefined_retention"] = r["undefined_retention"] == "True"
        r["placeholder_damaged"] = r["placeholder_damaged"] == "True"
        for t in THRESHOLDS:
            r[f"preserved_{t:.2f}"] = r[f"preserved_{t:.2f}"] == "True"
    return rows


def preserved_ids_at(preservation_rows: list[dict], threshold: float) -> set[str]:
    key = f"preserved_{threshold:.2f}"
    return {r["pair_id"] for r in preservation_rows if r[key]}


def cluster_robust_rate(decisions: list[dict]) -> dict:
    """Cluster-robust mean AI-win rate (clustered on pair_id) with a 95% CI
    and a t-test vs 0.5. Used standalone for per-category rates (Priority 2)
    and internally by `cluster_robust_group_test` as its no-complement
    fallback."""
    if not decisions:
        return None

    df = pd.DataFrame(decisions)
    df["y"] = (df["winner"] == "AI").astype(float)
    n_clusters = df["pair_id"].nunique()

    X = pd.DataFrame({"const": 1.0}, index=df.index)
    model = sm.OLS(df["y"], X).fit(cov_type="cluster", cov_kwds={"groups": df["pair_id"]})
    mean = model.params["const"]
    se = model.bse["const"]
    dfree = max(n_clusters - 1, 1)
    tcrit = sp_stats.t.ppf(0.975, dfree)
    t_half = (mean - 0.5) / se if se > 0 else float("nan")
    p_half = 2 * sp_stats.t.sf(abs(t_half), dfree) if se > 0 else None
    return {
        "mean": mean, "se": se,
        "ci_95": (mean - tcrit * se, mean + tcrit * se),
        "p_vs_0.5": p_half, "n_clusters": n_clusters, "n_obs": len(df),
    }


def cluster_robust_group_test(decisions: list[dict], subset_ids: set[str]) -> dict:
    """OLS of AI-win indicator on a preserved/non-preserved indicator, SEs
    clustered on pair_id. decisions: list of {pair_id, winner}. subset_ids:
    pair_ids in the "preserved" group; every other pair_id present in
    `decisions` is the complement.

    Returns preserved-group mean with a cluster-robust CI and a t-test vs 0.5,
    plus the preserved-vs-complement difference with its own cluster-robust
    t-test - both computed off the same fitted model, so the comparison group
    is always the true complement, never an overlapping superset.
    """
    if not decisions:
        return None

    df = pd.DataFrame(decisions)
    df["y"] = (df["winner"] == "AI").astype(float)
    df["preserved"] = df["pair_id"].isin(subset_ids).astype(float)

    n_clusters = df["pair_id"].nunique()
    if df["preserved"].nunique() < 2 or n_clusters < 2:
        # Degenerate: no complement to compare against (e.g. empty or
        # all-one-group subset). Fall back to a simple cluster-robust mean.
        simple = cluster_robust_rate(decisions)
        return {
            "preserved_mean": simple["mean"], "complement_mean": None,
            "diff": None, "se_diff": None, "p_diff_vs_0": None,
            "se_preserved_mean": simple["se"], "ci_95": simple["ci_95"],
            "p_preserved_vs_0.5": simple["p_vs_0.5"],
            "n_clusters": simple["n_clusters"], "n_obs": simple["n_obs"],
        }

    X = sm.add_constant(df[["preserved"]])
    model = sm.OLS(df["y"], X).fit(
        cov_type="cluster", cov_kwds={"groups": df["pair_id"]}
    )
    dfree = max(n_clusters - 1, 1)

    intercept = model.params["const"]       # complement (non-preserved) mean
    coef = model.params["preserved"]        # preserved - complement
    preserved_mean = intercept + coef

    cov = model.cov_params()
    var_preserved_mean = (
        cov.loc["const", "const"] + cov.loc["preserved", "preserved"]
        + 2 * cov.loc["const", "preserved"]
    )
    se_preserved_mean = var_preserved_mean ** 0.5

    se_diff = model.bse["preserved"]
    t_diff = coef / se_diff if se_diff > 0 else float("nan")
    p_diff = 2 * sp_stats.t.sf(abs(t_diff), dfree) if se_diff > 0 else None

    t_half = (preserved_mean - 0.5) / se_preserved_mean if se_preserved_mean > 0 else float("nan")
    p_half = 2 * sp_stats.t.sf(abs(t_half), dfree) if se_preserved_mean > 0 else None

    tcrit = sp_stats.t.ppf(0.975, dfree)
    ci = (preserved_mean - tcrit * se_preserved_mean, preserved_mean + tcrit * se_preserved_mean)

    return {
        "preserved_mean": preserved_mean,
        "complement_mean": intercept,
        "diff": coef,
        "se_diff": se_diff,
        "p_diff_vs_0": p_diff,
        "se_preserved_mean": se_preserved_mean,
        "ci_95": ci,
        "p_preserved_vs_0.5": p_half,
        "n_clusters": n_clusters,
        "n_obs": len(df),
    }


def rate(decisions: list[dict]) -> tuple[int, int, float | None]:
    n = len(decisions)
    if n == 0:
        return 0, 0, None
    k = sum(1 for d in decisions if d["winner"] == "AI")
    return k, n, k / n


def compute_payoff(screening_rows: list[dict], preservation_rows: list[dict]) -> dict:
    models = sorted({r["model"] for r in screening_rows})
    by_model_all = {m: [r for r in screening_rows if r["model"] == m] for m in models}

    pooled_k, pooled_n, pooled_rate_all = rate(screening_rows)

    thresholds_result = {}
    for t in THRESHOLDS:
        subset_ids = preserved_ids_at(preservation_rows, t)

        pooled_test = cluster_robust_group_test(screening_rows, subset_ids)
        k, n, subset_rate_simple = rate(
            [r for r in screening_rows if r["pair_id"] in subset_ids]
        )

        per_model = {}
        for m in models:
            m_all = by_model_all[m]
            m_test = cluster_robust_group_test(m_all, subset_ids)
            mk, mn, mrate = rate([r for r in m_all if r["pair_id"] in subset_ids])
            bk, bn, brate = rate(m_all)  # this model's own full-corpus rate
            per_model[m] = {
                "subset_k": mk, "subset_n": mn, "subset_rate": mrate,
                "local_baseline_k": bk, "local_baseline_n": bn, "local_baseline_rate": brate,
                "paper_table1_rate": PAPER_TABLE1_RATE.get(m),
                "cluster_test": m_test,
            }

        thresholds_result[t] = {
            "subset_pair_count": len(subset_ids),
            "pooled_k": k, "pooled_n": n, "pooled_rate": subset_rate_simple,
            "cluster_test": pooled_test,
            "per_model": per_model,
        }

    return {
        "models": models,
        "full_corpus_local_baseline": {"k": pooled_k, "n": pooled_n, "rate": pooled_rate_all},
        "paper_full_corpus_pooled_rate": PAPER_FULL_CORPUS_POOLED_RATE,
        "by_threshold": thresholds_result,
    }


def composition_breakdown(preservation_rows: list[dict], subset_ids: set[str],
                           word_counts: dict[str, int]) -> dict:
    from collections import Counter

    full_cats = Counter(r["category"] for r in preservation_rows)
    subset_cats = Counter(r["category"] for r in preservation_rows if r["pair_id"] in subset_ids)

    full_n = len(preservation_rows)
    subset_n = len(subset_ids)

    per_category = {}
    for cat in sorted(full_cats):
        full_frac = full_cats[cat] / full_n
        subset_frac = subset_cats.get(cat, 0) / subset_n if subset_n else 0.0
        per_category[cat] = {
            "full_count": full_cats[cat],
            "full_frac": full_frac,
            "subset_count": subset_cats.get(cat, 0),
            "subset_frac": subset_frac,
            "ratio": (subset_frac / full_frac) if full_frac else float("nan"),
        }

    full_lengths = [word_counts[pid] for pid in word_counts]
    subset_lengths = [word_counts[pid] for pid in subset_ids if pid in word_counts]

    return {
        "per_category": per_category,
        "full_mean_length": sum(full_lengths) / len(full_lengths) if full_lengths else None,
        "subset_mean_length": sum(subset_lengths) / len(subset_lengths) if subset_lengths else None,
    }


def write_payoff_csv(payoff: dict, out_path: str):
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            f"=== Full-corpus local baseline ({len(payoff['models'])} models, "
            "all their available decisions, unfiltered) ==="
        ])
        writer.writerow(["k", "n", "rate", "paper_8model_2109pair_rate_for_context"])
        b = payoff["full_corpus_local_baseline"]
        writer.writerow([b["k"], b["n"], b["rate"], payoff["paper_full_corpus_pooled_rate"]])
        writer.writerow([])

        for t in THRESHOLDS:
            tr = payoff["by_threshold"][t]
            ct = tr["cluster_test"]
            writer.writerow([f"=== Threshold t={t:.2f} (cluster-robust, clustered on pair_id) ==="])
            writer.writerow(["subset_pair_count", tr["subset_pair_count"]])
            writer.writerow([
                "pooled_k", "pooled_n", "pooled_rate", "cluster_ci_95_lo", "cluster_ci_95_hi",
                "n_clusters", "p_vs_0.5_clustered", "diff_vs_complement", "p_diff_vs_complement_clustered",
            ])
            writer.writerow([
                tr["pooled_k"], tr["pooled_n"], tr["pooled_rate"],
                ct["ci_95"][0], ct["ci_95"][1], ct["n_clusters"],
                ct["p_preserved_vs_0.5"], ct["diff"], ct["p_diff_vs_0"],
            ])
            writer.writerow([])
            writer.writerow([
                "model", "subset_k", "subset_n", "subset_rate",
                "local_baseline_k", "local_baseline_n", "local_baseline_rate",
                "paper_table1_rate", "cluster_ci_95_lo", "cluster_ci_95_hi",
                "p_vs_0.5_clustered", "diff_vs_complement", "p_diff_vs_complement_clustered",
            ])
            for m, pm in tr["per_model"].items():
                mct = pm["cluster_test"]
                writer.writerow([
                    m, pm["subset_k"], pm["subset_n"], pm["subset_rate"],
                    pm["local_baseline_k"], pm["local_baseline_n"], pm["local_baseline_rate"],
                    pm["paper_table1_rate"],
                    mct["ci_95"][0] if mct else None, mct["ci_95"][1] if mct else None,
                    mct["p_preserved_vs_0.5"] if mct else None,
                    mct["diff"] if mct else None,
                    mct["p_diff_vs_0"] if mct else None,
                ])
            writer.writerow([])
