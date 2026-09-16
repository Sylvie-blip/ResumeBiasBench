"""Priority 4 (reviewer follow-up, not in the original audit spec): logistic
regression predicting which version won from the per-pair linguistic feature
deltas (data/results/new_merged.csv), pooled across all 7 available models
with model fixed effects and pair-clustered standard errors.

CRITICAL (per instructions): token entropy and perplexity are monotonic
transforms of one another (perplexity = 2^entropy in this codebase's own
Metrics.py), so including both makes individual coefficients uninterpretable
- this is exactly the mechanism behind the existing paper's Table 4 showing
+5.990 (entropy) / -6.270 (perplexity), a collinearity artifact rather than
two independent findings. new_merged.csv's 9 columns already fold "perplexity"
in as its own column; it is dropped here, leaving 8 features:
entropy, rolling_entropy, sentence_entropy_variance, lexical_density,
pattern_regularity, repetition, sentence_evenness, tonal_stability.

Note on feature-name correspondence: new_merged.csv's column names do not map
1:1 onto the paper's named formulas (e.g. no separate sliding-window-variance
column; "repetition" and "sentence_evenness" stand in for TTR-like and
length-dispersion-like quantities respectively). This regression uses
new_merged.csv's 9 real columns as "the nine features" since that is the
actual data available, and this discrepancy is disclosed rather than papered
over with an assumed correspondence.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import roc_auc_score

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import analysis  # noqa: E402

ALL_FEATURES = [
    "entropy", "perplexity", "rolling_entropy", "sentence_entropy_variance",
    "lexical_density", "pattern_regularity", "repetition", "sentence_evenness",
    "tonal_stability",
]
DROPPED_FOR_COLLINEARITY = "perplexity"
FEATURES = [f for f in ALL_FEATURES if f != DROPPED_FOR_COLLINEARITY]


def build_pair_deltas(new_merged_path: str) -> pd.DataFrame:
    df = pd.read_csv(new_merged_path)
    df["pair_id"] = df["file"].str.replace("_AI", "", regex=False).str.replace(".txt", "", regex=False)

    human = df[df["group"] == "Human"].set_index("pair_id")[FEATURES]
    ai = df[df["group"] == "AI"].set_index("pair_id")[FEATURES]

    common = human.index.intersection(ai.index)
    deltas = (ai.loc[common] - human.loc[common]).add_prefix("delta_")
    deltas = deltas.reset_index()
    return deltas


def fit_pooled_regression(screening_rows: list[dict], deltas: pd.DataFrame) -> dict:
    df = pd.DataFrame(screening_rows).merge(deltas, on="pair_id", how="inner")
    df["y"] = (df["winner"] == "AI").astype(float)

    delta_cols = [f"delta_{f}" for f in FEATURES]
    # Standardize deltas so coefficients are comparable across features
    # (matches the paper's own StandardScaler convention for Table 4).
    X = df[delta_cols].copy()
    means = X.mean()
    stds = X.std()
    X_std = (X - means) / stds

    model_dummies = pd.get_dummies(df["model"], prefix="model", drop_first=True).astype(float)

    X_full = pd.concat([X_std, model_dummies], axis=1)
    X_full = sm.add_constant(X_full)

    fit = sm.Logit(df["y"], X_full).fit(disp=0)
    fit_clustered = sm.Logit(df["y"], X_full).fit(
        disp=0, cov_type="cluster", cov_kwds={"groups": df["pair_id"]}
    )

    pred_prob = fit_clustered.predict(X_full)
    auc = roc_auc_score(df["y"], pred_prob)

    coef_table = pd.DataFrame({
        "coef": fit_clustered.params,
        "se_clustered": fit_clustered.bse,
        "p_clustered": fit_clustered.pvalues,
    })

    return {
        "n_obs": len(df),
        "n_pairs": df["pair_id"].nunique(),
        "coef_table": coef_table,
        "auc": auc,
        "feature_cols": delta_cols,
        "model": fit_clustered,
    }


if __name__ == "__main__":
    deltas = build_pair_deltas(str(_REPO_ROOT / "data" / "results" / "new_merged.csv"))
    print(f"Pairs with computable deltas: {len(deltas)}")

    screening_rows = analysis.load_screening_outcomes(
        str(_REPO_ROOT / "results" / "qual_audit" / "screening_outcomes.csv")
    )
    result = fit_pooled_regression(screening_rows, deltas)

    print(f"n_obs={result['n_obs']}, n_pairs={result['n_pairs']}, AUC={result['auc']:.4f}\n")
    print("Coefficients (pair-clustered SEs), standardized deltas, 8 features "
          f"(dropped: {DROPPED_FOR_COLLINEARITY}):")
    ct = result["coef_table"]
    feat_rows = ct.loc[[c for c in ct.index if c.startswith("delta_")]]
    feat_rows = feat_rows.reindex(feat_rows["coef"].abs().sort_values(ascending=False).index)
    print(feat_rows.to_string())
    print()
    model_rows = ct.loc[[c for c in ct.index if c.startswith("model_")]]
    print("Model fixed effects (relative to reference model):")
    print(model_rows.to_string())

    out_path = _REPO_ROOT / "results" / "qual_audit" / "feature_regression_coefficients.csv"
    ct.to_csv(out_path)
    print(f"\nWrote {out_path}")
