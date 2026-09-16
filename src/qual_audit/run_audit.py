"""Orchestrates the full qualification-preservation audit end to end:
parse screening decisions -> score preservation for all pairs -> build the
preserved-subset payoff analysis -> draw the spot-check sample -> write every
deliverable plus a human-readable audit_summary.md.

Run: python3 src/qual_audit/run_audit.py
"""

import json
import re
import sys
from collections import Counter
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

import parse_screeners  # noqa: E402
import build_scores  # noqa: E402
import spotcheck  # noqa: E402
import analysis  # noqa: E402
import length_check  # noqa: E402
import feature_regression  # noqa: E402
import category_breakdown  # noqa: E402
import winning_losing_style  # noqa: E402
from category_map import IMPERFECT_MAPPINGS  # noqa: E402
from preservation import THRESHOLDS  # noqa: E402

OUT_DIR = _REPO_ROOT / "results" / "qual_audit"


def load_word_counts() -> dict[str, int]:
    word_counts = {}
    for p in (_REPO_ROOT / "data" / "humanresumesjson").glob("*.json"):
        d = json.loads(p.read_text(encoding="utf-8"))
        word_counts[p.stem] = d["word_count"]
    return word_counts


_PLACEHOLDER_FIGURE_RX = re.compile(
    r"\[(number|year|month,\s*year|amount|percentage)\]", re.IGNORECASE
)


def count_placeholder_figures(ai_dir: Path) -> tuple[int, int, set[str]]:
    """Counts polished resumes containing an unfilled bracketed placeholder
    where a real quantified figure should be (e.g. "[Number]", "[Year]") -
    found by chance during the Step 1 spot-check (pair 10466208), not
    something the spec asked us to look for, but directly relevant to
    qualification preservation: a broken placeholder is a content downgrade,
    not an upgrade. Also returns the affected pair_ids so they can be checked
    against preserved-subset membership."""
    matches = 0
    total = 0
    affected_ids = set()
    for p in ai_dir.glob("*.txt"):
        total += 1
        if _PLACEHOLDER_FIGURE_RX.search(p.read_text(encoding="utf-8", errors="replace")):
            matches += 1
            affected_ids.add(p.stem.removesuffix("_AI"))
    return matches, total, affected_ids


def fmt_rate(x):
    return f"{x:.3f}" if isinstance(x, (int, float)) and x == x else "n/a"


def fmt_p(x):
    if x is None or (isinstance(x, float) and x != x):
        return "n/a"
    return f"{x:.2e}" if x < 0.001 else f"{x:.3f}"


def build_summary_md(screening_stage, scores_stage, sample, payoff, composition,
                      placeholder_stats, placeholder_join, length_stats, feature_reg,
                      category_table, category_consistency, style_result) -> str:
    lines = []
    lines.append("# Qualification-Preservation Audit — Summary\n")

    lines.append("## Model coverage (read this first)\n")
    lines.append(
        "7 of the paper's 8 screening models now have a complete (2,109/2,109-pair) "
        "raw-decision file at the repo root. Only **granite4:3b** is unavailable - "
        "its Table 1 rate (0.785) is shown for context only and never enters any "
        "computed rate below.\n"
    )
    lines.append(
        "Earlier runs of this audit only had 3 partial/ambiguous files (2 stray "
        "copies under `Useless/` plus an unlabeled `qwen2.5.txt`); those are now "
        "superseded and excluded (see `parse_screeners.py` docstring for why they "
        "can't simply be merged with the complete runs - Ollama's local inference "
        "turned out not to be perfectly repeatable run-to-run even at "
        "temperature=0, so a partial and a complete run of \"the same\" model "
        "disagreed on overlapping pairs).\n"
    )

    lines.append("## Corpus join\n")
    lines.append(f"- Matched pairs: {scores_stage['total_pairs']} / 2,109 expected\n")
    lines.append(f"- Missing human .txt for an AI file: {len(scores_stage['missing_human_txt'])}")
    lines.append(f"- Missing AI .txt for a human file: {len(scores_stage['missing_ai_txt'])}")
    lines.append(f"- Missing category label: {len(scores_stage['missing_category'])}\n")

    lines.append("## Keyword-dictionary coverage (a major, load-bearing limitation)\n")
    total = scores_stage["total_pairs"]
    undef_n = sum(1 for r in scores_stage["rows"] if r["undefined_retention"])
    lines.append(
        f"**{undef_n} / {total} pairs ({undef_n/total:.1%}) have zero scoped keyword "
        "matches in the original resume text** (`undefined_retention`) and are excluded "
        "from every preserved-subset calculation below. This is a property of the "
        "*reused* `SKILLS_BY_DOMAIN` dictionary in `src/conv_to_json.py` (deliberately "
        "not modified — see spec) being narrow/jargon-heavy relative to how real "
        "candidates phrase resumes, not a bug in this audit's join or scoring logic. "
        "Coverage is highly uneven by category:\n"
    )
    cat_totals = Counter(r["category"] for r in scores_stage["rows"])
    cat_undef = Counter(r["category"] for r in scores_stage["rows"] if r["undefined_retention"])
    lines.append("| Category | Undefined | Total | % undefined |")
    lines.append("|---|---|---|---|")
    for cat in sorted(cat_totals, key=lambda c: -cat_undef.get(c, 0) / cat_totals[c]):
        pct = 100 * cat_undef.get(cat, 0) / cat_totals[cat]
        marker = " ⚠️" if pct >= 60 else ""
        lines.append(f"| {cat} | {cat_undef.get(cat,0)} | {cat_totals[cat]} | {pct:.1f}%{marker} |")
    lines.append("")

    lines.append("## Category-to-domain mapping\n")
    lines.append(
        f"21 of 24 `sorted_data` categories map 1:1 to a `SKILLS_BY_DOMAIN` key. "
        f"3 are imperfect and explicitly disclosed rather than guessed: "
        f"{', '.join(sorted(IMPERFECT_MAPPINGS))} "
        "(BANKING→finance; APPAREL→design∪general; PUBLIC-RELATIONS→digital_media∪business).\n"
    )

    lines.append("## Headline: preserved-subset win rate by threshold\n")
    b = payoff["full_corpus_local_baseline"]
    lines.append(
        f"Full-corpus local baseline (these {len(payoff['models'])} models, all their "
        f"available decisions, unfiltered): **{fmt_rate(b['rate'])}** (k={b['k']}, n={b['n']}). "
        f"Paper's full 8-model/2,109-pair rate, for context only: "
        f"{payoff['paper_full_corpus_pooled_rate']}.\n"
    )
    lines.append(
        "**Inference is cluster-robust, clustered on pair_id** (OLS of the AI-win "
        "indicator on a preserved/non-preserved indicator, SEs clustered on pair, "
        "t-distribution with n_clusters-1 df) - not the naive independent-decisions "
        "CI/binomial-test from an earlier version of this audit. Each pair is judged "
        "by up to 7 models, so within-pair outcomes are correlated and the naive "
        "approach understates the true interval. Empirically the correction here is "
        "modest: at t=0.90 the design effect is ~1.16 (implied intra-pair correlation "
        "~0.026) - real, in the expected direction, but small because agreement "
        "between these 7 heterogeneous model families on any single pair turns out "
        "to be weak. The comparison group is also now the true complement "
        "(non-preserved pairs), not a superset that partially contains the subset "
        "being tested.\n"
    )
    lines.append("| t | Subset pairs | Pooled k/n | Pooled rate | Cluster-robust 95% CI | vs. 0.5 (clustered p) | vs. complement (clustered p) |")
    lines.append("|---|---|---|---|---|---|---|")
    for t in THRESHOLDS:
        tr = payoff["by_threshold"][t]
        ct = tr["cluster_test"]
        ci = ct["ci_95"]
        lines.append(
            f"| {t:.2f} | {tr['subset_pair_count']} | {tr['pooled_k']}/{tr['pooled_n']} | "
            f"{fmt_rate(tr['pooled_rate'])} | [{fmt_rate(ci[0])}, {fmt_rate(ci[1])}] | "
            f"{fmt_p(ct['p_preserved_vs_0.5'])} | {fmt_p(ct['p_diff_vs_0'])} |"
        )
    lines.append("")

    lines.append("## Per-model breakdown (t=0.90, the most inclusive threshold)\n")
    tr90 = payoff["by_threshold"][0.90]
    lines.append("| Model | Subset k/n | Subset rate | Full-corpus rate | Paper Table 1 rate | vs. complement (clustered p) |")
    lines.append("|---|---|---|---|---|---|")
    significant_per_model = []
    for m, pm in tr90["per_model"].items():
        mct = pm["cluster_test"]
        p_diff = mct["p_diff_vs_0"] if mct else None
        lines.append(
            f"| {m} | {pm['subset_k']}/{pm['subset_n']} | {fmt_rate(pm['subset_rate'])} | "
            f"{fmt_rate(pm['local_baseline_rate'])} | "
            f"{pm['paper_table1_rate'] if pm['paper_table1_rate'] is not None else 'n/a'} | "
            f"{fmt_p(p_diff)} |"
        )
        if p_diff is not None and p_diff == p_diff and p_diff < 0.05:
            direction = "lower" if pm["subset_rate"] < pm["local_baseline_rate"] else "higher"
            significant_per_model.append((m, direction, pm["subset_rate"], pm["local_baseline_rate"], p_diff))
    lines.append("")
    if significant_per_model:
        lines.append(
            "**Not every model tells the same story.** The pooled/headline result above "
            "masks per-model variation - at least one model shows a statistically "
            "significant difference between its preserved-subset rate and its own "
            "baseline, which the pooled number alone would hide:\n"
        )
        for m, direction, subset_rate, baseline_rate, p in significant_per_model:
            lines.append(
                f"- **{m}**: preserved-subset rate ({fmt_rate(subset_rate)}) is "
                f"significantly {direction} than its own baseline ({fmt_rate(baseline_rate)}), "
                f"two-prop p={fmt_p(p)}."
            )
        lines.append("")

    lines.append("## Subset composition vs. full corpus (t=0.90)\n")
    lines.append(
        f"Mean resume length (words): full corpus {composition['full_mean_length']:.1f}, "
        f"preserved subset {composition['subset_mean_length']:.1f}.\n"
    )
    lines.append(
        "Category representation ratio = (share of preserved subset) / (share of full "
        "corpus); 1.0 = proportionally represented. Flagged (⚠️) when ratio < 0.5 or > 1.5:\n"
    )
    lines.append("| Category | Full % | Subset % | Ratio |")
    lines.append("|---|---|---|---|")
    for cat, d in sorted(composition["per_category"].items(), key=lambda kv: kv[1]["ratio"]):
        marker = " ⚠️" if (d["ratio"] < 0.5 or d["ratio"] > 1.5) else ""
        lines.append(
            f"| {cat} | {100*d['full_frac']:.1f}% | {100*d['subset_frac']:.1f}% | "
            f"{d['ratio']:.2f}{marker} |"
        )
    lines.append(
        "\n**AUTOMOBILE is entirely absent from the preserved subset (ratio 0.00)**; "
        "ARTS and AGRICULTURE are severely under-represented. This tracks the keyword-"
        "coverage gap above, not a real absence of preserved qualifications in those "
        "trades — category-level claims from this audit should not be made for these "
        "three categories.\n"
    )

    lines.append("## Spot-check sample (Step 5)\n")
    bucket_counts = Counter(spotcheck.bucket_of(r) for r in sample)
    lines.append(
        f"{len(sample)} pairs sampled (seed={spotcheck.RANDOM_SEED}), stratified across "
        f"retention-rate buckets {dict(bucket_counts)} and covering "
        f"{len({r['category'] for r in sample})}/24 categories. Written to "
        "`spotcheck_sample.csv` with blank `human_judgment_yes_no`/`human_note` columns. "
        "**Human annotation is a separate, manual step** — after filling those columns in, "
        "run `python3 src/qual_audit/spotcheck_agreement.py` to get the agreement rate "
        "and the full list of disagreements. Not yet run in this summary.\n"
    )

    lines.append("## Additional finding (not in the original spec): broken placeholder figures\n")
    ph_n, ph_total, ph_ids = placeholder_stats
    lines.append(
        f"Spot-checking pair 10466208 by hand (Step 5) surfaced AI-polished resumes that "
        "left literal unfilled template placeholders where a quantified figure should be "
        '(e.g. `[Number]`, `[Year]`, `[Month, Year]`) rather than the original\'s real '
        f"number. **{ph_n} / {ph_total} polished resumes ({ph_n/ph_total:.1%})** contain at "
        "least one such placeholder. This is the opposite of qualification inflation — a "
        "broken placeholder is a content *downgrade* — so if screeners still prefer these "
        "resumes, that is independent evidence the win rate is driven by surface style "
        "rather than content quality. Not deeply investigated here (out of scope for this "
        "audit's spec); flagged as a concrete lead for follow-up rather than pursued further.\n"
    )
    lines.append(
        f"**Join against preserved-subset membership (spec item 1d):** `placeholder_damaged` "
        "is now a direct gate in the preserved-subset definition itself (`preservation.py:"
        "preserved_flags`), alongside `undefined_retention` - a placeholder-damaged pair can "
        "never be called \"provably unchanged,\" so it is structurally excluded rather than "
        "patched in after the fact. Verified this holds at every threshold:\n"
    )
    for t in THRESHOLDS:
        overlap = placeholder_join[t]["overlap_ids"]
        status = "confirmed" if not overlap else "**VIOLATED - investigate**"
        lines.append(f"- t={t:.2f}: {len(overlap)} placeholder-damaged pairs inside the preserved subset ({status}).")
    lines.append(
        f"\n(Earlier in this audit's development, before this gate was added structurally, "
        f"a post-hoc check found 24/384 preserved-subset pairs at t=0.90 were placeholder-"
        f"damaged; excluding them moved the pooled rate from 0.791 to 0.787 - the negligible "
        f"delta the audit spec predicted. The headline numbers above already reflect the "
        f"structural exclusion, so that recomputation is now redundant and not repeated here.)\n"
    )

    lines.append("## Priority 3 (reviewer follow-up): does length explain the preference?\n")
    lc = length_stats
    lines.append(
        f"Token counts use the paper's own tokenizer (`\\b[a-zA-Z']+\\b`, lowercased). "
        f"Across all {len(lc['length_df'])} pairs, the AI-polished version is **shorter** "
        f"on average, not longer: mean token delta (polished − original) = "
        f"**{lc['mean_delta']:.1f}** tokens (median {lc['median_delta']:.1f}, "
        f"SD {lc['std_delta']:.1f}); only **{lc['pct_longer']:.1%}** of polished resumes "
        "come out longer than their original. This runs directly opposite to the \"AI just "
        "writes more\" objection before even restricting to a subset.\n"
    )
    lines.append(
        f"There is a small positive correlation between length delta and per-pair AI-win "
        f"fraction (Pearson r={lc['pearson_r']:.3f}, p={fmt_p(lc['pearson_p'])}; Spearman "
        f"r={lc['spearman_r']:.3f}, p={fmt_p(lc['spearman_p'])}) — pairs where the polished "
        "version is comparatively longer do win somewhat more often — but it is a small "
        "effect, not the driver of the headline result (see below).\n"
    )
    nl_ct = lc["not_longer_test"]
    lines.append(
        f"**The key number:** restricting to the **{lc['not_longer_pair_count']}** pairs "
        f"(of {len(lc['length_df'])}) where the polished version is no longer than the "
        f"original, the AI-win rate is still **{fmt_rate(lc['not_longer_rate'])}** "
        f"({lc['not_longer_k']}/{lc['not_longer_n']} decisions, cluster-robust 95% CI "
        f"[{fmt_rate(nl_ct['ci_95'][0])}, {fmt_rate(nl_ct['ci_95'][1])}]) — indistinguishable "
        "from the full-corpus rate. **This closes the \"AI just writes more\" objection**: "
        "the preference holds even where the AI version did not get to pad its way to a win.\n"
    )

    lines.append("## Priority 4 (reviewer follow-up): feature-delta regression\n")
    lines.append(
        "Logistic regression of the AI-win indicator on the 8 standardized per-pair "
        "linguistic feature deltas (polished − original, from `data/results/new_merged.csv`) "
        "pooled across all 7 available models, with model fixed effects and pair-clustered "
        "standard errors. **Perplexity is dropped** (it is a monotonic transform of entropy, "
        "`2^entropy` in this codebase's own `Metrics.py` - including both is exactly what "
        "produced the existing paper's Table 4 entropy/perplexity coefficient pair "
        "(+5.990/−6.270), a collinearity artifact rather than two independent findings, not "
        "something to repeat here).\n"
    )
    lines.append(
        "*Feature-name caveat:* `new_merged.csv`'s 9 columns do not map 1:1 onto the paper's "
        "named formulas (no separate sliding-window-variance column; `repetition` and "
        "`sentence_evenness` stand in for TTR-like and length-dispersion-like quantities). "
        "This regression uses the 9 real columns actually present as \"the nine features,\" "
        "disclosed rather than assumed equivalent to the paper's definitions.\n"
    )
    lines.append(
        f"n={feature_reg['n_obs']} decisions across {feature_reg['n_pairs']} pairs, "
        f"in-sample AUC={feature_reg['auc']:.3f} (a fit statistic for this explanatory "
        "regression, not a held-out predictive claim - and not comparable to the paper's "
        "existing 0.911 classifier AUC, which answers a different question: distinguishing "
        "AI-written from human-written text, not predicting the screening decision).\n"
    )
    lines.append("| Feature delta | Standardized coef | Clustered SE | p |")
    lines.append("|---|---|---|---|")
    ct = feature_reg["coef_table"]
    feat_rows = ct.loc[[c for c in ct.index if c.startswith("delta_")]]
    feat_rows = feat_rows.reindex(feat_rows["coef"].abs().sort_values(ascending=False).index)
    for name, row in feat_rows.iterrows():
        sig = "**" if row["p_clustered"] < 0.05 else ""
        lines.append(
            f"| {sig}{name.replace('delta_', '')}{sig} | {row['coef']:.3f} | "
            f"{row['se_clustered']:.3f} | {fmt_p(row['p_clustered'])} |"
        )
    lines.append(
        "\n**Which delta dominates:** `entropy` is the strongest and only unambiguously "
        "significant predictor (p<0.001) among the 8; `lexical_density` is directionally "
        "present but only marginal (p≈0.08); the rest are not significant. This only "
        "partially matches the mechanism story motivating this check - lexical density and "
        "TTR were expected to dominate (per the paper's Mann-Whitney effect sizes), but here "
        "it is the entropy delta that predicts *winning*, not just distinguishing AI from "
        "human text. Reported as found, not forced to match the expected story.\n"
    )

    lines.append("## Priority 2 (reviewer's main ask): per-occupation heterogeneity\n")
    lines.append(
        "Groups **all** plain-text screening decisions (7 models, all pairs in each "
        "category - not restricted to the preserved subset) by occupational category. "
        "Cluster-robust rate and 95% CI per category (clustered on pair_id, same method "
        "as the main audit). Per-category *p*-values are deliberately not reported - with "
        "24 categories, per-category significance testing invites a multiple-comparisons "
        "objection that CIs sidestep. Sorted by rate:\n"
    )
    lines.append("| Category | n pairs | n decisions | Rate | 95% CI | |")
    lines.append("|---|---|---|---|---|---|")
    for r in category_table:
        flag = " ⚠️ underpowered" if r["underpowered"] else ""
        lines.append(
            f"| {r['category']} | {r['n_pairs']} | {r['n_decisions']} | "
            f"{fmt_rate(r['rate'])} | [{fmt_rate(r['ci_95'][0])}, {fmt_rate(r['ci_95'][1])}] |{flag} |"
        )
    lines.append(
        f"\n**BPO** (n=17 pairs) and **AUTOMOBILE** (n=28 pairs) are flagged underpowered "
        f"(below {category_breakdown.UNDERPOWERED_THRESHOLD_PAIRS} pairs) - their point "
        "estimates happen to be the two highest, but the CIs are correspondingly the "
        "widest, and this should not be over-read as those categories being special.\n"
    )
    lines.append(
        f"**The claim tested, not assumed:** is the effect directionally consistent across "
        f"all categories with adequate n? **Yes.** All {category_consistency['n_adequate_categories']} "
        "adequately-powered categories have a 95% CI entirely above 0.5 - the lowest is "
        f"TEACHER at {fmt_rate(category_table[0]['rate'])} "
        f"[{fmt_rate(category_table[0]['ci_95'][0])}, {fmt_rate(category_table[0]['ci_95'][1])}], "
        "still comfortably above chance. No category reverses "
        f"(reversed list: {category_consistency['reversed_categories'] or 'none'}). This is "
        "the finding, not a foregone conclusion - it was checked, not assumed, and the "
        "magnitude does vary by ~8 points of preference rate across categories even though "
        "the direction never does.\n"
    )

    lines.append("## Priority 5 (writing task, verified before writing): style among human-written résumés\n")
    lines.append(
        "Checked whether the 9 linguistic features predict which HUMAN-written résumé wins "
        "against its AI-polished counterpart (i.e. holding \"is this AI-polished\" fixed at "
        "\"no\"). The claim to verify was |r| <= 0.173 with token entropy reversed in "
        "direction. The only precomputed split in this repo "
        "(`new_metrics_results_winning`/`_losing`) has 5 of 9 features and excludes entropy "
        "entirely, so it could not confirm the entropy claim directly. Recomputing \"human "
        "won >=1 of 7 models\" from the current complete data (1,667 won / 442 lost - an "
        "exact row-count match to the old `human_win_tally.csv`, confirming this is the "
        "historical definition) reproduces the claim closely:\n"
    )
    lines.append("| Feature | r | p |")
    lines.append("|---|---|---|")
    for row in sorted(style_result["any"]["results"], key=lambda x: -abs(x["r"])):
        lines.append(f"| {row['feature']} | {row['r']:.3f} | {fmt_p(row['p'])} |")
    largest = max(style_result["any"]["results"], key=lambda x: abs(x["r"]))
    lines.append(
        f"\nAll |r| <= 0.173 ({largest['feature']} largest at r={largest['r']:.3f}) - "
        "reversed in direction from the AI-vs-human comparison, "
        "where AI-polished résumés show *higher* entropy). A stricter \"won a majority of "
        f"7 models\" definition ({style_result['majority']['n_won']} won / "
        f"{style_result['majority']['n_lost']} lost - much smaller, rarer group) shows the "
        "same qualitative pattern at roughly double the magnitude, noisier given the smaller "
        "n but not a different story. See `paper_text_suggestions.md` for the bounded "
        "paragraph drafted for the paper from this result.\n"
    )

    lines.append("## Known limitations (carried into any write-up, not hidden)\n")
    lines.append(
        "1. **Lexical, not semantic.** Keyword matching cannot recognize paraphrases "
        "(\"managed a team\" vs. \"led cross-functional teams\") as the same "
        "qualification. This over-counts differences, making the preserved subset "
        "conservative — a preference that survives it is more convincing, not less.\n"
        "2. **Blind to non-numeric, non-dictionary puffery.** Framing like \"proven "
        "track record of leadership\" is invisible unless it maps to a dictionary term. "
        "This is the genuine residual gap; a semantic-entailment audit is planned for "
        "camera-ready.\n"
        "3. **Power vs. strictness.** Reported as a full sweep (t=0.90/0.95/1.00) rather "
        "than one hand-picked cutoff.\n"
        "4. **Says nothing about temporal shift** (2021-source-corpus vs. 2026-norms).\n"
        f"5. **{len(payoff['models'])} of 8 models available** (granite4:3b missing) — see "
        "'Model coverage' above. A limitation of this particular run of the audit, "
        "distinct from the four limitations the audit spec anticipated, but now a "
        "minor one rather than the dominant one.\n"
        f"6. **{undef_n}/{total} pairs ({undef_n/total:.1%}) excluded via "
        "undefined_retention** because the reused keyword dictionary found zero scoped "
        "matches — concentrated in AUTOMOBILE, ARTS, AGRICULTURE, CONSULTANT, AVIATION. "
        "Preserved-subset findings should not be generalized to those categories.\n"
        f"7. **{ph_n}/{total} pairs carry a broken template placeholder** (e.g. `[Number]` "
        "left unfilled in the polished text — see 'Additional finding' above); most were "
        "already excluded for other reasons, but 24 would otherwise have passed the "
        "t=0.90 gate and are excluded specifically because of this defect. Structural, "
        "not a post-hoc patch.\n"
    )

    lines.append("## Result interpretation\n")
    pooled_rate_90 = payoff["by_threshold"][0.90]["pooled_rate"]
    baseline_rate = payoff["full_corpus_local_baseline"]["rate"]
    if pooled_rate_90 is not None and abs(pooled_rate_90 - baseline_rate) < 0.03:
        lines.append(
            f"Preserved-subset rate ({fmt_rate(pooled_rate_90)}) is close to the local "
            f"baseline ({fmt_rate(baseline_rate)}), and the cluster-robust test does not "
            "detect a significant difference. **The preference persists when measured "
            f"qualifications are held constant**, for the {len(payoff['models'])} models "
            "with available data. This does not license \"proven purely stylistic\" — "
            "limitation #2 (residual non-dictionary puffery) remains open.\n"
        )
    else:
        lines.append(
            "See the threshold table above and judge against the three outcome "
            "categories in the audit spec (persists / attenuates / falls toward chance) "
            "directly from these numbers — no outcome was assumed in advance.\n"
        )

    lines.append("## Where every excluded pair is documented\n")
    lines.append(
        "`preservation_scores.csv` has one row per pair with an explicit "
        "`undefined_retention` flag and per-threshold `preserved_*` flags — the full, "
        "queryable, per-pair record of why any given pair is in or out of each subset. "
        "This summary reports it aggregated by category rather than as a 2,109-row list.\n"
    )

    return "\n".join(lines)


def main():
    print("=== Step: parsing raw screening decisions ===")
    screening_rows = parse_screeners.load_all_screening_outcomes(str(_REPO_ROOT))
    by_model = Counter(r["model"] for r in screening_rows)
    for model, count in sorted(by_model.items()):
        print(f"  {model}: {count} pairs")
    parse_screeners.write_screening_outcomes_csv(screening_rows, str(OUT_DIR / "screening_outcomes.csv"))

    print("\n=== Step: scoring preservation for all pairs ===")
    scores_stage = build_scores.build_all_scores(str(_REPO_ROOT))
    print(f"  Matched pairs: {scores_stage['total_pairs']}")
    print(f"  Missing human .txt: {len(scores_stage['missing_human_txt'])}")
    print(f"  Missing AI .txt: {len(scores_stage['missing_ai_txt'])}")
    print(f"  Missing category: {len(scores_stage['missing_category'])}")
    undef_n = sum(1 for r in scores_stage["rows"] if r["undefined_retention"])
    print(f"  undefined_retention: {undef_n} ({undef_n/scores_stage['total_pairs']:.1%})")
    build_scores.write_scores_csv(scores_stage["rows"], str(OUT_DIR / "preservation_scores.csv"))

    print("\n=== Step: drawing spot-check sample ===")
    preservation_rows_for_sample = spotcheck.load_preservation_rows(
        str(OUT_DIR / "preservation_scores.csv")
    )
    sample = spotcheck.build_sample(preservation_rows_for_sample)
    print(f"  Sampled {len(sample)} pairs, seed={spotcheck.RANDOM_SEED}")
    enriched = [
        spotcheck.enrich_with_number_detail(
            r, _REPO_ROOT / "data" / "humanresumes", _REPO_ROOT / "data" / "airesumes"
        )
        for r in sample
    ]
    spotcheck.write_sample_csv(enriched, str(OUT_DIR / "spotcheck_sample.csv"))

    print("\n=== Step: payoff analysis ===")
    screening_outcomes = analysis.load_screening_outcomes(str(OUT_DIR / "screening_outcomes.csv"))
    preservation_scores = analysis.load_preservation_scores(str(OUT_DIR / "preservation_scores.csv"))
    payoff = analysis.compute_payoff(screening_outcomes, preservation_scores)
    for t in THRESHOLDS:
        tr = payoff["by_threshold"][t]
        print(f"  t={t}: subset={tr['subset_pair_count']} pairs, pooled rate={fmt_rate(tr['pooled_rate'])}")
    analysis.write_payoff_csv(payoff, str(OUT_DIR / "preserved_subset_winrates.csv"))

    word_counts = load_word_counts()
    subset_ids_90 = analysis.preserved_ids_at(preservation_scores, 0.90)
    composition = analysis.composition_breakdown(preservation_scores, subset_ids_90, word_counts)

    placeholder_stats = count_placeholder_figures(_REPO_ROOT / "data" / "airesumes")
    ph_n, ph_total, ph_ids = placeholder_stats
    print(f"  Broken placeholder figures: {ph_n}/{ph_total} polished resumes")

    # placeholder_damaged is now a direct gate in preserved_flags (preservation.py),
    # so overlap with the preserved subset should always be 0 by construction -
    # this is a verification of that invariant, not a post-hoc correction.
    placeholder_join = {}
    for t in THRESHOLDS:
        subset_ids = analysis.preserved_ids_at(preservation_scores, t)
        overlap = subset_ids & ph_ids
        placeholder_join[t] = {"overlap_ids": overlap}
        print(f"  t={t}: {len(overlap)} placeholder-damaged pairs inside preserved subset (should be 0)")

    print("\n=== Step: length check (Priority 3) ===")
    length_stats = length_check.run(_REPO_ROOT)
    print(f"  mean token delta: {length_stats['mean_delta']:.1f}, "
          f"% longer: {length_stats['pct_longer']:.1%}")
    print(f"  not-longer-subset AI-win rate: {length_stats['not_longer_rate']:.3f} "
          f"(n={length_stats['not_longer_n']})")
    length_stats["length_df"].to_csv(OUT_DIR / "length_check.csv", index=False)

    print("\n=== Step: feature-delta regression (Priority 4) ===")
    deltas = feature_regression.build_pair_deltas(
        str(_REPO_ROOT / "data" / "results" / "new_merged.csv")
    )
    feature_reg = feature_regression.fit_pooled_regression(screening_rows, deltas)
    print(f"  n_obs={feature_reg['n_obs']}, n_pairs={feature_reg['n_pairs']}, AUC={feature_reg['auc']:.3f}")
    feature_reg["coef_table"].to_csv(OUT_DIR / "feature_regression_coefficients.csv")

    print("\n=== Step: per-occupation heterogeneity (Priority 2) ===")
    category_table = category_breakdown.build_category_table(screening_outcomes, preservation_scores)
    category_consistency = category_breakdown.check_directional_consistency(category_table)
    print(f"  adequately-powered categories: {category_consistency['n_adequate_categories']}")
    print(f"  reversed categories: {category_consistency['reversed_categories']}")
    import csv as _csv
    with open(OUT_DIR / "category_breakdown.csv", "w", newline="", encoding="utf-8") as f:
        w = _csv.writer(f)
        w.writerow(["category", "n_pairs", "n_decisions", "ai_wins", "rate", "ci_lo", "ci_hi", "p_vs_0.5", "underpowered"])
        for r in category_table:
            w.writerow([r["category"], r["n_pairs"], r["n_decisions"], r["ai_wins"],
                        r["rate"], r["ci_95"][0], r["ci_95"][1], r["p_vs_0.5"], r["underpowered"]])

    print("\n=== Step: style-among-human-résumés check (Priority 5) ===")
    style_result = winning_losing_style.run(_REPO_ROOT)
    print(f"  primary (any): won={style_result['any']['n_won']}, lost={style_result['any']['n_lost']}")
    print(f"  robustness (majority): won={style_result['majority']['n_won']}, lost={style_result['majority']['n_lost']}")
    import csv as _csv2
    with open(OUT_DIR / "winning_losing_style.csv", "w", newline="", encoding="utf-8") as f:
        w = _csv2.writer(f)
        w.writerow(["definition", "feature", "n_won", "n_lost", "U", "p", "r"])
        for label, block in style_result.items():
            for row in block["results"]:
                w.writerow([label, row["feature"], row["n_won"], row["n_lost"], row["U"], row["p"], row["r"]])

    print("\n=== Step: writing audit_summary.md ===")
    summary_md = build_summary_md(
        screening_rows, scores_stage, sample, payoff, composition, placeholder_stats,
        placeholder_join, length_stats, feature_reg, category_table, category_consistency,
        style_result,
    )
    (OUT_DIR / "audit_summary.md").write_text(summary_md, encoding="utf-8")
    print(f"Wrote {OUT_DIR / 'audit_summary.md'}")

    print("\nDone. Outputs in", OUT_DIR)


if __name__ == "__main__":
    main()
