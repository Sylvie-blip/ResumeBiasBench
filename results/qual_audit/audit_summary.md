# Qualification-Preservation Audit — Summary

## Model coverage (read this first)

7 of the paper's 8 screening models now have a complete (2,109/2,109-pair) raw-decision file at the repo root. Only **granite4:3b** is unavailable - its Table 1 rate (0.785) is shown for context only and never enters any computed rate below.

Earlier runs of this audit only had 3 partial/ambiguous files (2 stray copies under `Useless/` plus an unlabeled `qwen2.5.txt`); those are now superseded and excluded (see `parse_screeners.py` docstring for why they can't simply be merged with the complete runs - Ollama's local inference turned out not to be perfectly repeatable run-to-run even at temperature=0, so a partial and a complete run of "the same" model disagreed on overlapping pairs).

## Corpus join

- Matched pairs: 2109 / 2,109 expected

- Missing human .txt for an AI file: 0
- Missing AI .txt for a human file: 0
- Missing category label: 0

## Keyword-dictionary coverage (a major, load-bearing limitation)

**923 / 2109 pairs (43.8%) have zero scoped keyword matches in the original resume text** (`undefined_retention`) and are excluded from every preserved-subset calculation below. This is a property of the *reused* `SKILLS_BY_DOMAIN` dictionary in `src/conv_to_json.py` (deliberately not modified — see spec) being narrow/jargon-heavy relative to how real candidates phrase resumes, not a bug in this audit's join or scoring logic. Coverage is highly uneven by category:

| Category | Undefined | Total | % undefined |
|---|---|---|---|
| AUTOMOBILE | 27 | 28 | 96.4% ⚠️ |
| ARTS | 84 | 89 | 94.4% ⚠️ |
| AGRICULTURE | 48 | 52 | 92.3% ⚠️ |
| CONSULTANT | 81 | 95 | 85.3% ⚠️ |
| AVIATION | 70 | 97 | 72.2% ⚠️ |
| DESIGNER | 67 | 98 | 68.4% ⚠️ |
| DIGITAL-MEDIA | 54 | 79 | 68.4% ⚠️ |
| ADVOCATE | 65 | 104 | 62.5% ⚠️ |
| HEALTHCARE | 60 | 100 | 60.0% ⚠️ |
| TEACHER | 45 | 89 | 50.6% |
| SALES | 50 | 104 | 48.1% |
| BPO | 8 | 17 | 47.1% |
| BANKING | 47 | 100 | 47.0% |
| FINANCE | 50 | 109 | 45.9% |
| FITNESS | 39 | 96 | 40.6% |
| PUBLIC-RELATIONS | 32 | 94 | 34.0% |
| CONSTRUCTION | 30 | 93 | 32.3% |
| INFORMATION-TECH | 21 | 103 | 20.4% |
| CHEF | 19 | 96 | 19.8% |
| ENGINEERING | 15 | 100 | 15.0% |
| APPAREL | 8 | 76 | 10.5% |
| HR | 2 | 97 | 2.1% |
| BUSINESS-DEV | 1 | 94 | 1.1% |
| ACCOUNTANT | 0 | 99 | 0.0% |

## Category-to-domain mapping

21 of 24 `sorted_data` categories map 1:1 to a `SKILLS_BY_DOMAIN` key. 3 are imperfect and explicitly disclosed rather than guessed: APPAREL, BANKING, PUBLIC-RELATIONS (BANKING→finance; APPAREL→design∪general; PUBLIC-RELATIONS→digital_media∪business).

## Headline: preserved-subset win rate by threshold

Full-corpus local baseline (these 7 models, all their available decisions, unfiltered): **0.790** (k=11669, n=14763). Paper's full 8-model/2,109-pair rate, for context only: 0.79.

**Inference is cluster-robust, clustered on pair_id** (OLS of the AI-win indicator on a preserved/non-preserved indicator, SEs clustered on pair, t-distribution with n_clusters-1 df) - not the naive independent-decisions CI/binomial-test from an earlier version of this audit. Each pair is judged by up to 7 models, so within-pair outcomes are correlated and the naive approach understates the true interval. Empirically the correction here is modest: at t=0.90 the design effect is ~1.16 (implied intra-pair correlation ~0.026) - real, in the expected direction, but small because agreement between these 7 heterogeneous model families on any single pair turns out to be weak. The comparison group is also now the true complement (non-preserved pairs), not a superset that partially contains the subset being tested.

| t | Subset pairs | Pooled k/n | Pooled rate | Cluster-robust 95% CI | vs. 0.5 (clustered p) | vs. complement (clustered p) |
|---|---|---|---|---|---|---|
| 0.90 | 360 | 1983/2520 | 0.787 | [0.770, 0.804] | 5.36e-189 | 0.660 |
| 0.95 | 359 | 1978/2513 | 0.787 | [0.770, 0.804] | 1.93e-188 | 0.679 |
| 1.00 | 359 | 1978/2513 | 0.787 | [0.770, 0.804] | 1.93e-188 | 0.679 |

## Per-model breakdown (t=0.90, the most inclusive threshold)

| Model | Subset k/n | Subset rate | Full-corpus rate | Paper Table 1 rate | vs. complement (clustered p) |
|---|---|---|---|---|---|
| cogito:3b | 310/360 | 0.861 | 0.880 | 0.88 | 0.249 |
| gemma3:12b | 292/360 | 0.811 | 0.790 | 0.79 | 0.264 |
| gemma3:4b | 265/360 | 0.736 | 0.716 | 0.716 | 0.355 |
| mistral-small3.2:latest | 293/360 | 0.814 | 0.793 | 0.793 | 0.263 |
| qwen2.5:14b | 323/360 | 0.897 | 0.889 | 0.888 | 0.556 |
| qwen2.5:32b | 307/360 | 0.853 | 0.863 | 0.863 | 0.528 |
| qwen2.5:3b | 193/360 | 0.536 | 0.602 | 0.602 | 0.006 |

**Not every model tells the same story.** The pooled/headline result above masks per-model variation - at least one model shows a statistically significant difference between its preserved-subset rate and its own baseline, which the pooled number alone would hide:

- **qwen2.5:3b**: preserved-subset rate (0.536) is significantly lower than its own baseline (0.602), two-prop p=0.006.

## Subset composition vs. full corpus (t=0.90)

Mean resume length (words): full corpus 810.9, preserved subset 756.0.

Category representation ratio = (share of preserved subset) / (share of full corpus); 1.0 = proportionally represented. Flagged (⚠️) when ratio < 0.5 or > 1.5:

| Category | Full % | Subset % | Ratio |
|---|---|---|---|
| AUTOMOBILE | 1.3% | 0.0% | 0.00 ⚠️ |
| ARTS | 4.2% | 0.6% | 0.13 ⚠️ |
| AGRICULTURE | 2.5% | 0.8% | 0.34 ⚠️ |
| CONSULTANT | 4.5% | 2.2% | 0.49 ⚠️ |
| ACCOUNTANT | 4.7% | 2.8% | 0.59 |
| AVIATION | 4.6% | 2.8% | 0.60 |
| TEACHER | 4.2% | 2.8% | 0.66 |
| HR | 4.6% | 3.1% | 0.66 |
| HEALTHCARE | 4.7% | 3.9% | 0.82 |
| DIGITAL-MEDIA | 3.7% | 3.3% | 0.89 |
| BANKING | 4.7% | 4.4% | 0.94 |
| SALES | 4.9% | 4.7% | 0.96 |
| FINANCE | 5.2% | 5.0% | 0.97 |
| CHEF | 4.6% | 4.4% | 0.98 |
| DESIGNER | 4.6% | 5.0% | 1.08 |
| ADVOCATE | 4.9% | 5.6% | 1.13 |
| CONSTRUCTION | 4.4% | 5.0% | 1.13 |
| PUBLIC-RELATIONS | 4.5% | 5.3% | 1.18 |
| INFORMATION-TECH | 4.9% | 6.4% | 1.31 |
| BPO | 0.8% | 1.1% | 1.38 |
| APPAREL | 3.6% | 5.3% | 1.46 |
| FITNESS | 4.6% | 6.7% | 1.46 |
| ENGINEERING | 4.7% | 8.9% | 1.87 ⚠️ |
| BUSINESS-DEV | 4.5% | 10.0% | 2.24 ⚠️ |

**AUTOMOBILE is entirely absent from the preserved subset (ratio 0.00)**; ARTS and AGRICULTURE are severely under-represented. This tracks the keyword-coverage gap above, not a real absence of preserved qualifications in those trades — category-level claims from this audit should not be made for these three categories.

## Spot-check sample (Step 5)

50 pairs sampled (seed=20260717), stratified across retention-rate buckets {'undefined': 23, 'lt_0.50': 5, '0.50_to_0.90': 6, '0.90_to_0.95': 5, 'exactly_1.00': 11} and covering 24/24 categories. Written to `spotcheck_sample.csv` with blank `human_judgment_yes_no`/`human_note` columns. **Human annotation is a separate, manual step** — after filling those columns in, run `python3 src/qual_audit/spotcheck_agreement.py` to get the agreement rate and the full list of disagreements. Not yet run in this summary.

## Additional finding (not in the original spec): broken placeholder figures

Spot-checking pair 10466208 by hand (Step 5) surfaced AI-polished resumes that left literal unfilled template placeholders where a quantified figure should be (e.g. `[Number]`, `[Year]`, `[Month, Year]`) rather than the original's real number. **94 / 2109 polished resumes (4.5%)** contain at least one such placeholder. This is the opposite of qualification inflation — a broken placeholder is a content *downgrade* — so if screeners still prefer these resumes, that is independent evidence the win rate is driven by surface style rather than content quality. Not deeply investigated here (out of scope for this audit's spec); flagged as a concrete lead for follow-up rather than pursued further.

**Join against preserved-subset membership (spec item 1d):** `placeholder_damaged` is now a direct gate in the preserved-subset definition itself (`preservation.py:preserved_flags`), alongside `undefined_retention` - a placeholder-damaged pair can never be called "provably unchanged," so it is structurally excluded rather than patched in after the fact. Verified this holds at every threshold:

- t=0.90: 0 placeholder-damaged pairs inside the preserved subset (confirmed).
- t=0.95: 0 placeholder-damaged pairs inside the preserved subset (confirmed).
- t=1.00: 0 placeholder-damaged pairs inside the preserved subset (confirmed).

(Earlier in this audit's development, before this gate was added structurally, a post-hoc check found 24/384 preserved-subset pairs at t=0.90 were placeholder-damaged; excluding them moved the pooled rate from 0.791 to 0.787 - the negligible delta the audit spec predicted. The headline numbers above already reflect the structural exclusion, so that recomputation is now redundant and not repeated here.)

## Priority 3 (reviewer follow-up): does length explain the preference?

Token counts use the paper's own tokenizer (`\b[a-zA-Z']+\b`, lowercased). Across all 2109 pairs, the AI-polished version is **shorter** on average, not longer: mean token delta (polished − original) = **-239.2** tokens (median -165.0, SD 299.3); only **14.0%** of polished resumes come out longer than their original. This runs directly opposite to the "AI just writes more" objection before even restricting to a subset.

There is a small positive correlation between length delta and per-pair AI-win fraction (Pearson r=0.156, p=5.21e-13; Spearman r=0.161, p=9.07e-14) — pairs where the polished version is comparatively longer do win somewhat more often — but it is a small effect, not the driver of the headline result (see below).

**The key number:** restricting to the **1813** pairs (of 2109) where the polished version is no longer than the original, the AI-win rate is still **0.784** (9949/12691 decisions, cluster-robust 95% CI [0.776, 0.792]) — indistinguishable from the full-corpus rate. **This closes the "AI just writes more" objection**: the preference holds even where the AI version did not get to pad its way to a win.

## Priority 4 (reviewer follow-up): feature-delta regression

Logistic regression of the AI-win indicator on the 8 standardized per-pair linguistic feature deltas (polished − original, from `data/results/new_merged.csv`) pooled across all 7 available models, with model fixed effects and pair-clustered standard errors. **Perplexity is dropped** (it is a monotonic transform of entropy, `2^entropy` in this codebase's own `Metrics.py` - including both is exactly what produced the existing paper's Table 4 entropy/perplexity coefficient pair (+5.990/−6.270), a collinearity artifact rather than two independent findings, not something to repeat here).

*Feature-name caveat:* `new_merged.csv`'s 9 columns do not map 1:1 onto the paper's named formulas (no separate sliding-window-variance column; `repetition` and `sentence_evenness` stand in for TTR-like and length-dispersion-like quantities). This regression uses the 9 real columns actually present as "the nine features," disclosed rather than assumed equivalent to the paper's definitions.

n=14763 decisions across 2109 pairs, in-sample AUC=0.670 (a fit statistic for this explanatory regression, not a held-out predictive claim - and not comparable to the paper's existing 0.911 classifier AUC, which answers a different question: distinguishing AI-written from human-written text, not predicting the screening decision).

| Feature delta | Standardized coef | Clustered SE | p |
|---|---|---|---|
| lexical_density | -0.178 | 0.103 | 0.084 |
| **entropy** | 0.150 | 0.028 | 1.07e-07 |
| repetition | 0.085 | 0.112 | 0.449 |
| tonal_stability | -0.018 | 0.022 | 0.427 |
| pattern_regularity | -0.015 | 0.040 | 0.701 |
| rolling_entropy | 0.011 | 0.023 | 0.622 |
| sentence_evenness | 0.004 | 0.025 | 0.860 |
| sentence_entropy_variance | 0.003 | 0.027 | 0.919 |

**Which delta dominates:** `entropy` is the strongest and only unambiguously significant predictor (p<0.001) among the 8; `lexical_density` is directionally present but only marginal (p≈0.08); the rest are not significant. This only partially matches the mechanism story motivating this check - lexical density and TTR were expected to dominate (per the paper's Mann-Whitney effect sizes), but here it is the entropy delta that predicts *winning*, not just distinguishing AI from human text. Reported as found, not forced to match the expected story.

## Priority 2 (reviewer's main ask): per-occupation heterogeneity

Groups **all** plain-text screening decisions (7 models, all pairs in each category - not restricted to the preserved subset) by occupational category. Cluster-robust rate and 95% CI per category (clustered on pair_id, same method as the main audit). Per-category *p*-values are deliberately not reported - with 24 categories, per-category significance testing invites a multiple-comparisons objection that CIs sidestep. Sorted by rate:

| Category | n pairs | n decisions | Rate | 95% CI | |
|---|---|---|---|---|---|
| TEACHER | 89 | 623 | 0.737 | [0.699, 0.774] | |
| ADVOCATE | 104 | 728 | 0.757 | [0.716, 0.797] | |
| HEALTHCARE | 100 | 700 | 0.760 | [0.726, 0.794] | |
| CHEF | 96 | 672 | 0.775 | [0.740, 0.811] | |
| HR | 97 | 679 | 0.779 | [0.745, 0.813] | |
| ARTS | 89 | 623 | 0.782 | [0.747, 0.817] | |
| AVIATION | 97 | 679 | 0.782 | [0.747, 0.817] | |
| FITNESS | 96 | 672 | 0.784 | [0.750, 0.818] | |
| AGRICULTURE | 52 | 364 | 0.786 | [0.744, 0.827] | |
| CONSTRUCTION | 93 | 651 | 0.786 | [0.751, 0.822] | |
| PUBLIC-RELATIONS | 94 | 658 | 0.787 | [0.755, 0.820] | |
| ACCOUNTANT | 99 | 693 | 0.794 | [0.762, 0.825] | |
| ENGINEERING | 100 | 700 | 0.796 | [0.761, 0.830] | |
| BANKING | 100 | 700 | 0.797 | [0.765, 0.830] | |
| CONSULTANT | 95 | 665 | 0.798 | [0.770, 0.827] | |
| APPAREL | 76 | 532 | 0.803 | [0.768, 0.838] | |
| DIGITAL-MEDIA | 79 | 553 | 0.803 | [0.768, 0.838] | |
| INFORMATION-TECH | 103 | 721 | 0.809 | [0.774, 0.843] | |
| DESIGNER | 98 | 686 | 0.809 | [0.778, 0.840] | |
| BUSINESS-DEV | 94 | 658 | 0.810 | [0.781, 0.839] | |
| SALES | 104 | 728 | 0.816 | [0.786, 0.846] | |
| FINANCE | 109 | 763 | 0.817 | [0.789, 0.844] | |
| BPO | 17 | 119 | 0.824 | [0.752, 0.895] | ⚠️ underpowered |
| AUTOMOBILE | 28 | 196 | 0.832 | [0.779, 0.884] | ⚠️ underpowered |

**BPO** (n=17 pairs) and **AUTOMOBILE** (n=28 pairs) are flagged underpowered (below 30 pairs) - their point estimates happen to be the two highest, but the CIs are correspondingly the widest, and this should not be over-read as those categories being special.

**The claim tested, not assumed:** is the effect directionally consistent across all categories with adequate n? **Yes.** All 22 adequately-powered categories have a 95% CI entirely above 0.5 - the lowest is TEACHER at 0.737 [0.699, 0.774], still comfortably above chance. No category reverses (reversed list: none). This is the finding, not a foregone conclusion - it was checked, not assumed, and the magnitude does vary by ~8 points of preference rate across categories even though the direction never does.

## Priority 5 (writing task, verified before writing): style among human-written résumés

Checked whether the 9 linguistic features predict which HUMAN-written résumé wins against its AI-polished counterpart (i.e. holding "is this AI-polished" fixed at "no"). The claim to verify was |r| <= 0.173 with token entropy reversed in direction. The only precomputed split in this repo (`new_metrics_results_winning`/`_losing`) has 5 of 9 features and excludes entropy entirely, so it could not confirm the entropy claim directly. Recomputing "human won >=1 of 7 models" from the current complete data (1,667 won / 442 lost - an exact row-count match to the old `human_win_tally.csv`, confirming this is the historical definition) reproduces the claim closely:

| Feature | r | p |
|---|---|---|
| entropy | -0.118 | 1.30e-04 |
| perplexity | -0.118 | 1.30e-04 |
| sentence_evenness | -0.044 | 0.156 |
| lexical_density | 0.038 | 0.213 |
| repetition | 0.029 | 0.353 |
| rolling_entropy | -0.019 | 0.535 |
| tonal_stability | 0.015 | 0.627 |
| pattern_regularity | 0.008 | 0.792 |
| sentence_entropy_variance | -0.002 | 0.948 |

All |r| <= 0.173 (entropy largest at r=-0.118) - reversed in direction from the AI-vs-human comparison, where AI-polished résumés show *higher* entropy). A stricter "won a majority of 7 models" definition (114 won / 1995 lost - much smaller, rarer group) shows the same qualitative pattern at roughly double the magnitude, noisier given the smaller n but not a different story. See `paper_text_suggestions.md` for the bounded paragraph drafted for the paper from this result.

## Known limitations (carried into any write-up, not hidden)

1. **Lexical, not semantic.** Keyword matching cannot recognize paraphrases ("managed a team" vs. "led cross-functional teams") as the same qualification. This over-counts differences, making the preserved subset conservative — a preference that survives it is more convincing, not less.
2. **Blind to non-numeric, non-dictionary puffery.** Framing like "proven track record of leadership" is invisible unless it maps to a dictionary term. This is the genuine residual gap; a semantic-entailment audit is planned for camera-ready.
3. **Power vs. strictness.** Reported as a full sweep (t=0.90/0.95/1.00) rather than one hand-picked cutoff.
4. **Says nothing about temporal shift** (2021-source-corpus vs. 2026-norms).
5. **7 of 8 models available** (granite4:3b missing) — see 'Model coverage' above. A limitation of this particular run of the audit, distinct from the four limitations the audit spec anticipated, but now a minor one rather than the dominant one.
6. **923/2109 pairs (43.8%) excluded via undefined_retention** because the reused keyword dictionary found zero scoped matches — concentrated in AUTOMOBILE, ARTS, AGRICULTURE, CONSULTANT, AVIATION. Preserved-subset findings should not be generalized to those categories.
7. **94/2109 pairs carry a broken template placeholder** (e.g. `[Number]` left unfilled in the polished text — see 'Additional finding' above); most were already excluded for other reasons, but 24 would otherwise have passed the t=0.90 gate and are excluded specifically because of this defect. Structural, not a post-hoc patch.

## Result interpretation

Preserved-subset rate (0.787) is close to the local baseline (0.790), and the cluster-robust test does not detect a significant difference. **The preference persists when measured qualifications are held constant**, for the 7 models with available data. This does not license "proven purely stylistic" — limitation #2 (residual non-dictionary puffery) remains open.

## Where every excluded pair is documented

`preservation_scores.csv` has one row per pair with an explicit `undefined_retention` flag and per-threshold `preserved_*` flags — the full, queryable, per-pair record of why any given pair is in or out of each subset. This summary reports it aggregated by category rather than as a 2,109-row list.
