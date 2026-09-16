# Figure Report

Generated for the ResumeBiasBench paper. All rates below are recomputed
directly from raw win counts or from the pre-existing `results/qual_audit/`
analysis outputs (never read out of the paper's own tables), per the brief's
ground rules. Two exceptions to that rule are explicit and noted where they
occur: granite4:3b (Figures 6/8), which has no per-decision file anywhere in
this repository, and the paper's own Table 1 rates, which are shown only as
a comparison column, never as the plotted value.

Two source scripts were changed, additively only, and backed up first:
- `src/classifiers/logistic_regression.py` → `.bak`: perplexity-drop is now
  behind an `--include-perplexity` flag (default off, identical to prior
  behavior). Diff is a pure addition; the one changed line produces the
  same `drop_cols` list as before when the flag is absent.
- `src/scripts/compare_metrics.py` → `.bak`: now also writes
  `results/mannwhitney_effect_sizes.csv` (one row per feature) in addition
  to the unchanged console output. Diff is a pure addition.

No files under `data/` or `results/` were deleted or overwritten. The only
new file under `results/` is `results/mannwhitney_effect_sizes.csv`.

**Methodological note on `logistic_regression.py`:** the script hardcodes
an absolute input path, `/Users/sylviadong/Documents/new_merged.csv`, which
did not exist on disk (a pre-existing quirk, not touched by the additive
flag change — the brief's "change nothing else" was taken literally). To
run it, `data/results/new_merged.csv` was copied to that exact path; this
is a file created outside the project directory, not a modification to
`data/` or `results/`.

**Interval methods used (stated once here, per the brief):**
- *Wilson score interval* for a simple proportion, computed where no
  existing interval already covers a rate: per-model full-corpus rates
  (Figures 6, 7, 11's full-corpus series), each backed by one independent
  decision per pair, so no clustering applies.
- *Cluster-robust interval, clustered on `pair_id`* (OLS of the AI-win
  indicator, HC-cluster SEs, t-distribution with n_clusters−1 df), reused
  as-is wherever `results/qual_audit/` already computed one by this method:
  per-category rates (Figure 10, from `category_breakdown.csv`) and
  preserved-subset rates (Figure 11's subset series, from
  `preserved_subset_winrates.csv`). This matches
  `src/qual_audit/analysis.py`'s existing methodology; it was not
  recomputed by a different method per the brief's CI ground rule.

**Tie count (ground rules, checked before Figure 6):** could not be
computed, and not just left unreported. `pick_winner()` in
`run_screening.py` resolves `key(ai) >= key(human)` in AI's favor, but
`run_screening.py` writes only the **winner's** `(total, experience,
technical)` triple to the output file — the loser's score is discarded and
never written anywhere. Verified directly: every `results_*.txt` file
contains exactly one `SCORE:` line per `PAIR` block, for the winner only
(confirmed by inspection of `results_cogito_3b.txt` and by
`parse_screeners.py`'s own regex, which only captures `WINNER:`). Since a
decision is a tie exactly when `key(ai) == key(human)`, and at most one of
the two keys is ever recorded per decision, there is no stored data in this
repository from which the true tie count or tie rate can be reconstructed
for any of the 14,763 decisions. This is a data-collection gap in the
original screening run, not a computation this audit failed to do. Because
the rate cannot be quantified, Figure 6 was not given a second series or a
footnoted "ties excluded" rate (the brief's threshold for that was ">1% of
decisions," which requires a number that does not exist) — instead its
caption states the caveat directly: AI-preference rates should be read
knowing that an unknown fraction of AI wins could be ties awarded to AI by
the scoring rule, not clear wins.

---

## Figure 5 — `l1_coefficients_perplexity.{png,pdf}`

- **Source:** `data/results/new_merged.csv` (4,218 rows: 2,109 Human + 2,109
  AI), via `src/classifiers/logistic_regression.py` run twice (default, and
  with `--include-perplexity`). Fixed `random_state=1234`, `test_size=0.80`,
  L1 penalty with `C=1/0.0001=10000` (very weak regularization).
- **n:** 4,218 texts (843 in the 20% train split used to fit the model).
- **Interval method:** n/a — these are point coefficients, not proportions.
- **Table 4 reproduction check (required before plotting):** the
  with-perplexity run gives entropy = **+5.990217**, perplexity =
  **−6.270441**, matching the paper's Table 4 pair (+5.990 / −6.270) to
  three decimal places. The classifier is unchanged since Table 4 was
  produced, so plotting proceeded.
- **Disagreement found:** the paper's text describes the *without*-perplexity
  panel as showing "only entropy and lexical density survive." The
  recomputed without-perplexity coefficients do not match that description:
  lexical_density is indeed the largest (+1.867), but entropy is one of the
  **smallest** surviving coefficients (−0.235) — smaller in magnitude than
  repetition (−0.849), pattern_regularity (−0.780), rolling_entropy
  (−0.325), and sentence_evenness (−0.297). With `C=10000` the L1 penalty is
  extremely weak and does not zero out any coefficient in either run (the
  smallest, tonal_stability, is only ≈0.01-0.04), so "survive" in the sense
  of sparsity does not really describe either panel. **Plotted as
  recomputed**, not adjusted to match the paper's description; flagged here
  per the ground rules.
- All 9 without-panel and with-panel coefficients (for the record):
  - Without: entropy −0.234709, rolling_entropy −0.324682,
    sentence_entropy_variance −0.175895, lexical_density +1.866775,
    pattern_regularity −0.780122, repetition −0.849170, sentence_evenness
    −0.297083, tonal_stability +0.012901.
  - With: entropy +5.990217, perplexity −6.270441, rolling_entropy
    −0.288538, sentence_entropy_variance −0.180860, lexical_density
    +1.616888, pattern_regularity −0.551018, repetition −0.342498,
    sentence_evenness −0.270952, tonal_stability −0.036697.

## Figure 6 — `preference_by_model.{png,pdf}`

- **Source:** `results_qwen2.53b.txt`, `results_qwen2.5_14b.txt`,
  `results_qwen2.5_32b.txt`, `results_gemma3_4b.txt`,
  `results_gemma3_12b.txt`, `results_mistral-small3.2_latest.txt`,
  `results_cogito_3b.txt` (raw `WINNER:` counts, grep'd directly, not read
  from any precomputed CSV). granite4:3b's point is the paper's own Table 1
  value (0.785) — the brief's explicit, sole exception to "never read a rate
  from the paper's tables," because no raw decisions for this model exist
  anywhere in the repo (confirmed: no `results_granite*` file, no mention of
  a raw count anywhere outside `README.md`/`audit_summary.md` prose).
- **n:** 2,109 decisions per model for the 7 directly-computed models;
  granite4:3b's n is unknown (no per-decision file), so it is drawn with an
  open marker and no CI, and its true n is not claimed to be 2,109.
- **Interval method:** Wilson score interval, k/n = AI-win count / 2,109,
  per model.
- **Disagreements found:** qwen2.5:14b recomputes to 1874/2109 = 0.888573,
  which rounds to **0.889**, not the paper's published **0.888** (the exact
  rounding error the brief flagged in advance). All 6 other models'
  recomputed rates match the paper's Table 1 values to the precision given
  (cogito:3b 0.8800, gemma3:12b 0.7899, gemma3:4b 0.7165, mistral-small3.2
  0.7928, qwen2.5:32b 0.8634, qwen2.5:3b 0.6017) — no further disagreements.
- **Ties:** see the ground-rules section above — the count is not
  computable from stored data; the caveat is stated in the figure caption.

## Figure 7 — `qwen_scale.{png,pdf}`

- **Source:** `results_qwen2.53b.txt`, `results_qwen2.5_14b.txt`,
  `results_qwen2.5_32b.txt` (raw `WINNER:` counts).
- **n:** 2,109 pairs at each of 3 sizes.
- **Interval method:** Wilson score interval.
- **Values:** 3B: 1269/2109 = 0.601707; 14B: 1874/2109 = 0.888573; 32B:
  1821/2109 = 0.863442 — matching the brief's stated numbers exactly.
- **Disagreement found:** same qwen2.5:14b rounding note as Figure 6
  (0.889 recomputed vs. 0.888 published). No trend line was fit, per the
  brief; the series is presented as three points, one model family,
  non-monotonic, suggestive only.

## Figure 8 — `json_mitigation.{png,pdf}` — **NOT PRODUCED**

No data exists anywhere in this repository for a JSON-structured screening
condition. Searched: the repo root (only the 7 plain-text `results_*.txt`
files and the superseded `qwen2.5.txt` exist), `data/` and `results/`
(including `results/qual_audit/`), and `Useless/` (only 2 stray plain-text
copies, both already accounted for as superseded partial runs). `README.md`
states "Structuring resumes as JSON further amplifies screener preference"
as a key finding, and `data/airesumesjson/` / `data/humanresumesjson/` hold
JSON-*formatted resumes*, but no screening run was ever executed against
them that left a result file in this repo, under any name — there is only
one prompt template (`prompts/ranking_prompt.txt`) and no JSON-specific
variant.

Per the brief's instructions, this gap is reported rather than papered
over: no substitute data source was used, and no number was pulled from the
paper's prose to fill the gap (which the ground rules explicitly forbid).
**Figure 8 requires re-running the JSON-condition screening (or locating an
existing output file outside this repository) before it can be produced.**
No script was written for it, since there is nothing for a script to load.

## Figure 10 — `category_forest.{png,pdf}`

- **Source:** `results/qual_audit/category_breakdown.csv` (already computed:
  cluster-robust rate and 95% CI per category, pooled across all 7 available
  models' decisions in that category).
- **n:** 24 categories; n_pairs ranges 17 (BPO) to 109 (FINANCE); n_decisions
  = n_pairs × (number of models with a decision on that pair, ≤7).
- **Interval method:** cluster-robust (clustered on `pair_id`), reused as
  computed in `results/qual_audit/`, not recomputed by a different method.
- **Disagreement found:** none — this table has no direct equivalent in the
  paper's published tables (it replaces Table 2's row-by-row 0.5 check); the
  directional claim it supports (all 22 adequately-powered categories have a
  95% CI entirely above 0.5) was independently re-verified from the CSV here
  and matches `audit_summary.md`'s own statement.
- BPO (17 pairs) and AUTOMOBILE (28 pairs) are drawn with open, lighter
  markers (underpowered, <30 pairs) rather than omitted.

## Figure 11 — `preserved_vs_full.{png,pdf}`

- **Source:** `results/qual_audit/preserved_subset_winrates.csv`, the t=0.90
  per-model table (columns `local_baseline_k/n/rate` for the full corpus,
  `subset_k/n/rate` and `cluster_ci_95_lo/hi` for the preserved subset).
- **n:** full corpus n=2,109 per model; preserved subset n=360 per model at
  t=0.90.
- **Interval method:** preserved-subset CI reused exactly as computed in the
  source file (cluster-robust, clustered on `pair_id`). Full-corpus CI is
  not present in that file and was computed here as a Wilson interval on
  `local_baseline_k/local_baseline_n`, appropriate because each model
  contributes exactly one (non-repeated) decision per pair at the full-corpus
  level, so there is nothing to cluster.
- **Disagreement found:** none — the file's own `paper_table1_rate` column
  matches every model's recomputed `local_baseline_rate` to the given
  precision (including qwen2.5:3b: 0.60171 vs. 0.602).
- **Headline confirmed:** qwen2.5:3b's preserved-subset rate is 0.536
  (95% CI [0.485, 0.588]) — the only model whose subset interval includes
  parity (0.5) — against a full-corpus rate of 0.602 with a CI well clear of
  parity. The other 6 models' full-corpus and subset points sit within
  ~0.02 of each other.

## Figure 12 — `length_change_distribution.{png,pdf}`

- **Source:** `results/qual_audit/length_check.csv`, column `delta_tokens`
  (paper's own tokenizer, `\b[a-zA-Z']+\b`, lowercased).
- **n:** 2,109 pairs.
- **Interval method:** n/a — descriptive statistics (median, mean, %
  positive), not a proportion or regression estimate.
- **Values (recomputed):** median **−165.0**, mean **−239.17** (reported to
  1 decimal as **−239.2**), positive tail **14.0%** — all three match
  `audit_summary.md`'s reported values exactly; no disagreement found.
- **Clipping:** x-axis clipped to [−1200, 250] tokens for legibility. **29
  pairs (1.4%)** fall below −1200 (min observed: −2997) and are folded into
  a single hatched "pileup" bin at the left edge rather than dropped from
  the median/mean/tail statistics above, which use the full, unclipped data.

## Figure 13 — `mechanism_scatter.{png,pdf}`

- **Source (x-axis):** `results/mannwhitney_effect_sizes.csv` — newly
  generated by the modified `src/scripts/compare_metrics.py` run as
  `python src/scripts/compare_metrics.py data/results/new_merged.csv`
  (console output unchanged; this CSV is the additive output added for this
  figure).
- **Source (y-axis):** `results/qual_audit/feature_regression_coefficients.csv`
  (already computed; not recomputed here).
- **n:** effect sizes from 2,109 AI + 2,109 Human texts; regression
  coefficients from 14,763 decisions across 2,109 pairs (per
  `feature_regression.py`).
- **Interval method:** none plotted — point estimates only, no trend line
  and no correlation coefficient, per the brief (the figure's subject is the
  absence of a relationship, not an estimate of one).
- **Feature-count note (not a disagreement, but stated for clarity):**
  `mannwhitney_effect_sizes.csv` has 9 rows (including perplexity);
  `feature_regression_coefficients.csv` has 8 delta-features (perplexity
  dropped there for collinearity with entropy, matching the same
  collinearity this report documents for Figure 5). The plot uses the 8
  features common to both, as the brief specifies.
- **Disagreement found:** none — the two largest-magnitude effect sizes,
  lexical_density (r=−0.618) and repetition (r=−0.609), match the README's
  headline numbers for "lexical density (r=−0.618)" and "type-token ratio
  (r=−0.609)" respectively. (`repetition` is this codebase's stand-in
  column for a TTR-like quantity — a naming caveat already documented in
  `audit_summary.md`, not a new discrepancy.)
- **Finding confirmed:** no diagonal. lexical_density and repetition have
  the largest separation effect sizes but near-zero win-prediction
  coefficients (−0.178, +0.085); token entropy has a modest separation
  effect size (0.250) but the largest win-prediction coefficient (+0.150).
