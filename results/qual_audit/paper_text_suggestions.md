# Suggested paper text (drafted from the audit, not inserted anywhere)

No editable LaTeX/Word source for `ACM_AIES_2026_Paper.pdf` was found in this
repo (only the compiled PDF) — these are text blocks to paste into the actual
source manually. Each is labeled with which priority/section it addresses.

---

## Priority 4: one-sentence fix for the existing Table 4 discussion

Insert after the sentence beginning "The largest weights are assigned to
token entropy ... and perplexity ...":

> Because perplexity is defined as a monotonic transform of entropy
> (perplexity = 2^H), these two coefficients are not independent evidence —
> their large, opposite-signed magnitudes reflect collinearity between two
> encodings of the same underlying quantity, and should be read as one
> signal, not two.

---

## Priority 5: style among human-written résumés (bounded paragraph)

**Verification note (not for the paper, context for whoever reviews this):**
the claimed numbers (|r| ≤ 0.173, entropy reversed) were checked against data,
not taken on faith. The only precomputed winning/losing split in this repo
(`data/results/new_metrics_results_winning`/`_losing`) has just 5 of 9
features and excludes entropy entirely, so it couldn't confirm the entropy
claim directly. Recomputing "human resume won against at least one of the 7
available models" (matching the old `human_win_tally.csv`'s definition
exactly — 1,667 won / 442 lost, an exact row-count match) against the real,
complete 9-feature data reproduces the claim closely: every |r| ≤ 0.173, with
entropy the largest at r = −0.118 and reversed in direction, exactly as
described. A stricter "won a majority of 7 models" definition (only 114 pairs
qualify) shows the same qualitative pattern but roughly double the magnitude
(entropy r ≈ −0.32) — noisier given the much smaller n, but not a different
story. The paragraph below uses the better-powered, historically-matched
1,667/442 split.

**Paragraph for the paper:**

> Restricting this analysis to human-written résumés alone and asking whether
> the same nine linguistic features predict which one wins against its
> AI-polished counterpart, the picture is far weaker than the AI-vs-human
> comparison in Table 2. Effect sizes are small (|r| ≤ 0.173 across all nine
> features), and — notably — token entropy runs in the *opposite* direction
> from what an AI-style-proximity account would predict: human résumés that
> win exhibit *lower* entropy than those that lose, whereas AI-polished
> résumés show *higher* entropy than human-written ones overall. Surface
> stylistic features therefore only partially account for the preference
> observed among human-written documents alone; this result bounds rather
> than confirms the signal-corruption account of the paper's central finding.
> The paired human-vs-AI audit carries the primary evidential weight for that
> claim, at a significance level (p < 10⁻²⁰⁰) that leaves ample room for this
> more modest, partially-inconsistent result to stand alongside it without
> weakening the overall argument.

