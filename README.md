# ResumeBiasBench

**ResumeBiasBench** is the first benchmark for studying AI-induced stylistic bias in automated resume screening systems. It examines whether LLM-based screeners systematically favor generative-AI-polished resumes over human-written ones — regardless of actual candidate qualifications.

This repository contains the dataset, code, and figures accompanying the paper:

> **ResumeBiasBench: Benchmarking AI-Induced Stylistic Bias in Automated Resume Screening**  
> ACM AIES 2026

---

## Key Findings

- Across 8 open-source LLMs, AI-polished resumes were selected **79.0%** of the time in head-to-head comparisons against human-written originals from the same candidate.
- The bias is driven by measurable surface-level stylistic differences, not qualifications. The two strongest discriminating features are **lexical density** (r = −0.618) and **type-token ratio** (r = −0.609).
- A Random Forest classifier distinguishes AI-polished from human-written resumes with **AUC ≈ 0.91**, confirming that stylistic artifacts are systematic and detectable.
- Structuring resumes as JSON further amplifies screener preference for AI-modified content.

---

## Dataset

The benchmark is built on the publicly available [Kaggle Resume Dataset](https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset) (Bhawal 2021, CC0 license), sourced from livecareer.com.

After cleaning: **2,109 resumes** across **24 occupational categories**.

Three parallel versions exist per resume:
| Version | Location | Description |
|---|---|---|
| Human-written | `data/humanresumes/` | Original plaintext resumes |
| AI-polished | `data/airesumes/` | GPT-4o-mini polished versions |
| Human JSON | `data/humanresumesjson/` | Structured JSON of human resumes |
| AI JSON | `data/airesumesjson/` | Structured JSON of AI resumes |
| Source PDFs | `data/sorted_data/` | Original PDFs organized by job category |

> **Note:** The `data/` corpus directories are excluded from this repository due to size. Download the source PDFs from Kaggle and run the preprocessing pipeline to regenerate them (see [Reproducing Results](#reproducing-results)).

---

## Linguistic Features

Nine surface-level features are extracted per resume:

| Feature | Description |
|---|---|
| Token Entropy | Shannon entropy of word distribution |
| Perplexity | 2^H (rescaled entropy) |
| Sliding-Window Entropy Mean | Mean entropy across 50-token windows |
| Sliding-Window Entropy Variance | Variance across sliding windows |
| Sentence Entropy Variance | Variance of per-sentence entropy |
| Lexical Density | Proportion of content words (spaCy POS) |
| Type-Token Ratio (TTR) | Unique types / total tokens |
| Pattern Regularity | Proportion of repeated trigrams |
| Sentiment Variance | Variance of VADER per-sentence scores |

---

## Screening Models

Eight open-source LLMs served locally via [Ollama](https://ollama.com):

- `qwen2.5:3b`, `qwen2.5:14b`, `qwen2.5:32b`
- `gemma3:4b`, `gemma3:12b`
- `mistral-small3.2:latest`
- `granite4:3b`
- `cogito:3b`

AI polishing was performed using **ChatGPT-4o-mini** via the OpenAI API.

---

## Repository Structure

```
.
├── run_screening.py              # Main pairwise LLM screening pipeline
├── prompts/
│   └── ranking_prompt.txt        # Prompt template used for LLM comparisons
├── src/
│   ├── Metrics.py                # Compute all 9 linguistic features
│   ├── Analysis_Metrics.py       # Feature analysis and group comparisons
│   ├── Analysis_Metrics_Entropy.py
│   ├── conv_to_json.py           # Convert plaintext resumes to JSON
│   ├── merge_csv.py              # Merge metric CSV outputs
│   ├── sort_folder.py            # Filter resume folders by CSV allowlist
│   ├── split_by_threshold.py     # Split dataset by human-win threshold
│   ├── stats_from_csv.py         # Statistical summary of CSV metric files
│   ├── classifiers/
│   │   └── logistic_regression.py  # All 6 ML classifiers + ROC curves
│   ├── scripts/
│   │   ├── pdf_to_txt.py         # Extract plaintext from source PDFs
│   │   └── compare_metrics.py    # Mann-Whitney U tests (Table 2)
│   └── visualization/
│       ├── bar_stats.py          # AI vs. human preference bar charts
│       ├── box_plots.py          # Box plots across dataset splits
│       ├── box_plots_clean.py    # Cleaned box plot version
│       ├── histograms.py         # Histogram overlays across CSV groups
│       └── job_category_bar.py   # Resume count per job category (Figure 1)
└── data/
    ├── human_win_tally.csv       # Per-resume human win counts across all models
    ├── human_win_tally - 1s removed.csv
    ├── results/                  # LLM screening outputs and metric score CSVs
    └── figures/                  # All paper figures (PNGs)
        ├── compare_*.png         # Figure 3: 9-metric human vs. AI distributions
        ├── resume_categories_counts_new_part*.png  # Figure 1
        └── roc_plots/            # Figure 4: ROC curves for 6 classifiers
```

---

## Reproducing Results

### 1. Install dependencies

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

Install and start [Ollama](https://ollama.com), then pull the screening models:

```bash
ollama pull qwen2.5:3b
# repeat for each model listed above
```

### 2. Preprocess source PDFs → plaintext

```bash
python src/scripts/pdf_to_txt.py \
  --input data/sorted_data/ \
  --output data/humanresumes/
```

### 3. Generate AI-polished resumes

Use `prompts/ranking_prompt.txt` as the system prompt with the OpenAI GPT-4o-mini API to produce `data/airesumes/`.

### 4. Convert to JSON

```bash
python src/conv_to_json.py   # outputs to data/humanresumesjson/ and data/airesumesjson/
```

### 5. Run pairwise LLM screening

```bash
python run_screening.py \
  --model qwen2.5:3b \
  --script prompts/ranking_prompt.txt \
  --ai_dir data/airesumes \
  --human_dir data/humanresumes \
  --out_file data/results/pairwise_qwen2.5_3b.txt
```

Repeat for each of the 8 models.

### 6. Extract linguistic features

```bash
python src/Metrics.py \
  --dir data/humanresumes \
  --dir2 data/airesumes \
  --label1 Human --label2 AI \
  --plot-dir data/results/metric_scores
```

### 7. Statistical analysis (Table 2)

```bash
python src/scripts/compare_metrics.py data/results/metric_scores/metric_scores_combined.csv
```

### 8. Train classifiers (Table 3 + Figure 4)

```bash
python src/classifiers/logistic_regression.py
```

---

## Citation

If you use ResumeBiasBench in your research, please cite:

```bibtex
@inproceedings{resumebiasbench2026,
  title     = {ResumeBiasBench: Benchmarking AI-Induced Stylistic Bias in Automated Resume Screening},
  booktitle = {Proceedings of the AAAI/ACM Conference on AI, Ethics, and Society (AIES)},
  year      = {2026}
}
```

---

## License

Code: MIT License  
Dataset: The underlying Kaggle Resume Dataset is released under [CC0 1.0 Public Domain](https://creativecommons.org/publicdomain/zero/1.0/).
