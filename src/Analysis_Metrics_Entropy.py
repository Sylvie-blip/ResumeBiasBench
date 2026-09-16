import argparse
import re
import os
import glob
import math
import csv
from collections import Counter
from statistics import variance, mean


# plotting
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

"""USAGE:
1 text analysis: 
python Analysis_Metrics_Entropy.py text1.txt \
  --entropy \
  --perplexity \
  --rolling-entropy \
  --sentence-entropy-variance \
  --single-score \
  --out output.txt


2 text comparison:
python Analysis_Metrics_Entropy.py text1.txt text2.txt \
  --entropy \
  --perplexity \
  --rolling-entropy \
  --sentence-entropy-variance \
  --cross-entropy \
  --kl \
  --style-distance \
  --single-score \
  --out output.txt \
  
Batch analyze all .txt files in a directory and plot per-metric:
python /Users/sylviadong/Documents/Resume_Bias/Bias_code/Analysis_Metrics_Entropy.py \
  --dir /Users/sylviadong/Documents/humanresumes_losing \
  --plot-dir /Users/sylviadong/Documents/new_ent_results_losing \
  --entropy --perplexity --rolling-entropy --sentence-entropy-variance 

Batch analyse 2 directories and compare:
python3 /Users/sylviadong/Documents/Resume_Bias/Bias_code/Analysis_Metrics_Entropy.py \
  --dir /Users/sylviadong/Documents/humanresumes \
  --dir2 /Users/sylviadong/Documents/airesumes \
  --label1 Human --label2 AI \
  --plot-dir /Users/sylviadong/Documents/new_metrics_results \
  --entropy --perplexity --rolling-entropy --sentence-entropy-variance


  note --single-score only applies to text1.txt in 2 text comparison

  Single text score
        Higher → richer, noisier, more human-like
        Lower → smoother, more normalized, AI-like
  Style distance (two texts)
        Near 0 → stylistically similar
        Large value → different authors or generation process


Metric interpretations:

entropy: how spread out the word usage is -- low = predicable/formulaic common w/ AI
perplexity: how surprising/how many word choices there are in itself -- low = predictable/conventional w/ AI
rolling entropy: whether entropy is constant -- low = consistent style common w/ AI
sentence entropy variance: variance in sentence complexity -- low = uniform sentences common w/ AI  
cross entropy: how surprising text1 is when modeled by text2 -- low = similar style
kl divergence: how different text1 is from text2 -- low = similar style

in total lower score = more AI-like
"""


# ------------------ Helpers ------------------



def tokenize_words(text):
    return re.findall(r"\b[a-zA-Z']+\b", text.lower())

def tokenize_sentences(text):
    return [s for s in re.split(r'[.!?]+', text) if s.strip()]

def word_distribution(words):
    counts = Counter(words)
    total = len(words)
    return {w: c / total for w, c in counts.items()}

# ------------------ Core Metrics ------------------

def token_entropy(words):
    dist = word_distribution(words)
    return -sum(p * math.log2(p) for p in dist.values())

def perplexity_approx(words):
    return 2 ** token_entropy(words)

def cross_entropy(words_p, words_q, epsilon=1e-12):
    p = word_distribution(words_p)
    q = word_distribution(words_q)

    return -sum(
        p[w] * math.log2(q.get(w, epsilon))
        for w in p
    )

def kl_divergence(words_p, words_q, epsilon=1e-12):
    p = word_distribution(words_p)
    q = word_distribution(words_q)

    return sum(
        p[w] * math.log2(p[w] / q.get(w, epsilon))
        for w in p
    )

# ------------------ Variance / Burstiness ------------------

def rolling_entropy(words, window=50):
    if len(words) < window:
        return 0

    entropies = []
    for i in range(len(words) - window + 1):
        chunk = words[i:i + window]
        entropies.append(token_entropy(chunk))

    return variance(entropies) if len(entropies) > 1 else 0

def sentence_entropy_variance(sentences):
    entropies = []
    for s in sentences:
        words = tokenize_words(s)
        if words:
            entropies.append(token_entropy(words))

    return variance(entropies) if len(entropies) > 1 else 0

# ------------------ Style Distance ------------------

def style_distance(metrics):
    """
    Weighted normalized style distance score
    Lower = more similar / more stable
    """
    weights = {
        "entropy": 1.0,
        "perplexity": 0.5,
        "rolling_entropy": 1.2,
        "sentence_entropy_variance": 1.2,
        "cross_entropy": 1.5,
        "kl_divergence": 1.5
    }

    score = 0
    for k, v in metrics.items():
        if k in weights:
            score += weights[k] * v

    return score

def single_text_score(words, sentences):
    """
    Produces a single scalar score representing
    stylistic richness + variance
    """
    return (
        token_entropy(words)
        + 0.5 * perplexity_approx(words)
        + rolling_entropy(words)
        + sentence_entropy_variance(sentences)
    )

# ------------------ Main ------------------

def main():
    parser = argparse.ArgumentParser(description="Advanced text style analyzer")

    parser.add_argument("file1", nargs="?", help="First input file (or omitted when --dir is used)")
    parser.add_argument("file2", nargs="?", help="Optional second file for pairwise comparison")
    parser.add_argument("--dir", help="Directory containing .txt files to batch-analyze")
    parser.add_argument("--dir2", help="Second directory for comparison; if provided, both dirs will be plotted together")
    parser.add_argument("--label1", help="Label for first directory in comparison plots (optional)")
    parser.add_argument("--label2", help="Label for second directory in comparison plots (optional)")
    parser.add_argument("--plot-dir", default="metric_plots", help="Directory where metric plots will be saved when using --dir")

    parser.add_argument("--entropy", action="store_true")
    parser.add_argument("--perplexity", action="store_true")
    parser.add_argument("--cross-entropy", action="store_true")
    parser.add_argument("--kl", action="store_true")
    parser.add_argument("--rolling-entropy", action="store_true")
    parser.add_argument("--sentence-entropy-variance", action="store_true")
    parser.add_argument("--style-distance", action="store_true")
    parser.add_argument("--single-score", action="store_true")

    parser.add_argument("--out", help="Output results to text file")

    args = parser.parse_args()

    # If --dir provided, run batch mode: compute metrics for each .txt file found and plot per-metric
    if args.dir:
        dir_path = os.path.abspath(args.dir)
        txt_files = sorted(glob.glob(os.path.join(dir_path, "*.txt")))

        if not txt_files:
            print(f"No .txt files found in {dir_path}")
            return

        # Decide which metrics to compute. If no metric flags provided, compute a sensible default set.
        metric_flags = {
            "entropy": args.entropy,
            "perplexity": args.perplexity,
            "rolling_entropy": args.rolling_entropy,
            "sentence_entropy_variance": args.sentence_entropy_variance,
            "single_text_score": args.single_score,
        }

        if not any(metric_flags.values()):
            # default set
            metric_flags = {k: True for k in metric_flags}

        all_results_a = []
        metric_names = set()

        for path in txt_files:
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    text = f.read()
            except Exception:
                # fallback to latin-1 if utf-8 fails (common for PDF->txt conversions)
                with open(path, 'r', encoding='latin-1', errors='ignore') as f:
                    text = f.read()

            words = tokenize_words(text)
            sentences = tokenize_sentences(text)

            res = {"file": os.path.basename(path)}

            if metric_flags.get("entropy"):
                res["entropy"] = token_entropy(words)
                metric_names.add("entropy")

            if metric_flags.get("perplexity"):
                res["perplexity"] = perplexity_approx(words)
                metric_names.add("perplexity")

            if metric_flags.get("rolling_entropy"):
                res["rolling_entropy"] = rolling_entropy(words)
                metric_names.add("rolling_entropy")

            if metric_flags.get("sentence_entropy_variance"):
                res["sentence_entropy_variance"] = sentence_entropy_variance(sentences)
                metric_names.add("sentence_entropy_variance")

            if metric_flags.get("single_text_score"):
                res["single_text_score"] = single_text_score(words, sentences)
                metric_names.add("single_text_score")

            all_results_a.append(res)

        # If a second directory is provided, compute metrics for it as well
        all_results_b = []
        label_a = args.label1 or os.path.basename(os.path.abspath(args.dir))
        label_b = args.label2 or (os.path.basename(os.path.abspath(args.dir2)) if args.dir2 else None)

        if args.dir2:
            dir2_path = os.path.abspath(args.dir2)
            txt_files_b = sorted(glob.glob(os.path.join(dir2_path, "*.txt")))
            if not txt_files_b:
                print(f"No .txt files found in {dir2_path}")
                # continue with only dir A
            else:
                for path in txt_files_b:
                    try:
                        with open(path, 'r', encoding='utf-8') as f:
                            text = f.read()
                    except Exception:
                        with open(path, 'r', encoding='latin-1', errors='ignore') as f:
                            text = f.read()

                    words = tokenize_words(text)
                    sentences = tokenize_sentences(text)

                    res = {"file": os.path.basename(path)}

                    if metric_flags.get("entropy"):
                        res["entropy"] = token_entropy(words)
                        metric_names.add("entropy")

                    if metric_flags.get("perplexity"):
                        res["perplexity"] = perplexity_approx(words)
                        metric_names.add("perplexity")

                    if metric_flags.get("rolling_entropy"):
                        res["rolling_entropy"] = rolling_entropy(words)
                        metric_names.add("rolling_entropy")

                    if metric_flags.get("sentence_entropy_variance"):
                        res["sentence_entropy_variance"] = sentence_entropy_variance(sentences)
                        metric_names.add("sentence_entropy_variance")

                    if metric_flags.get("single_text_score"):
                        res["single_text_score"] = single_text_score(words, sentences)
                        metric_names.add("single_text_score")

                    all_results_b.append(res)

        # Ensure output/plot directory exists
        plot_dir = args.plot_dir
        os.makedirs(plot_dir, exist_ok=True)

        # Write combined CSV (with group column) and also separate CSVs for A/B when available
        combined_csv = os.path.join(plot_dir, 'metric_scores_combined.csv')
        fieldnames = ['group', 'file'] + sorted(metric_names)
        with open(combined_csv, 'w', newline='', encoding='utf-8') as csvf:
            writer = csv.DictWriter(csvf, fieldnames=fieldnames)
            writer.writeheader()
            for r in all_results_a:
                row = {'group': label_a, 'file': r.get('file')}
                for m in metric_names:
                    row[m] = r.get(m, '')
                writer.writerow(row)
            if all_results_b:
                for r in all_results_b:
                    row = {'group': label_b, 'file': r.get('file')}
                    for m in metric_names:
                        row[m] = r.get(m, '')
                    writer.writerow(row)

        print(f"Wrote combined scores to {combined_csv}")

        # also write per-directory CSVs
        csv_a = os.path.join(plot_dir, 'metric_scores_A.csv')
        with open(csv_a, 'w', newline='', encoding='utf-8') as csvf:
            writer = csv.DictWriter(csvf, fieldnames=['file'] + sorted(metric_names))
            writer.writeheader()
            for r in all_results_a:
                row = {k: r.get(k, '') for k in (['file'] + sorted(metric_names))}
                writer.writerow(row)
        print(f"Wrote scores for dir A to {csv_a}")

        if all_results_b:
            csv_b = os.path.join(plot_dir, 'metric_scores_B.csv')
            with open(csv_b, 'w', newline='', encoding='utf-8') as csvf:
                writer = csv.DictWriter(csvf, fieldnames=['file'] + sorted(metric_names))
                writer.writeheader()
                for r in all_results_b:
                    row = {k: r.get(k, '') for k in (['file'] + sorted(metric_names))}
                    writer.writerow(row)
            print(f"Wrote scores for dir B to {csv_b}")

        # Create one histogram per metric. If dir2 provided, overlay both datasets in different colors.
        for metric in sorted(metric_names):
            vals_a = [r.get(metric, None) for r in all_results_a]
            numeric_a = [v for v in vals_a if isinstance(v, (int, float))]

            vals_b = [r.get(metric, None) for r in all_results_b] if all_results_b else []
            numeric_b = [v for v in vals_b if isinstance(v, (int, float))]

            plt.figure(figsize=(10, 6))
            # if both present, overlay with same bins
            if numeric_a and numeric_b:
                n_a, bins, patches = plt.hist(numeric_a, bins='auto', color='C0', alpha=0.6, label=label_a)
                plt.hist(numeric_b, bins=bins, color='C1', alpha=0.6, label=label_b)
                plt.legend()
                out_png = os.path.join(plot_dir, f"{metric}_compare.png")
                plt.title(f"{metric.replace('_', ' ').title()} — {label_a} vs {label_b}")
            else:
                # only one dataset available, plot single histogram
                if numeric_a:
                    try:
                        plt.hist(numeric_a, bins='auto', color='C0', alpha=0.8)
                    except Exception:
                        plt.hist(numeric_a, bins=20, color='C0', alpha=0.8)
                    out_png = os.path.join(plot_dir, f"{metric}.png")
                    plt.title(metric.replace('_', ' ').title())
                elif numeric_b:
                    try:
                        plt.hist(numeric_b, bins='auto', color='C1', alpha=0.8)
                    except Exception:
                        plt.hist(numeric_b, bins=20, color='C1', alpha=0.8)
                    out_png = os.path.join(plot_dir, f"{metric}.png")
                    plt.title(metric.replace('_', ' ').title())
                else:
                    # nothing numeric to plot for this metric
                    print(f"No numeric values for metric '{metric}' in either directory; skipping plot.")
                    plt.close()
                    continue

            plt.xlabel(metric.replace('_', ' ').title())
            plt.ylabel('Count')
            plt.tight_layout()
            plt.savefig(out_png)
            plt.close()
            print(f"Saved histogram: {out_png}")

        return

    # ------------------ Single / pairwise file mode ------------------

    if not args.file1:
        print("Please provide a file path or use --dir to analyze a directory of .txt files.")
        return

    with open(args.file1, "r", encoding="utf-8") as f:
        text1 = f.read()

    words1 = tokenize_words(text1)
    sentences1 = tokenize_sentences(text1)

    results = {}

    if args.entropy:
        results["entropy"] = token_entropy(words1)

    if args.perplexity:
        results["perplexity"] = perplexity_approx(words1)

    if args.rolling_entropy:
        results["rolling_entropy"] = rolling_entropy(words1)

    if args.sentence_entropy_variance:
        results["sentence_entropy_variance"] = sentence_entropy_variance(sentences1)

    if args.file2:
        with open(args.file2, "r", encoding="utf-8") as f:
            text2 = f.read()

        words2 = tokenize_words(text2)

        if args.cross_entropy:
            results["cross_entropy"] = cross_entropy(words1, words2)

        if args.kl:
            results["kl_divergence"] = kl_divergence(words1, words2)

    if args.style_distance:
        results["style_distance"] = style_distance(results)

    if args.single_score:
        results["single_text_score"] = single_text_score(words1, sentences1)

    # ------------------ Output ------------------

    output_lines = []
    for k, v in results.items():
        line = f"{k}: {v}"
        print(line)
        output_lines.append(line)

    if args.out:
        title = os.path.basename(args.file1)

        with open(args.out, "a", encoding="utf-8") as f:
            f.write(f"\n    {title}:\n")
            f.write("\n".join(output_lines) + "\n")

if __name__ == "__main__":
    main()
