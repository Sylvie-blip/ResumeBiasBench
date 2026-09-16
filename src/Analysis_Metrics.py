import argparse
import re
import os
import glob
import math
import csv
from collections import Counter
from statistics import variance

# Optional NLP helpers
_SPACY_AVAILABLE = False
_TEXTBLOB_AVAILABLE = False
_NLP = None
try:
    import spacy
    try:
        _NLP = spacy.load("en_core_web_sm")
        _SPACY_AVAILABLE = True
    except Exception:
        _NLP = None
        _SPACY_AVAILABLE = False
except Exception:
    _NLP = None
    _SPACY_AVAILABLE = False

try:
    from textblob import TextBlob
    _TEXTBLOB_AVAILABLE = True
except Exception:
    _TEXTBLOB_AVAILABLE = False

# plotting
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

"""USAGE:
python Analysis_Metrics.py resume.txt \
  --repetition \
  --lexical-density \
  --pattern-regularity \
  --entropy \
  --perplexity \
  --sentence-evenness \
  --tonal-stability

ex: keyword matching
python Analysis_Metrics.py resume.txt \
  --keywords accounting excel quickbooks \
  --entropy \
  --perplexity \
  
Batch directory:
python /Users/sylviadong/Documents/Resume_Bias/Bias_code/Analysis_Metrics.py \
  --dir /Users/sylviadong/Documents/humanresumes_losing \
  --plot-dir /Users/sylviadong/Documents/new_metrics_results_losing \
  --repetition --lexical-density --pattern-regularity --sentence-evenness --tonal-stability
  
Batch 2 directories and compare:
python3 /Users/sylviadong/Documents/Resume_Bias/Bias_code/Analysis_Metrics.py \
  --dir /Users/sylviadong/Documents/humanresumes \
  --dir2 /Users/sylviadong/Documents/airesumes \
  --label1 Human --label2 AI \
  --plot-dir /Users/sylviadong/Documents/new_metrics_results \
   --repetition --lexical-density --pattern-regularity --sentence-evenness --tonal-stability

Batch w/ keywords:
python Analysis_Metrics.py \
  --dir /Users/sylviadong/Documents/Resume_Bias/Bias_code/Txt_Resumes \
  --plot-dir /Users/sylviadong/Documents/Resume_Bias/Bias_code/H_Met_plots \
  --keywords accounting excel quickbooks
"""


# =========================
# Tokenization helpers
# =========================

def tokenize_words(text):
    return re.findall(r"\b[a-zA-Z']+\b", text.lower())


def tokenize_sentences(text):
    return [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]


# =========================
# Basic metrics
# =========================

def repetition_rate(words):
    """Type-Token Ratio (TTR): unique token types divided by total token count."""
    if not words:
        return 0
    toks = list(words)
    types = set(toks)
    return len(types) / len(toks) if toks else 0


def lexical_density(words):
    """Lexical density: unique content-word types / total alphabetic tokens.
    Uses spaCy POS tags when available (NOUN, PROPN, VERB, ADJ, ADV). Falls back to
    a stopword-based approach.
    """
    if not words:
        return 0.0
    # accept list of tokens or raw text
    if isinstance(words, list):
        text = ' '.join(words)
    else:
        text = str(words)

    if _SPACY_AVAILABLE and _NLP is not None:
        doc = _NLP(text)
        content_pos = {"NOUN", "PROPN", "VERB", "ADJ", "ADV"}
        content = [token.lemma_.lower() for token in doc if token.pos_ in content_pos and token.is_alpha]
        total_alpha = sum(1 for token in doc if token.is_alpha)
        return (len(set(content)) / total_alpha) if total_alpha else 0.0
    else:
        stopwords = {
            "the","is","am","are","was","were","be","been","being",
            "a","an","and","or","but","if","then","so",
            "of","in","on","at","to","for","with","by","from"
        }
        toks = tokenize_words(text)
        content = [w for w in toks if w.lower() not in stopwords]
        return (len(set(content)) / len(toks)) if toks else 0.0


def keyword_matching_rate(words, keywords):
    if not keywords:
        return 0
    return sum(1 for w in words if w in keywords) / len(words) if words else 0


def pattern_regularity(words, n=3):
    if len(words) < n:
        return 0
    ngrams = Counter(zip(*[words[i:] for i in range(n)]))
    repeated = sum(c for c in ngrams.values() if c > 1)
    return repeated / sum(ngrams.values())


# =========================
# Variance-based metrics
# =========================

def tonal_stability(sentences):
    """Compute tonal stability using TextBlob when available, else fall back to a
    small lexicon-based scoring. Returns 1 / (1 + variance(scores)) to mirror prior
    behavior (higher -> more stable).
    """
    scores = []
    if _TEXTBLOB_AVAILABLE:
        for s in sentences:
            s_text = s.strip()
            if not s_text:
                continue
            try:
                tb = TextBlob(s_text)
                scores.append(float(tb.sentiment.polarity))
            except Exception:
                scores.append(0.0)
    else:
        positive = {"good","great","positive","benefit","success","effective","improve","achieve","lead","managed"}
        negative = {"bad","poor","negative","fail","harm","ineffective","decline"}
        for s in sentences:
            toks = tokenize_words(s)
            score = sum(w in positive for w in toks) - sum(w in negative for w in toks)
            scores.append(float(score))

    if len(scores) < 2:
        return 1.0
    try:
        return 1.0 / (1.0 + variance(scores))
    except Exception:
        return 0.0


def sentence_length_evenness(sentences):
    """Compute sentence evenness using mean sentence length, std dev and coefficient
    of variation (std / mean). Returns the coefficient of variation; lower means more even.
    """
    lengths = [len(tokenize_words(s)) for s in sentences]
    if not lengths:
        return 0.0
    if len(lengths) < 2:
        return 0.0
    mu = sum(lengths) / len(lengths)
    if mu == 0:
        return 0.0
    # population or sample std? use sample stdev
    sd = math.sqrt(sum((x - mu) ** 2 for x in lengths) / (len(lengths) - 1))
    cv = sd / mu
    return cv


# =========================
# Entropy / perplexity
# =========================

def token_entropy(words):
    if not words:
        return 0

    counts = Counter(words)
    total = len(words)
    entropy = 0

    for count in counts.values():
        p = count / total
        entropy -= p * math.log2(p)

    return entropy


def perplexity_approx(words):
    return 2 ** token_entropy(words)


# =========================
# Normalization helpers
# =========================

def clamp(x, lo=0.0, hi=1.0):
    return max(lo, min(hi, x))


def normalize(value, expected_min, expected_max):
    if expected_max == expected_min:
        return 0
    return clamp((value - expected_min) / (expected_max - expected_min))


# =========================
# Composite score
# =========================

def composite_style_score(metrics):
    """
    Returns a normalized style regularity score in [0,1]
    Higher = more regular / more AI-like
    """

    weights = {
        "repetition": 0.15,
        "lexical_density": 0.15,
        "pattern_regularity": 0.20,
        "sentence_evenness": 0.15,
        "tonal_stability": 0.15,
        "entropy": 0.10,
        "perplexity": 0.10,
    }

    normalized = {}

    if "repetition" in metrics:
        normalized["repetition"] = normalize(metrics["repetition"], 0.1, 0.6)

    if "lexical_density" in metrics:
        normalized["lexical_density"] = 1 - normalize(metrics["lexical_density"], 0.3, 0.7)

    if "pattern_regularity" in metrics:
        normalized["pattern_regularity"] = normalize(metrics["pattern_regularity"], 0.1, 0.6)

    if "sentence_evenness" in metrics:
        normalized["sentence_evenness"] = normalize(metrics["sentence_evenness"], 0.4, 1.0)

    if "tonal_stability" in metrics:
        normalized["tonal_stability"] = normalize(metrics["tonal_stability"], 0.3, 1.0)

    if "entropy" in metrics:
        normalized["entropy"] = 1 - normalize(metrics["entropy"], 6.0, 9.5)

    if "perplexity" in metrics:
        normalized["perplexity"] = 1 - normalize(metrics["perplexity"], 50, 500)

    score = 0
    total_weight = 0

    for k, v in normalized.items():
        w = weights.get(k, 0)
        score += v * w
        total_weight += w

    return score / total_weight if total_weight else 0


# =========================
# Main
# =========================

def main():
    parser = argparse.ArgumentParser(description="Stylometric text analysis tool")

    parser.add_argument("file", nargs='?', help="Path to input .txt file (omit when using --dir)")
    parser.add_argument("--dir", help="Directory containing .txt files to batch-analyze")
    parser.add_argument("--dir2", help="Second directory for comparison; if provided, both dirs will be plotted together")
    parser.add_argument("--label1", help="Label for first directory in comparison plots (optional)")
    parser.add_argument("--label2", help="Label for second directory in comparison plots (optional)")
    parser.add_argument("--plot-dir", default="metric_plots", help="Directory to save per-metric plots and CSV when using --dir")
    parser.add_argument("--output", default="Analysis_Metrics_Output.txt",
                        help="File to append analysis results to")

    parser.add_argument("--repetition", action="store_true")
    parser.add_argument("--lexical-density", action="store_true")
    parser.add_argument("--keywords", nargs="*")
    parser.add_argument("--pattern-regularity", action="store_true")
    parser.add_argument("--tonal-stability", action="store_true")
    parser.add_argument("--sentence-evenness", action="store_true")
    parser.add_argument("--entropy", action="store_true")
    parser.add_argument("--perplexity", action="store_true")

    args = parser.parse_args()

    # Batch directory mode (support optional second directory comparison)
    if args.dir:
        dir_path = os.path.abspath(args.dir)
        txt_files = sorted(glob.glob(os.path.join(dir_path, '*.txt')))

        if not txt_files:
            print(f"No .txt files found in {dir_path}")
            return

        # determine which metrics to compute; default to a reasonable set if none provided
        metric_flags = {
            'repetition': args.repetition,
            'lexical_density': args.lexical_density,
            'pattern_regularity': args.pattern_regularity,
            'sentence_evenness': args.sentence_evenness,
            'tonal_stability': args.tonal_stability,
            'entropy': args.entropy,
            'perplexity': args.perplexity,
            'keyword_rate': bool(args.keywords)
        }

        if not any(metric_flags.values()):
            # default set
            for k in metric_flags:
                metric_flags[k] = True

        all_results_a = []
        metric_names = set()

        for path in txt_files:
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    text = f.read()
            except Exception:
                with open(path, 'r', encoding='latin-1', errors='ignore') as f:
                    text = f.read()

            words = tokenize_words(text)
            sentences = tokenize_sentences(text)

            r = {'file': os.path.basename(path)}

            if metric_flags.get('repetition'):
                r['repetition'] = repetition_rate(words)
                metric_names.add('repetition')

            if metric_flags.get('lexical_density'):
                r['lexical_density'] = lexical_density(words)
                metric_names.add('lexical_density')

            if metric_flags.get('pattern_regularity'):
                r['pattern_regularity'] = pattern_regularity(words)
                metric_names.add('pattern_regularity')

            if metric_flags.get('sentence_evenness'):
                r['sentence_evenness'] = sentence_length_evenness(sentences)
                metric_names.add('sentence_evenness')

            if metric_flags.get('tonal_stability'):
                r['tonal_stability'] = tonal_stability(sentences)
                metric_names.add('tonal_stability')

            if metric_flags.get('entropy'):
                r['entropy'] = token_entropy(words)
                metric_names.add('entropy')

            if metric_flags.get('perplexity'):
                r['perplexity'] = perplexity_approx(words)
                metric_names.add('perplexity')

            if metric_flags.get('keyword_rate'):
                keywords = set(w.lower() for w in (args.keywords or []))
                r['keyword_rate'] = keyword_matching_rate(words, keywords)
                metric_names.add('keyword_rate')

            all_results_a.append(r)

        # If a second directory is provided, compute metrics for it as well
        all_results_b = []
        label_a = args.label1 or os.path.basename(os.path.abspath(args.dir))
        label_b = args.label2 or (os.path.basename(os.path.abspath(args.dir2)) if args.dir2 else None)

        if args.dir2:
            dir2_path = os.path.abspath(args.dir2)
            txt_files_b = sorted(glob.glob(os.path.join(dir2_path, '*.txt')))
            if not txt_files_b:
                print(f"No .txt files found in {dir2_path}")
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

                    r = {'file': os.path.basename(path)}

                    if metric_flags.get('repetition'):
                        r['repetition'] = repetition_rate(words)
                        metric_names.add('repetition')

                    if metric_flags.get('lexical_density'):
                        r['lexical_density'] = lexical_density(words)
                        metric_names.add('lexical_density')

                    if metric_flags.get('pattern_regularity'):
                        r['pattern_regularity'] = pattern_regularity(words)
                        metric_names.add('pattern_regularity')

                    if metric_flags.get('sentence_evenness'):
                        r['sentence_evenness'] = sentence_length_evenness(sentences)
                        metric_names.add('sentence_evenness')

                    if metric_flags.get('tonal_stability'):
                        r['tonal_stability'] = tonal_stability(sentences)
                        metric_names.add('tonal_stability')

                    if metric_flags.get('entropy'):
                        r['entropy'] = token_entropy(words)
                        metric_names.add('entropy')

                    if metric_flags.get('perplexity'):
                        r['perplexity'] = perplexity_approx(words)
                        metric_names.add('perplexity')

                    if metric_flags.get('keyword_rate'):
                        keywords = set(w.lower() for w in (args.keywords or []))
                        r['keyword_rate'] = keyword_matching_rate(words, keywords)
                        metric_names.add('keyword_rate')

                    all_results_b.append(r)

        # ensure plot dir exists
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

        # plot each metric as an overlaid histogram when two dirs provided
        for metric in sorted(metric_names):
            vals_a = [r.get(metric, None) for r in all_results_a]
            numeric_a = [v for v in vals_a if isinstance(v, (int, float))]

            vals_b = [r.get(metric, None) for r in all_results_b] if all_results_b else []
            numeric_b = [v for v in vals_b if isinstance(v, (int, float))]

            plt.figure(figsize=(10, 6))
            if numeric_a and numeric_b:
                n_a, bins, patches = plt.hist(numeric_a, bins='auto', color='C0', alpha=0.6, label=label_a)
                plt.hist(numeric_b, bins=bins, color='C1', alpha=0.6, label=label_b)
                plt.legend()
                out_png = os.path.join(plot_dir, f"{metric}_compare.png")
                plt.title(f"{metric.replace('_', ' ').title()} — {label_a} vs {label_b}")
            else:
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

    # single-file mode
    if not args.file:
        print('Please provide a file path or use --dir to analyze a directory of .txt files.')
        return

    with open(args.file, "r", encoding="utf-8") as f:
        text = f.read()

    words = tokenize_words(text)
    sentences = tokenize_sentences(text)

    results = {}
    output_lines = []

    if args.repetition:
        results["repetition"] = repetition_rate(words)
        output_lines.append(f"Repetition rate: {results['repetition']:.4f}")

    if args.lexical_density:
        results["lexical_density"] = lexical_density(words)
        output_lines.append(f"Lexical density: {results['lexical_density']:.4f}")

    if args.keywords:
        keywords = set(w.lower() for w in args.keywords)
        results["keyword_rate"] = keyword_matching_rate(words, keywords)
        output_lines.append(f"Keyword matching rate: {results['keyword_rate']:.4f}")

    if args.pattern_regularity:
        results["pattern_regularity"] = pattern_regularity(words)
        output_lines.append(f"Pattern regularity: {results['pattern_regularity']:.4f}")

    if args.tonal_stability:
        results["tonal_stability"] = tonal_stability(sentences)
        output_lines.append(f"Tonal stability: {results['tonal_stability']:.4f}")

    if args.sentence_evenness:
        results["sentence_evenness"] = sentence_length_evenness(sentences)
        output_lines.append(f"Sentence length evenness: {results['sentence_evenness']:.4f}")

    if args.entropy:
        results["entropy"] = token_entropy(words)
        output_lines.append(f"Token entropy: {results['entropy']:.4f}")

    if args.perplexity:
        results["perplexity"] = perplexity_approx(words)
        output_lines.append(f"Perplexity (approx): {results['perplexity']:.2f}")

    style_score = composite_style_score(results)
    output_lines.append(f"\nComposite style regularity score: {style_score:.4f}")

    with open(args.output, "a", encoding="utf-8") as out:
        out.write("\n--- Text Analysis ---\n")
        for line in output_lines:
            out.write(line + "\n")


if __name__ == "__main__":
    main()
