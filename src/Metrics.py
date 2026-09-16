#!/usr/bin/env python3
"""Unified Metrics script

Features updated per request:
- tokenization: Hugging Face GPT-2 tokenizer when available (fallback to regex)
- perplexity: compute model-based perplexity using GPT-2 log-probabilities when available
- lexical density: uses spaCy POS tags when available
- repetition rate: type-token ratio (TTR)
- sentence evenness: replaced by coefficient of variation (std/mean)
- tonal stability: VADER sentiment analyzer when available (fallback lexicon)

The script falls back gracefully when optional libraries are not installed.
"""

# python3 /Users/sylviadong/Documents/Resume_Bias/Bias_code/Metrics.py --dir /Users/sylviadong/Documents/humanresumes --dir2 /Users/sylviadong/Documents/airesumes --label1 Human --label2 AI --plot-dir /Users/sylviadong/Documents/new_metrics_results

import argparse
import os
import re
import glob
import math
import csv
from collections import Counter
from statistics import mean, stdev, variance

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Optional heavy deps: transformers/torch, spacy, vader
_HF_AVAILABLE = False
_TORCH_AVAILABLE = False
_SPACY_AVAILABLE = False
_VADER_AVAILABLE = False
_TEXTBLOB_AVAILABLE = False
HF_TOKENIZER = None
HF_MODEL = None
HF_DEVICE = None
HF_MAX_LEN = 1024

try:
    import torch
    _TORCH_AVAILABLE = True
except Exception:
    _TORCH_AVAILABLE = False

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    HF_TOKENIZER = AutoTokenizer.from_pretrained("gpt2")
    HF_MAX_LEN = getattr(HF_TOKENIZER, 'model_max_length', 2048)
    _HF_AVAILABLE = True
except Exception:
    HF_TOKENIZER = None
    _HF_AVAILABLE = False

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
    # Try NLTK's VADER first (downloads lexicon if missing), else fallback to vaderSentiment
    try:
        from nltk.sentiment.vader import SentimentIntensityAnalyzer
        import nltk
        try:
            nltk.data.find('sentiment/vader_lexicon.zip')
        except Exception:
            try:
                nltk.download('vader_lexicon')
            except Exception:
                pass
        _VADER = SentimentIntensityAnalyzer()
        _VADER_AVAILABLE = True
    except Exception:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        _VADER = SentimentIntensityAnalyzer()
        _VADER_AVAILABLE = True
except Exception:
    _VADER = None
    _VADER_AVAILABLE = False

try:
    from textblob import TextBlob
    _TEXTBLOB_AVAILABLE = True
except Exception:
    _TEXTBLOB_AVAILABLE = False


# ------------------ Tokenization ------------------
def tokenize_words(text):
    """Return list of tokens. Prefer GPT-2 tokenizer (subword tokens) when available.
    Falls back to simple alphabetic word regex.
    """
    if _HF_AVAILABLE and HF_TOKENIZER is not None:
        try:
            return HF_TOKENIZER.tokenize(text)
        except Exception:
            pass
    return re.findall(r"\b[a-zA-Z']+\b", text.lower())


def tokenize_sentences(text):
    return [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]


# ------------------ Metrics ------------------

def repetition_rate(tokens_or_text):
    """Type-token ratio (TTR): unique types / total tokens."""
    if isinstance(tokens_or_text, str):
        toks = tokenize_words(tokens_or_text)
    else:
        toks = list(tokens_or_text)
    if not toks:
        return 0.0
    return len(set(toks)) / len(toks)


def lexical_density(tokens_or_text):
    """Lexical density using spaCy POS tags when available.
    Returns unique content-word types / total alphabetic tokens.
    Content POS: NOUN, PROPN, VERB, ADJ, ADV.
    """
    if isinstance(tokens_or_text, list):
        text = " ".join(tokens_or_text)
    else:
        text = str(tokens_or_text)

    if _SPACY_AVAILABLE and _NLP is not None:
        doc = _NLP(text)
        content_pos = {"NOUN", "PROPN", "VERB", "ADJ", "ADV"}
        content = [token.lemma_.lower() for token in doc if token.pos_ in content_pos and token.is_alpha]
        total_alpha = sum(1 for token in doc if token.is_alpha)
        return (len(set(content)) / total_alpha) if total_alpha else 0.0
    else:
        # fallback: simple stopword-based unique content types / total tokens
        stopwords = {
            "the", "is", "am", "are", "was", "were", "be", "been", "being",
            "a", "an", "and", "or", "but", "if", "then", "so",
            "of", "in", "on", "at", "to", "for", "with", "by", "from"
        }
        toks = tokenize_words(text)
        content = [w for w in toks if w.lower() not in stopwords]
        return (len(set(content)) / len(toks)) if toks else 0.0


def pattern_regularity(tokens, n=3):
    if len(tokens) < n:
        return 0.0
    ngrams = Counter(zip(*[tokens[i:] for i in range(n)]))
    repeated = sum(c for c in ngrams.values() if c > 1)
    total = sum(ngrams.values())
    return repeated / total if total else 0.0


def sentence_length_cv(sentences):
    """Coefficient of variation (std / mean) of sentence lengths (alpha tokens)."""
    lengths = []
    for s in sentences:
        if _SPACY_AVAILABLE and _NLP is not None:
            doc = _NLP(s)
            L = sum(1 for t in doc if t.is_alpha)
        else:
            L = len(re.findall(r"\b[a-zA-Z']+\b", s))
        lengths.append(L)
    if len(lengths) < 2:
        return 0.0
    mu = mean(lengths)
    if mu == 0:
        return 0.0
    sd = stdev(lengths)
    return sd / mu


# ------------------ Sentiment / tonal stability ------------------
def tonal_stability(sentences):
    """Compute tonal stability using TextBlob when available, else VADER, else lexicon.
    Returns 1 / (1 + variance(sentiment_scores)).
    - TextBlob: uses sentence polarity in [-1,1]
    - VADER: uses compound score in [-1,1]
    - Lexicon fallback: integer score per sentence (pos - neg counts)
    """
    scores = []

    # Prefer TextBlob if available
    if _TEXTBLOB_AVAILABLE:
        for s in sentences:
            s_text = s.strip()
            if not s_text:
                continue
            try:
                tb = TextBlob(s_text)
                # polarity in [-1.0, 1.0]
                scores.append(float(tb.sentiment.polarity))
            except Exception:
                scores.append(0.0)
    elif _VADER_AVAILABLE and _VADER is not None:
        for s in sentences:
            s_text = s.strip()
            if not s_text:
                continue
            try:
                out = _VADER.polarity_scores(s_text)
                scores.append(out.get('compound', 0.0))
            except Exception:
                scores.append(0.0)
    else:
        positive = {"good", "great", "positive", "benefit", "success", "effective", "improve", "achieve", "lead", "led", "managed"}
        negative = {"bad", "poor", "negative", "fail", "harm", "ineffective", "decline"}
        for s in sentences:
            toks = re.findall(r"\b[a-zA-Z']+\b", s.lower())
            score = sum(w in positive for w in toks) - sum(w in negative for w in toks)
            scores.append(float(score))

    if len(scores) < 2:
        return 1.0
    try:
        return 1.0 / (1.0 + variance(scores))
    except Exception:
        # fallback if variance computation fails
        return 0.0


# ------------------ Entropy / perplexity ------------------
def word_distribution(tokens):
    counts = Counter(tokens)
    total = len(tokens)
    return {w: c / total for w, c in counts.items()} if total else {}


def token_entropy(tokens):
    if not tokens:
        return 0.0
    dist = word_distribution(tokens)
    return -sum(p * math.log2(p) for p in dist.values())


def _compute_gpt2_perplexity(text):
    """Compute perplexity using GPT-2 (chunked). Returns exp(avg_nll).
    Requires HF_MODEL and torch to be available.
    """
    # We'll compute token-level negative log likelihoods from model logits using a sliding window.
    # This follows the HF perplexity example pattern: for long texts we process windows with overlap
    # and only count NLLs for the new tokens in each window to avoid double-counting.
    global HF_MODEL, HF_DEVICE
    if HF_TOKENIZER is None:
        raise RuntimeError("GPT-2 tokenizer not available")

    # lazy-load model if needed
    if HF_MODEL is None:
        HF_MODEL = AutoModelForCausalLM.from_pretrained("gpt2")
        if _TORCH_AVAILABLE:
            HF_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            HF_MODEL.to(HF_DEVICE)

    enc = HF_TOKENIZER.encode(text, add_special_tokens=False)
    n = len(enc)
    if n == 0:
        return float('nan')

    max_len = HF_MAX_LEN or 1024
    # choose a stride so windows overlap; standard is max_len - overlap
    overlap = 50 if max_len > 50 else 0
    stride = max_len - overlap

    import torch as _torch
    from torch.nn.functional import log_softmax

    nlls = []
    try:
        for start_idx in range(0, n, stride):
            end_idx = min(start_idx + max_len, n)
            window_ids = enc[start_idx:end_idx]
            if not window_ids:
                continue
            input_ids = _torch.tensor([window_ids], dtype=_torch.long)
            if _TORCH_AVAILABLE and HF_DEVICE is not None:
                input_ids = input_ids.to(HF_DEVICE)

            with _torch.no_grad():
                outputs = HF_MODEL(input_ids)
                logits = outputs.logits  # shape (1, seq_len, vocab_size)
                # shift logits and labels for next-token prediction
                shift_logits = logits[:, :-1, :]
                shift_labels = input_ids[:, 1:]

                log_probs = log_softmax(shift_logits, dim=-1)

                # determine which token positions in this window are "new" (not overlapped by previous window)
                # The first window covers [0, end_idx). For start_idx>0 we should only count positions from
                # (start_idx - previous_start) up to end_idx. Simpler: compute token indices relative to window
                # corresponding to positions max(1, overlap_start) .. seq_len-1
                # relative_new_start is 1 when start_idx==0, else (overlap)
                if start_idx == 0:
                    rel_start = 0
                else:
                    rel_start = overlap

                # labels available at positions 0..seq_len-2 correspond to prediction for tokens 1..seq_len-1
                seq_len = shift_labels.size(1)
                # compute indices to take: from rel_start to seq_len-1 (inclusive)
                take_start = rel_start
                take_end = seq_len  # exclusive

                if take_start >= take_end:
                    continue

                # gather log-probs for the target token ids for the desired positions
                # shift_labels[0, i] is the token id at position i+1 in the window
                target_labels = shift_labels[0, take_start:take_end]
                target_log_probs = log_probs[0, take_start:take_end, :].gather(1, target_labels.unsqueeze(1)).squeeze(1)
                # negative log likelihoods
                nll_tensor = -target_log_probs
                # move to cpu and extend list
                nlls.extend(nll_tensor.cpu().tolist())

        if not nlls:
            return float('nan')
        avg_nll = float(sum(nlls)) / len(nlls)
        return math.exp(avg_nll)
    except Exception:
        # fallback to the simpler loss-based method if anything goes wrong
        try:
            total_loss = 0.0
            total_toks = 0
            for i in range(0, n, max_len):
                chunk = enc[i:i+max_len]
                input_ids = _torch.tensor([chunk], dtype=_torch.long)
                if _TORCH_AVAILABLE and HF_DEVICE is not None:
                    input_ids = input_ids.to(HF_DEVICE)
                with _torch.no_grad():
                    outputs = HF_MODEL(input_ids, labels=input_ids)
                    loss = outputs.loss.item()
                    toks = len(chunk)
                    total_loss += loss * toks
                    total_toks += toks
            avg_loss = total_loss / total_toks if total_toks else float('nan')
            return math.exp(avg_loss)
        except Exception:
            return 2 ** token_entropy(tokenize_words(text))


def perplexity_model_or_entropy(tokens_or_text):
    """Compute perplexity using GPT-2 model when available, else 2**entropy on token distribution.
    Accepts list of tokens or raw text.
    """
    if isinstance(tokens_or_text, list):
        text = ' '.join(tokens_or_text)
        tokens = tokens_or_text
    else:
        text = str(tokens_or_text)
        tokens = tokenize_words(text)

    if _HF_AVAILABLE and HF_TOKENIZER is not None and _TORCH_AVAILABLE:
        try:
            return _compute_gpt2_perplexity(text)
        except Exception:
            return 2 ** token_entropy(tokens)
    else:
        return 2 ** token_entropy(tokens)


# ------------------ Burstiness / rolling metrics ------------------
def rolling_entropy(tokens, window=50):
    if len(tokens) < window:
        return 0.0
    entropies = []
    for i in range(len(tokens) - window + 1):
        chunk = tokens[i:i + window]
        entropies.append(token_entropy(chunk))
    return variance(entropies) if len(entropies) > 1 else 0.0


def sentence_entropy_variance(sentences):
    entropies = []
    for s in sentences:
        toks = tokenize_words(s)
        if toks:
            entropies.append(token_entropy(toks))
    return variance(entropies) if len(entropies) > 1 else 0.0


# Backwards-compatible aliases for older metric names used elsewhere in the repo
def keyword_matching_rate(words, keywords):
    if not keywords:
        return 0.0
    kws = set(k.lower() for k in keywords)
    return sum(1 for w in words if w.lower() in kws) / len(words) if words else 0.0


def sentence_length_evenness(sentences):
    return sentence_length_cv(sentences)


def perplexity_approx(tokens_or_text):
    return perplexity_model_or_entropy(tokens_or_text)


def cross_entropy(words_p, words_q, epsilon=1e-12):
    p = word_distribution(words_p)
    q = word_distribution(words_q)
    return -sum(p[w] * math.log2(q.get(w, epsilon)) for w in p)


def kl_divergence(words_p, words_q, epsilon=1e-12):
    p = word_distribution(words_p)
    q = word_distribution(words_q)
    return sum(p[w] * math.log2(p[w] / q.get(w, epsilon)) for w in p)


# ------------------ I/O / Main ------------------
DEFAULT_PLOT_DIR = "metric_plots"


def write_single_csv(output_csv, file_path, metrics):
    fieldnames = ["file"] + sorted(metrics.keys())
    write_header = not os.path.exists(output_csv)
    with open(output_csv, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        row = {k: metrics.get(k, '') for k in fieldnames}
        row['file'] = os.path.basename(file_path)
        writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(description="Unified Metrics tool")
    parser.add_argument('file1', nargs='?', help='input file')
    parser.add_argument('file2', nargs='?', help='optional second file')
    parser.add_argument('--dir', help='directory of .txt files')
    parser.add_argument('--dir2', help='second directory to compare')
    parser.add_argument('--label1', help='Label for first directory in comparison plots (optional)')
    parser.add_argument('--label2', help='Label for second directory in comparison plots (optional)')
    parser.add_argument('--plot-dir', default=DEFAULT_PLOT_DIR)
    parser.add_argument('--repetition', action='store_true')
    parser.add_argument('--lexical-density', action='store_true')
    parser.add_argument('--pattern-regularity', action='store_true')
    parser.add_argument('--entropy', action='store_true')
    parser.add_argument('--perplexity', action='store_true')
    parser.add_argument('--rolling-entropy', action='store_true')
    parser.add_argument('--sentence-entropy-variance', action='store_true')
    parser.add_argument('--sentence-evenness', action='store_true')
    parser.add_argument('--tonal-stability', action='store_true')
    parser.add_argument('--keywords', nargs='*')
    parser.add_argument('--out-csv', default='Metrics_Output.csv')

    args = parser.parse_args()

    metric_flags = {
        'repetition': args.repetition,
        'lexical_density': args.lexical_density,
        'pattern_regularity': args.pattern_regularity,
        'sentence_evenness': args.sentence_evenness,
        'tonal_stability': args.tonal_stability,
        'entropy': args.entropy,
        'perplexity': args.perplexity,
        'rolling_entropy': args.rolling_entropy,
        'sentence_entropy_variance': args.sentence_entropy_variance,
        'keyword_rate': bool(args.keywords)
    }
    if not any(metric_flags.values()):
        for k in metric_flags:
            metric_flags[k] = True

    # directory mode
    if args.dir:
        dir_path = os.path.abspath(args.dir)
        files = sorted(glob.glob(os.path.join(dir_path, '*.txt')))
        if not files:
            print('no txt files')
            return
        all_a = []
        all_b = []
        metric_names = set()
        for p in files:
            try:
                with open(p, 'r', encoding='utf-8') as f:
                    text = f.read()
            except Exception:
                with open(p, 'r', encoding='latin-1', errors='ignore') as f:
                    text = f.read()
            toks = tokenize_words(text)
            sents = tokenize_sentences(text)
            r = {'file': os.path.basename(p)}
            if metric_flags['repetition']:
                r['repetition'] = repetition_rate(toks)
                metric_names.add('repetition')
            if metric_flags['lexical_density']:
                r['lexical_density'] = lexical_density(text)
                metric_names.add('lexical_density')
            if metric_flags['pattern_regularity']:
                r['pattern_regularity'] = pattern_regularity(toks)
                metric_names.add('pattern_regularity')
            if metric_flags['sentence_evenness']:
                r['sentence_evenness'] = sentence_length_cv(sents) if False else sentence_length_cv(sents)
                metric_names.add('sentence_evenness')
            if metric_flags['tonal_stability']:
                r['tonal_stability'] = tonal_stability(sents)
                metric_names.add('tonal_stability')
            if metric_flags['entropy']:
                r['entropy'] = token_entropy(toks)
                metric_names.add('entropy')
            if metric_flags['perplexity']:
                r['perplexity'] = perplexity_model_or_entropy(toks)
                metric_names.add('perplexity')
            if metric_flags['rolling_entropy']:
                r['rolling_entropy'] = rolling_entropy(toks)
                metric_names.add('rolling_entropy')
            if metric_flags['sentence_entropy_variance']:
                r['sentence_entropy_variance'] = sentence_entropy_variance(sents)
                metric_names.add('sentence_entropy_variance')
            if metric_flags['keyword_rate']:
                kws = set(w.lower() for w in (args.keywords or []))
                r['keyword_rate'] = sum(1 for w in toks if w.lower() in kws) / len(toks) if toks else 0
                metric_names.add('keyword_rate')
            all_a.append(r)

        # optional dir2
        if args.dir2:
            files_b = sorted(glob.glob(os.path.join(os.path.abspath(args.dir2), '*.txt')))
            for p in files_b:
                try:
                    with open(p, 'r', encoding='utf-8') as f:
                        text = f.read()
                except Exception:
                    with open(p, 'r', encoding='latin-1', errors='ignore') as f:
                        text = f.read()
                toks = tokenize_words(text)
                sents = tokenize_sentences(text)
                r = {'file': os.path.basename(p)}
                if metric_flags['repetition']:
                    r['repetition'] = repetition_rate(toks)
                    metric_names.add('repetition')
                if metric_flags['lexical_density']:
                    r['lexical_density'] = lexical_density(text)
                    metric_names.add('lexical_density')
                if metric_flags['pattern_regularity']:
                    r['pattern_regularity'] = pattern_regularity(toks)
                    metric_names.add('pattern_regularity')
                if metric_flags['sentence_evenness']:
                    r['sentence_evenness'] = sentence_length_cv(sents)
                    metric_names.add('sentence_evenness')
                if metric_flags['tonal_stability']:
                    r['tonal_stability'] = tonal_stability(sents)
                    metric_names.add('tonal_stability')
                if metric_flags['entropy']:
                    r['entropy'] = token_entropy(toks)
                    metric_names.add('entropy')
                if metric_flags['perplexity']:
                    r['perplexity'] = perplexity_model_or_entropy(toks)
                    metric_names.add('perplexity')
                if metric_flags['rolling_entropy']:
                    r['rolling_entropy'] = rolling_entropy(toks)
                    metric_names.add('rolling_entropy')
                if metric_flags['sentence_entropy_variance']:
                    r['sentence_entropy_variance'] = sentence_entropy_variance(sents)
                    metric_names.add('sentence_entropy_variance')
                if metric_flags['keyword_rate']:
                    kws = set(w.lower() for w in (args.keywords or []))
                    r['keyword_rate'] = sum(1 for w in toks if w.lower() in kws) / len(toks) if toks else 0
                    metric_names.add('keyword_rate')
                all_b.append(r)

        # write CSVs and plots
        plot_dir = args.plot_dir
        os.makedirs(plot_dir, exist_ok=True)
        combined_csv = os.path.join(plot_dir, 'metric_scores_combined.csv')
        fieldnames = ['group', 'file'] + sorted(metric_names)
        with open(combined_csv, 'w', newline='', encoding='utf-8') as outf:
            writer = csv.DictWriter(outf, fieldnames=fieldnames)
            writer.writeheader()
            for r in all_a:
                row = {'group': os.path.basename(os.path.abspath(args.dir)), 'file': r['file']}
                for m in metric_names:
                    row[m] = r.get(m, '')
                writer.writerow(row)
            if all_b:
                for r in all_b:
                    row = {'group': os.path.basename(os.path.abspath(args.dir2)), 'file': r['file']}
                    for m in metric_names:
                        row[m] = r.get(m, '')
                    writer.writerow(row)
        print(f'Wrote combined CSV to {combined_csv}')

        # per-metric plots (overlay if both)
        for metric in sorted(metric_names):
            vals_a = [r.get(metric) for r in all_a]
            num_a = [v for v in vals_a if isinstance(v, (int, float))]
            vals_b = [r.get(metric) for r in all_b] if all_b else []
            num_b = [v for v in vals_b if isinstance(v, (int, float))]
            plt.figure(figsize=(8,5))
            if num_a and num_b:
                n, bins, _ = plt.hist(num_a, bins='auto', alpha=0.6, label='A')
                plt.hist(num_b, bins=bins, alpha=0.6, label='B')
                plt.legend()
            elif num_a:
                plt.hist(num_a, bins='auto', alpha=0.8)
            elif num_b:
                plt.hist(num_b, bins='auto', alpha=0.8)
            else:
                continue
            out_png = os.path.join(plot_dir, f'{metric}.png')
            plt.title(metric)
            plt.tight_layout()
            plt.savefig(out_png)
            plt.close()

        return

    # single-file mode
    if not args.file1:
        print('Provide --dir or a file path')
        return

    # read file1
    try:
        with open(args.file1, 'r', encoding='utf-8') as f:
            text1 = f.read()
    except Exception:
        with open(args.file1, 'r', encoding='latin-1', errors='ignore') as f:
            text1 = f.read()

    toks1 = tokenize_words(text1)
    sents1 = tokenize_sentences(text1)
    results = {}
    if metric_flags['repetition']:
        results['repetition'] = repetition_rate(toks1)
    if metric_flags['lexical_density']:
        results['lexical_density'] = lexical_density(text1)
    if metric_flags['pattern_regularity']:
        results['pattern_regularity'] = pattern_regularity(toks1)
    if metric_flags['sentence_evenness']:
        results['sentence_evenness'] = sentence_length_cv(sents1)
    if metric_flags['tonal_stability']:
        results['tonal_stability'] = tonal_stability(sents1)
    if metric_flags['entropy']:
        results['entropy'] = token_entropy(toks1)
    if metric_flags['perplexity']:
        results['perplexity'] = perplexity_model_or_entropy(toks1)
    if metric_flags['rolling_entropy']:
        results['rolling_entropy'] = rolling_entropy(toks1)
    if metric_flags['sentence_entropy_variance']:
        results['sentence_entropy_variance'] = sentence_entropy_variance(sents1)
    if metric_flags['keyword_rate']:
        kws = set(w.lower() for w in (args.keywords or []))
        results['keyword_rate'] = sum(1 for w in toks1 if w.lower() in kws) / len(toks1) if toks1 else 0

        write_single_csv(args.out_csv, args.file1, results)
        print(f'Wrote single-file metrics to {args.out_csv}')


    if __name__ == '__main__':
        main()


if __name__ == '__main__':
    main()



def write_single_csv(output_csv, file_path, metrics):
    # write a CSV with header and one row
    fieldnames = ["file"] + sorted(metrics.keys())
    write_header = not os.path.exists(output_csv)
    with open(output_csv, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        row = {k: metrics.get(k, '') for k in fieldnames}
        row['file'] = os.path.basename(file_path)
        writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(description="Unified text metrics tool")
    parser.add_argument("file1", nargs='?', help="First input file (or omit when using --dir)")
    parser.add_argument("file2", nargs='?', help="Optional second file for pairwise comparison")
    parser.add_argument("--dir", help="Directory containing .txt files to batch-analyze")
    parser.add_argument("--dir2", help="Second directory for comparison")
    parser.add_argument("--label1", help="Label for first directory in comparison plots (optional)")
    parser.add_argument("--label2", help="Label for second directory in comparison plots (optional)")
    parser.add_argument("--plot-dir", default=DEFAULT_PLOT_DIR, help="Directory where metric plots will be saved when using --dir")

    # metric toggles
    parser.add_argument("--repetition", action="store_true")
    parser.add_argument("--lexical-density", action="store_true")
    parser.add_argument("--pattern-regularity", action="store_true")
    parser.add_argument("--entropy", action="store_true")
    parser.add_argument("--perplexity", action="store_true")
    parser.add_argument("--rolling-entropy", action="store_true")
    parser.add_argument("--sentence-entropy-variance", action="store_true")
    parser.add_argument("--sentence-evenness", action="store_true")
    parser.add_argument("--tonal-stability", action="store_true")
    parser.add_argument("--keywords", nargs='*')

    parser.add_argument("--out-csv", help="CSV file to write single-file results to (single-file mode)", default="Metrics_Output.csv")

    args = parser.parse_args()

    # Decide which metrics to compute. Default to a sensible set if none selected
    metric_flags = {
        'repetition': args.repetition,
        'lexical_density': args.lexical_density,
        'pattern_regularity': args.pattern_regularity,
        'sentence_evenness': args.sentence_evenness,
        'tonal_stability': args.tonal_stability,
        'entropy': args.entropy,
        'perplexity': args.perplexity,
        'rolling_entropy': args.rolling_entropy,
        'sentence_entropy_variance': args.sentence_entropy_variance,
        'keyword_rate': bool(args.keywords)
    }

    if not any(metric_flags.values()):
        # default set
        for k in metric_flags:
            metric_flags[k] = True

    # Directory / batch mode
    if args.dir:
        dir_path = os.path.abspath(args.dir)
        txt_files = sorted(glob.glob(os.path.join(dir_path, "*.txt")))
        if not txt_files:
            print(f"No .txt files found in {dir_path}")
            return

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

            res = {'file': os.path.basename(path)}

            if metric_flags.get('repetition'):
                res['repetition'] = repetition_rate(words)
                metric_names.add('repetition')
            if metric_flags.get('lexical_density'):
                res['lexical_density'] = lexical_density(words)
                metric_names.add('lexical_density')
            if metric_flags.get('pattern_regularity'):
                res['pattern_regularity'] = pattern_regularity(words)
                metric_names.add('pattern_regularity')
            if metric_flags.get('sentence_evenness'):
                res['sentence_evenness'] = sentence_length_evenness(sentences)
                metric_names.add('sentence_evenness')
            if metric_flags.get('tonal_stability'):
                res['tonal_stability'] = tonal_stability(sentences)
                metric_names.add('tonal_stability')
            if metric_flags.get('entropy'):
                res['entropy'] = token_entropy(words)
                metric_names.add('entropy')
            if metric_flags.get('perplexity'):
                res['perplexity'] = perplexity_approx(words)
                metric_names.add('perplexity')
            if metric_flags.get('rolling_entropy'):
                res['rolling_entropy'] = rolling_entropy(words)
                metric_names.add('rolling_entropy')
            if metric_flags.get('sentence_entropy_variance'):
                res['sentence_entropy_variance'] = sentence_entropy_variance(sentences)
                metric_names.add('sentence_entropy_variance')
            if metric_flags.get('keyword_rate'):
                keywords = set(w.lower() for w in (args.keywords or []))
                res['keyword_rate'] = keyword_matching_rate(words, keywords)
                metric_names.add('keyword_rate')

            all_results_a.append(res)

        # optional second directory
        all_results_b = []
        label_a = args.label1 or os.path.basename(os.path.abspath(args.dir))
        label_b = args.label2 or (os.path.basename(os.path.abspath(args.dir2)) if args.dir2 else None)
        if args.dir2:
            dir2_path = os.path.abspath(args.dir2)
            txt_files_b = sorted(glob.glob(os.path.join(dir2_path, "*.txt")))
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

                    res = {'file': os.path.basename(path)}
                    if metric_flags.get('repetition'):
                        res['repetition'] = repetition_rate(words)
                        metric_names.add('repetition')
                    if metric_flags.get('lexical_density'):
                        res['lexical_density'] = lexical_density(words)
                        metric_names.add('lexical_density')
                    if metric_flags.get('pattern_regularity'):
                        res['pattern_regularity'] = pattern_regularity(words)
                        metric_names.add('pattern_regularity')
                    if metric_flags.get('sentence_evenness'):
                        res['sentence_evenness'] = sentence_length_evenness(sentences)
                        metric_names.add('sentence_evenness')
                    if metric_flags.get('tonal_stability'):
                        res['tonal_stability'] = tonal_stability(sentences)
                        metric_names.add('tonal_stability')
                    if metric_flags.get('entropy'):
                        res['entropy'] = token_entropy(words)
                        metric_names.add('entropy')
                    if metric_flags.get('perplexity'):
                        res['perplexity'] = perplexity_approx(words)
                        metric_names.add('perplexity')
                    if metric_flags.get('rolling_entropy'):
                        res['rolling_entropy'] = rolling_entropy(words)
                        metric_names.add('rolling_entropy')
                    if metric_flags.get('sentence_entropy_variance'):
                        res['sentence_entropy_variance'] = sentence_entropy_variance(sentences)
                        metric_names.add('sentence_entropy_variance')
                    if metric_flags.get('keyword_rate'):
                        keywords = set(w.lower() for w in (args.keywords or []))
                        res['keyword_rate'] = keyword_matching_rate(words, keywords)
                        metric_names.add('keyword_rate')

                    all_results_b.append(res)

        # ensure plot dir exists
        plot_dir = args.plot_dir
        os.makedirs(plot_dir, exist_ok=True)

        # Write combined CSV and per-dir CSVs
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

        # per-dir CSVs
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

        # plotting per metric
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

    # single-file or pairwise mode
    if not args.file1:
        print('Please provide a file path or use --dir to analyze a directory of .txt files.')
        return

    # read file1
    try:
        with open(args.file1, 'r', encoding='utf-8') as f:
            text1 = f.read()
    except Exception:
        with open(args.file1, 'r', encoding='latin-1', errors='ignore') as f:
            text1 = f.read()

    words1 = tokenize_words(text1)
    sentences1 = tokenize_sentences(text1)

    results = {}

    if metric_flags.get('repetition'):
        results['repetition'] = repetition_rate(words1)
    if metric_flags.get('lexical_density'):
        results['lexical_density'] = lexical_density(words1)
    if metric_flags.get('pattern_regularity'):
        results['pattern_regularity'] = pattern_regularity(words1)
    if metric_flags.get('sentence_evenness'):
        results['sentence_evenness'] = sentence_length_evenness(sentences1)
    if metric_flags.get('tonal_stability'):
        results['tonal_stability'] = tonal_stability(sentences1)
    if metric_flags.get('entropy'):
        results['entropy'] = token_entropy(words1)
    if metric_flags.get('perplexity'):
        results['perplexity'] = perplexity_approx(words1)
    if metric_flags.get('rolling_entropy'):
        results['rolling_entropy'] = rolling_entropy(words1)
    if metric_flags.get('sentence_entropy_variance'):
        results['sentence_entropy_variance'] = sentence_entropy_variance(sentences1)
    if metric_flags.get('keyword_rate'):
        keywords = set(w.lower() for w in (args.keywords or []))
        results['keyword_rate'] = keyword_matching_rate(words1, keywords)

    # if file2 provided, compute pairwise metrics
    if args.file2:
        try:
            with open(args.file2, 'r', encoding='utf-8') as f:
                text2 = f.read()
        except Exception:
            with open(args.file2, 'r', encoding='latin-1', errors='ignore') as f:
                text2 = f.read()

        words2 = tokenize_words(text2)
        # pairwise metrics
        if metric_flags.get('entropy'):
            results['cross_entropy_1_given_2'] = cross_entropy(words1, words2)
            results['kl_divergence_1_vs_2'] = kl_divergence(words1, words2)

    # For single-file mode, write a single-row CSV (user requested)
    write_single_csv(args.out_csv, args.file1, results)
    print(f"Wrote single-file metrics to {args.out_csv}")


if __name__ == '__main__':
    main()
