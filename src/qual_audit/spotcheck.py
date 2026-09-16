"""Step 5: stratified 50-pair manual spot-check sample.

Stratifies across both the retention_rate range (including the
undefined_retention bucket - the scorer's biggest reject reason) and
occupational category, so the check covers pairs the automated scorer
rejected, not just the preserved subset. Fixed seed for reproducibility.
"""

import csv
import random
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from preservation import score_numbers  # noqa: E402

SAMPLE_SIZE = 50
RANDOM_SEED = 20260717  # fixed seed, recorded here for reproducibility

# (label, predicate on retention_rate-as-float-or-None)
BUCKETS = [
    ("undefined", lambda r: r is None),
    ("lt_0.50", lambda r: r is not None and r < 0.50),
    ("0.50_to_0.90", lambda r: r is not None and 0.50 <= r < 0.90),
    ("0.90_to_0.95", lambda r: r is not None and 0.90 <= r < 0.95),
    ("0.95_to_1.00_exclusive", lambda r: r is not None and 0.95 <= r < 1.00),
    ("exactly_1.00", lambda r: r is not None and r >= 1.00),
]


def load_preservation_rows(path: str) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["_retention_rate"] = (
            None if r["retention_rate"] in ("", "None") else float(r["retention_rate"])
        )
    return rows


def bucket_of(row: dict) -> str:
    for label, pred in BUCKETS:
        if pred(row["_retention_rate"]):
            return label
    raise ValueError(f"No bucket matched row {row['pair_id']}")


def allocate_bucket_targets(bucket_sizes: dict[str, int], total: int) -> dict[str, int]:
    """Proportional allocation with a floor of min(3, population) per non-empty
    bucket, so small/rare buckets (like undefined_retention or exactly_1.00)
    aren't drowned out, then fills the remainder proportionally to leftover
    population."""
    targets = {}
    remaining_total = total
    remaining_pop = dict(bucket_sizes)

    for label, size in bucket_sizes.items():
        floor = min(3, size)
        targets[label] = floor
        remaining_total -= floor
        remaining_pop[label] = size - floor

    while remaining_total > 0 and sum(remaining_pop.values()) > 0:
        pool_total = sum(remaining_pop.values())
        progressed = False
        for label in list(remaining_pop):
            if remaining_total <= 0:
                break
            if remaining_pop[label] <= 0:
                continue
            share = max(1, round(remaining_total * remaining_pop[label] / pool_total))
            take = min(share, remaining_pop[label], remaining_total)
            if take > 0:
                targets[label] += take
                remaining_pop[label] -= take
                remaining_total -= take
                progressed = True
        if not progressed:
            break

    return targets


def sample_bucket(rows: list[dict], target_n: int, rng: random.Random) -> list[dict]:
    """Round-robins across categories within the bucket so the sample isn't
    dominated by whichever category happens to be largest."""
    by_category: dict[str, list[dict]] = {}
    for r in rows:
        by_category.setdefault(r["category"], []).append(r)
    for cat_rows in by_category.values():
        rng.shuffle(cat_rows)

    categories = list(by_category.keys())
    rng.shuffle(categories)

    selected = []
    idx = 0
    while len(selected) < target_n and any(by_category.values()):
        cat = categories[idx % len(categories)]
        if by_category[cat]:
            selected.append(by_category[cat].pop())
        idx += 1
        if idx > 10 * (target_n + len(categories)):
            break  # safety valve, shouldn't trigger
    return selected


def build_sample(preservation_rows: list[dict], sample_size: int = SAMPLE_SIZE,
                  seed: int = RANDOM_SEED) -> list[dict]:
    rng = random.Random(seed)

    by_bucket: dict[str, list[dict]] = {label: [] for label, _ in BUCKETS}
    for r in preservation_rows:
        by_bucket[bucket_of(r)].append(r)

    bucket_sizes = {label: len(rows) for label, rows in by_bucket.items()}
    targets = allocate_bucket_targets(bucket_sizes, sample_size)

    sample = []
    for label, _ in BUCKETS:
        sample.extend(sample_bucket(by_bucket[label], targets[label], rng))

    return sample


def enrich_with_number_detail(row: dict, human_dir: Path, ai_dir: Path) -> dict:
    pid = row["pair_id"]
    orig_text = (human_dir / f"{pid}.txt").read_text(encoding="utf-8", errors="replace")
    pol_text = (ai_dir / f"{pid}_AI.txt").read_text(encoding="utf-8", errors="replace")
    ns = score_numbers(orig_text, pol_text)

    return {
        "pair_id": pid,
        "category": row["category"],
        "retention_rate": row["retention_rate"],
        "undefined_retention": row["undefined_retention"],
        "dropped_terms": row["dropped_terms"],
        "added_terms": row["added_terms"],
        "numbers_added_detail": str(ns["added_detail"]),
        "numbers_removed_detail": str(ns["removed_detail"]),
        "numbers_changed_count": ns["numbers_changed"],
        "preserved_0.90": row["preserved_0.90"],
        "preserved_0.95": row["preserved_0.95"],
        "preserved_1.00": row["preserved_1.00"],
        "orig_full_text": orig_text,
        "pol_full_text": pol_text,
        "human_judgment_yes_no": "",
        "human_note": "",
    }


def write_sample_csv(sample_rows: list[dict], out_path: str):
    fieldnames = [
        "pair_id", "category", "retention_rate", "undefined_retention",
        "dropped_terms", "added_terms",
        "numbers_added_detail", "numbers_removed_detail", "numbers_changed_count",
        "preserved_0.90", "preserved_0.95", "preserved_1.00",
        "orig_full_text", "pol_full_text",
        "human_judgment_yes_no", "human_note",
    ]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in sample_rows:
            writer.writerow(row)


if __name__ == "__main__":
    preservation_rows = load_preservation_rows(
        str(_REPO_ROOT / "results" / "qual_audit" / "preservation_scores.csv")
    )
    sample = build_sample(preservation_rows)
    print(f"Sampled {len(sample)} pairs (target {SAMPLE_SIZE}), seed={RANDOM_SEED}")

    from collections import Counter
    print("By bucket:", Counter(bucket_of(r) for r in sample))
    print("Distinct categories covered:", len({r["category"] for r in sample}))

    human_dir = _REPO_ROOT / "data" / "humanresumes"
    ai_dir = _REPO_ROOT / "data" / "airesumes"
    enriched = [enrich_with_number_detail(r, human_dir, ai_dir) for r in sample]

    out_path = _REPO_ROOT / "results" / "qual_audit" / "spotcheck_sample.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_sample_csv(enriched, str(out_path))
    print(f"Wrote {out_path}")
