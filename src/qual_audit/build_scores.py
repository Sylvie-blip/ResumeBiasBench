"""Iterates all 2,109 corpus pairs, runs the Step 1+2+3 scorer on each, and
writes results/qual_audit/preservation_scores.csv.

Persists per-pair term sets (not just counts) per the spec, so Step 5's
spot-check can show dropped/added terms without re-running the extractor.
"""

import csv
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from category_map import load_id_to_category  # noqa: E402
from preservation import score_pair, THRESHOLDS  # noqa: E402


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def build_all_scores(repo_root: str):
    root = Path(repo_root)
    human_dir = root / "data" / "humanresumes"
    ai_dir = root / "data" / "airesumes"
    sorted_data_dir = root / "data" / "sorted_data"

    id_to_category = load_id_to_category(str(sorted_data_dir))

    human_ids = {p.stem for p in human_dir.glob("*.txt")}
    ai_ids = {p.stem.removesuffix("_AI") for p in ai_dir.glob("*_AI.txt")}

    pair_ids = sorted(human_ids & ai_ids, key=int)
    missing_human = ai_ids - human_ids
    missing_ai = human_ids - ai_ids
    missing_category = [pid for pid in pair_ids if pid not in id_to_category]

    rows = []
    for pid in pair_ids:
        if pid in missing_category:
            continue  # reported separately, excluded rather than silently guessed
        category = id_to_category[pid]
        orig_text = read_text(human_dir / f"{pid}.txt")
        pol_text = read_text(ai_dir / f"{pid}_AI.txt")

        result = score_pair(orig_text, pol_text, category)
        ts = result["term_scores"]
        ns = result["number_scores"]
        flags = result["preserved_flags"]

        rows.append(
            {
                "pair_id": pid,
                "category": category,
                "orig_term_count": len(ts["orig_terms"]),
                "pol_term_count": len(ts["pol_terms"]),
                "retained_count": len(ts["retained"]),
                "dropped_count": len(ts["dropped"]),
                "added_count": len(ts["added"]),
                "dropped_terms": ";".join(sorted(ts["dropped"])),
                "added_terms": ";".join(sorted(ts["added"])),
                "retention_rate": ts["retention_rate"],
                "undefined_retention": ts["undefined_retention"],
                "numbers_added": ns["numbers_added"],
                "numbers_removed": ns["numbers_removed"],
                "numbers_changed": ns["numbers_changed"],
                "placeholder_damaged": result["placeholder_damaged"],
                "preserved_0.90": flags[0.90],
                "preserved_0.95": flags[0.95],
                "preserved_1.00": flags[1.00],
            }
        )

    return {
        "rows": rows,
        "total_pairs": len(pair_ids),
        "missing_human_txt": sorted(missing_human, key=int) if missing_human else [],
        "missing_ai_txt": sorted(missing_ai, key=int) if missing_ai else [],
        "missing_category": missing_category,
    }


def write_scores_csv(rows: list[dict], out_path: str):
    fieldnames = [
        "pair_id", "category",
        "orig_term_count", "pol_term_count", "retained_count", "dropped_count",
        "added_count", "dropped_terms", "added_terms",
        "retention_rate", "undefined_retention",
        "numbers_added", "numbers_removed", "numbers_changed",
        "placeholder_damaged",
        "preserved_0.90", "preserved_0.95", "preserved_1.00",
    ]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


if __name__ == "__main__":
    result = build_all_scores(str(_REPO_ROOT))
    print(f"Total matched pairs: {result['total_pairs']}")
    print(f"Missing human .txt (AI-only stragglers): {len(result['missing_human_txt'])}")
    print(f"Missing AI .txt (human-only stragglers): {len(result['missing_ai_txt'])}")
    print(f"Missing category label: {len(result['missing_category'])}")

    undefined_n = sum(1 for r in result["rows"] if r["undefined_retention"])
    print(f"Pairs with undefined_retention (orig_terms empty): {undefined_n}")

    out_path = _REPO_ROOT / "results" / "qual_audit" / "preservation_scores.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_scores_csv(result["rows"], str(out_path))
    print(f"Wrote {out_path}")
