"""Step 5 (second half): run this AFTER a human has filled in the
human_judgment_yes_no / human_note columns of results/qual_audit/spotcheck_sample.csv.

Computes plain agreement between the human judgment ("does the polished
version contain any substantive qualification change the automated scorer
missed?") and the automated preserved/not-preserved label, and lists every
disagreement. Uses the t=0.90 preserved flag as the automated label being
checked, since that's the most inclusive (least strict) threshold - the one
most likely to disagree with a human "yes, something changed" judgment.
"""

import csv
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
SAMPLE_PATH = _REPO_ROOT / "results" / "qual_audit" / "spotcheck_sample.csv"


def load_annotated(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def compute_agreement(rows: list[dict]) -> dict:
    annotated = [r for r in rows if r["human_judgment_yes_no"].strip()]
    if not annotated:
        return {"annotated_n": 0}

    agree = 0
    disagreements = []
    for r in annotated:
        human_says_changed = r["human_judgment_yes_no"].strip().lower().startswith("y")
        auto_says_preserved = r["preserved_0.90"].strip() == "True"
        # Human "yes, changed" should correspond to auto "not preserved"; a
        # match means human_says_changed != auto_says_preserved.
        matches = human_says_changed != auto_says_preserved
        if matches:
            agree += 1
        else:
            disagreements.append(
                {
                    "pair_id": r["pair_id"],
                    "category": r["category"],
                    "retention_rate": r["retention_rate"],
                    "human_judgment": r["human_judgment_yes_no"],
                    "human_note": r["human_note"],
                    "auto_preserved_0.90": r["preserved_0.90"],
                }
            )

    return {
        "annotated_n": len(annotated),
        "agree_n": agree,
        "agreement_rate": agree / len(annotated),
        "disagreements": disagreements,
    }


if __name__ == "__main__":
    rows = load_annotated(SAMPLE_PATH)
    result = compute_agreement(rows)
    if result["annotated_n"] == 0:
        print(
            f"No annotations found yet in {SAMPLE_PATH}. "
            "Fill in human_judgment_yes_no (yes/no) and human_note, then re-run."
        )
        sys.exit(0)

    print(f"Annotated: {result['annotated_n']} / {len(rows)}")
    print(f"Agreement: {result['agree_n']}/{result['annotated_n']} "
          f"({result['agreement_rate']:.1%})")
    if result["disagreements"]:
        print("\nDisagreements:")
        for d in result["disagreements"]:
            print(f"  pair {d['pair_id']} ({d['category']}): human={d['human_judgment']!r} "
                  f"auto_preserved_0.90={d['auto_preserved_0.90']} note={d['human_note']!r}")
