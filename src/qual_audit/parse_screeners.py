"""Normalize the raw LLM-screening result files into one
{pair_id, model, winner} table.

All 7 currently-available files use the run_screening.py output format:

    ============================================================
    PAIR 10176815
    WINNER: HUMAN:10176815
    SCORE: T=10 A=9 L=6 E=9 TOTAL=34

7 of the paper's 8 models now have a complete (2,109-pair) file at the repo
root; only granite4:3b remains unavailable. History, for anyone re-deriving
this: earlier sessions only had 2 stray partial/complete copies under
Useless/ (results_gemma3_4b.txt at 1,385/2,109 pairs, results_mistral-
small3.2_latest.txt complete) plus an ambiguous-variant qwen2.5.txt in a
different, older script's format (826/2,109 pairs, no size variant recorded).
Those are now superseded and intentionally excluded here:
- The old partial gemma3:4b run disagreed with the new complete run on
  overlapping pair IDs (e.g. pair 10176815: HUMAN in the old partial run, AI
  in the new complete one) - Ollama's local inference isn't perfectly
  deterministic run-to-run even at temperature=0, so partial and complete
  runs of the "same" model cannot be merged as if they were one run without
  silently mixing two different judgments per pair. The complete run is used
  in full; the partial one is dropped rather than reconciled.
- The old root mistral-small3.2:latest file was byte-identical to the
  Useless/ copy, so no conflict there - just redundant.
- qwen2.5.txt's ambiguous variant is now moot: results_qwen2.53b.txt gives an
  unambiguous, complete qwen2.5:3b run.

These 7 filenames also exactly match the set referenced (but not previously
present) in data/human_win_tally.csv, which is a good independent sanity
check that these are the original screening outputs, not a different run.
"""

import csv
import re
from pathlib import Path

_RUN_SCREENING_PAIR_RX = re.compile(
    r"^PAIR (\d+)\s*\nWINNER:\s*(AI|HUMAN):\d+", re.MULTILINE
)

# All 7 currently-available models with an unambiguous identity, and a
# complete (2,109-pair) result file at the repo root. Dropping in
# granite4:3b later (if ever obtained) only requires adding one more entry
# here - no other code changes.
KNOWN_MODELS = {
    "qwen2.5:3b": "results_qwen2.53b.txt",
    "qwen2.5:14b": "results_qwen2.5_14b.txt",
    "qwen2.5:32b": "results_qwen2.5_32b.txt",
    "gemma3:4b": "results_gemma3_4b.txt",
    "gemma3:12b": "results_gemma3_12b.txt",
    "mistral-small3.2:latest": "results_mistral-small3.2_latest.txt",
    "cogito:3b": "results_cogito_3b.txt",
}


def parse_run_screening_file(path: Path, model: str):
    text = path.read_text(encoding="utf-8", errors="replace")
    rows = []
    for m in _RUN_SCREENING_PAIR_RX.finditer(text):
        pair_id, winner = m.group(1), m.group(2)
        rows.append(
            {
                "pair_id": pair_id,
                "model": model,
                "winner": winner,
                "source_file": str(path),
                "model_variant_known": True,
            }
        )
    return rows


def load_all_screening_outcomes(repo_root: str) -> list[dict]:
    root = Path(repo_root)
    rows = []
    for model, rel_path in KNOWN_MODELS.items():
        rows.extend(parse_run_screening_file(root / rel_path, model))
    return rows


def write_screening_outcomes_csv(rows: list[dict], out_path: str):
    fieldnames = ["pair_id", "model", "winner", "source_file", "model_variant_known"]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parents[2]
    rows = load_all_screening_outcomes(str(repo_root))

    by_model = {}
    for r in rows:
        by_model.setdefault(r["model"], 0)
        by_model[r["model"]] += 1

    print(f"Total decision rows: {len(rows)}")
    for model, count in sorted(by_model.items()):
        print(f"  {model}: {count} pairs")

    out_path = repo_root / "results" / "qual_audit" / "screening_outcomes.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_screening_outcomes_csv(rows, str(out_path))
    print(f"Wrote {out_path}")
