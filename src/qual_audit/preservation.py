"""Step 1 (qualification-term preservation) + Step 2 (numeric inflation) +
Step 3 (preserved-subset threshold sweep) of the qualification-preservation audit.

Step 1 reuses src/conv_to_json.py:extract_skills exactly as published - no new
matching logic. That function returns matches against the *global* flat skill
set (ALL_SKILLS, spanning all 24 domains); we scope it to a pair's occupational
category by intersecting its output with that category's mapped domain(s)
(case-insensitively, since extract_skills is known to emit duplicate skills in
different casings across domains, e.g. both "aws" and "AWS" - see
data/humanresumesjson/10138632.json for a real example of this).

Step 2 is new (the spec asks for a numeric-inflation pass independent of Step 1)
but deliberately simple: regex extraction only, multiset comparison, no LLM
calls. Extraction runs in a fixed order and masks each match out of the text
before the next pass, so e.g. a bare year inside a date range is never also
counted as a plain integer.
"""

import re
import sys
from collections import Counter
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "src"))
from conv_to_json import extract_skills, SKILLS_BY_DOMAIN  # noqa: E402

from category_map import domains_for_category  # noqa: E402


# ---------------------------------------------------------------------------
# Step 1: qualification-term (skill) preservation
# ---------------------------------------------------------------------------

def domain_skill_set_lower(category: str) -> set[str]:
    domains = domains_for_category(category)
    skills = set()
    for d in domains:
        skills.update(SKILLS_BY_DOMAIN[d])
    return {s.lower() for s in skills}


def scoped_terms(text: str, category: str) -> set[str]:
    """extract_skills(text) unmodified, then filtered to this category's domain(s)."""
    all_matches = extract_skills(text)
    domain_lower = domain_skill_set_lower(category)
    return {s for s in all_matches if s.lower() in domain_lower}


def score_terms(orig_text: str, pol_text: str, category: str) -> dict:
    orig_terms = scoped_terms(orig_text, category)
    pol_terms = scoped_terms(pol_text, category)

    retained = orig_terms & pol_terms
    dropped = orig_terms - pol_terms
    added = pol_terms - orig_terms

    undefined_retention = len(orig_terms) == 0
    retention_rate = (len(retained) / len(orig_terms)) if not undefined_retention else None

    return {
        "orig_terms": orig_terms,
        "pol_terms": pol_terms,
        "retained": retained,
        "dropped": dropped,
        "added": added,
        "retention_rate": retention_rate,
        "addition_count": len(added),
        "undefined_retention": undefined_retention,
    }


# ---------------------------------------------------------------------------
# Step 2: numeric inflation pass
# ---------------------------------------------------------------------------

_YEAR = r"(?:19\d{2}|20\d{2})"
_MONTH_NAME = (
    r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?"
    r"|aug(?:ust)?|sept?(?:ember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"
)
# A month qualifier before a year, either numeric ("01/") or spelled out
# ("January "/"Jan "). The AI polisher in this corpus systematically rewrites
# numeric MM/YYYY into "Month YYYY" (e.g. "07/2007" -> "July 2007"), so both
# forms must normalize identically or that rewrite alone would manufacture a
# numeric diff on nearly every pair.
_MONTH_PREFIX = rf"(?:\d{{1,2}}/|{_MONTH_NAME}\s+)"

# Date ranges, with an optional month qualifier on either endpoint (very common
# in this corpus: "01/2010 to 03/2013", "July 2007 - March 2010") and an
# open-ended second endpoint ("01/2010 to Current"/"Present"). Month info is
# deliberately dropped from the normalized value - only the year(s) are tagged
# - so that month-precision/format reformatting is treated as equivalent
# rather than manufacturing a false numeric diff.
_DATE_RANGE_RX = re.compile(
    rf"\b(?:{_MONTH_PREFIX})?({_YEAR})\s*(?:-|–|—|to)\s*"
    rf"(?:(?:{_MONTH_PREFIX})?({_YEAR})|current|present)\b",
    re.IGNORECASE,
)
# A standalone month-qualified year not caught by the range pass above (e.g.
# "Since 05/2015" or "Since January 2015" with no end date). Masks the whole
# token so the month component never leaks into the bare-number pass as
# unrelated noise, while still tagging the year as date_year for consistency
# with however the same date is written on the other side of the pair.
_MONTH_YEAR_RX = re.compile(rf"\b(?:\d{{1,2}}/|{_MONTH_NAME}\s+)({_YEAR})\b", re.IGNORECASE)
_MULTIPLIER_RX = re.compile(r"\b(\d+(?:\.\d+)?)[xX]\b")
_CURRENCY_RX = re.compile(
    r"(?:USD\s*|US\$|\$)\s*(\d[\d,]*(?:\.\d+)?)\s*"
    r"(million|mil\.?|m\b|thousand|k\b|billion|bn\b|b\b)?",
    re.IGNORECASE,
)
_PERCENT_RX = re.compile(r"\b(\d+(?:\.\d+)?)\s*(?:%|percent\b)", re.IGNORECASE)
_BARE_NUMBER_RX = re.compile(r"\b\d{1,3}(?:,\d{3})+(?:\.\d+)?\b|\b\d+(?:\.\d+)?\b")

_CURRENCY_SUFFIX_MULT = {
    "million": 1_000_000, "mil": 1_000_000, "mil.": 1_000_000, "m": 1_000_000,
    "thousand": 1_000, "k": 1_000,
    "billion": 1_000_000_000, "bn": 1_000_000_000, "b": 1_000_000_000,
}


def _mask(pattern: re.Pattern, text: str, on_match) -> tuple[str, list]:
    matches = []

    def _sub(m):
        matches.append(on_match(m))
        return " " * len(m.group(0))

    new_text = pattern.sub(_sub, text)
    return new_text, [m for m in matches if m is not None]


def extract_numeric_multiset(text: str) -> Counter:
    """Returns a Counter of (kind, normalized_value) -> occurrence count.

    kind distinguishes unit type (date_year / multiplier / currency / percentage /
    number) so a dollar amount is never conflated with a percentage or plain count
    of the same magnitude - this is a unit distinction, not semantic metric-matching,
    which the spec explicitly says not to attempt.
    """
    counter: Counter = Counter()
    working = text

    def date_range_cb(m):
        years = [("date_year", float(m.group(1)))]
        if m.group(2):  # None when the range was open-ended (.../Current/Present)
            years.append(("date_year", float(m.group(2))))
        return years

    working, date_pairs = _mask(_DATE_RANGE_RX, working, date_range_cb)
    for pair in date_pairs:
        for kind, val in pair:
            counter[(kind, val)] += 1

    def month_year_cb(m):
        return ("date_year", float(m.group(1)))

    working, month_years = _mask(_MONTH_YEAR_RX, working, month_year_cb)
    for kind, val in month_years:
        counter[(kind, val)] += 1

    def mult_cb(m):
        return ("multiplier", float(m.group(1)))

    working, mults = _mask(_MULTIPLIER_RX, working, mult_cb)
    for kind, val in mults:
        counter[(kind, val)] += 1

    def currency_cb(m):
        amount = float(m.group(1).replace(",", ""))
        suffix = (m.group(2) or "").lower().rstrip(".")
        mult = _CURRENCY_SUFFIX_MULT.get(suffix, 1)
        return ("currency", amount * mult)

    working, currencies = _mask(_CURRENCY_RX, working, currency_cb)
    for kind, val in currencies:
        counter[(kind, val)] += 1

    def percent_cb(m):
        return ("percentage", float(m.group(1)))

    working, percents = _mask(_PERCENT_RX, working, percent_cb)
    for kind, val in percents:
        counter[(kind, val)] += 1

    def number_cb(m):
        return ("number", float(m.group(0).replace(",", "")))

    working, numbers = _mask(_BARE_NUMBER_RX, working, number_cb)
    for kind, val in numbers:
        counter[(kind, val)] += 1

    return counter


def diff_numbers(orig_counter: Counter, pol_counter: Counter) -> dict:
    pure_added = pol_counter - orig_counter
    pure_removed = orig_counter - pol_counter

    # Heuristic (documented, not semantic per-claim matching): within the same
    # unit kind, pair up excess-added against excess-removed occurrences as
    # "changed" (e.g. currency 2,000,000 removed + currency 3,000,000 added ==
    # one inflated dollar figure), leaving any remaining imbalance as pure
    # additions/removals.
    changed_count = 0
    added_detail = dict(pure_added)
    removed_detail = dict(pure_removed)

    kinds = {k for (k, _v) in list(added_detail) + list(removed_detail)}
    for kind in kinds:
        added_total = sum(v for (k, _val), v in added_detail.items() if k == kind)
        removed_total = sum(v for (k, _val), v in removed_detail.items() if k == kind)
        pair_count = min(added_total, removed_total)
        changed_count += pair_count

        remaining = pair_count
        for key in [k for k in list(added_detail) if k[0] == kind]:
            if remaining <= 0:
                break
            take = min(remaining, added_detail[key])
            added_detail[key] -= take
            remaining -= take
            if added_detail[key] == 0:
                del added_detail[key]

        remaining = pair_count
        for key in [k for k in list(removed_detail) if k[0] == kind]:
            if remaining <= 0:
                break
            take = min(remaining, removed_detail[key])
            removed_detail[key] -= take
            remaining -= take
            if removed_detail[key] == 0:
                del removed_detail[key]

    return {
        "numbers_added": sum(added_detail.values()),
        "numbers_removed": sum(removed_detail.values()),
        "numbers_changed": changed_count,
        "added_detail": added_detail,
        "removed_detail": removed_detail,
    }


def score_numbers(orig_text: str, pol_text: str) -> dict:
    orig_counter = extract_numeric_multiset(orig_text)
    pol_counter = extract_numeric_multiset(pol_text)
    return diff_numbers(orig_counter, pol_counter)


# ---------------------------------------------------------------------------
# Step 3: preserved-subset threshold gate
# ---------------------------------------------------------------------------

THRESHOLDS = (0.90, 0.95, 1.00)

# Found by chance during the Step 5 spot-check (pair 10466208), not something
# the original spec asked us to look for: some AI-polished resumes left a
# literal unfilled template placeholder where a quantified figure should be
# (e.g. "[Number]", "[Year]") instead of the original's real number. That is a
# content *downgrade*, not an upgrade, so a pair with this defect cannot be
# called "provably unchanged" regardless of what the keyword/numeric diff
# says - it is gated out of the preserved subset alongside undefined_retention.
PLACEHOLDER_FIGURE_RX = re.compile(
    r"\[(number|year|month,\s*year|amount|percentage)\]", re.IGNORECASE
)


def has_placeholder_figure(pol_text: str) -> bool:
    return bool(PLACEHOLDER_FIGURE_RX.search(pol_text))


def preserved_flags(term_scores: dict, number_scores: dict, placeholder_damaged: bool) -> dict:
    flags = {}
    for t in THRESHOLDS:
        if term_scores["undefined_retention"] or placeholder_damaged:
            flags[t] = False
            continue
        flags[t] = (
            term_scores["retention_rate"] >= t
            and term_scores["addition_count"] == 0
            and number_scores["numbers_added"] == 0
            and number_scores["numbers_changed"] == 0
        )
    return flags


def score_pair(orig_text: str, pol_text: str, category: str) -> dict:
    term_scores = score_terms(orig_text, pol_text, category)
    number_scores = score_numbers(orig_text, pol_text)
    placeholder_damaged = has_placeholder_figure(pol_text)
    flags = preserved_flags(term_scores, number_scores, placeholder_damaged)
    return {
        "term_scores": term_scores,
        "number_scores": number_scores,
        "placeholder_damaged": placeholder_damaged,
        "preserved_flags": flags,
    }
