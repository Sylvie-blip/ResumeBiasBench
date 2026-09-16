"""Unit tests for the numeric normalizer's tricky cases (spec Step 2 guidance).
Run: python3 src/qual_audit/test_preservation.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from preservation import extract_numeric_multiset, diff_numbers, score_terms  # noqa: E402


def check(label, cond):
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {label}")
    if not cond:
        raise SystemExit(1)


def test_comma_number():
    a = extract_numeric_multiset("Managed a budget of 1,200 dollars.")
    b = extract_numeric_multiset("Managed a budget of 1200 dollars.")
    check("1,200 == 1200", a == b)


def test_currency_forms():
    a = extract_numeric_multiset("Raised $2M in funding.")
    b = extract_numeric_multiset("Raised $2,000,000 in funding.")
    c = extract_numeric_multiset("Raised USD 2M in funding.")
    check("$2M == $2,000,000", a == b)
    check("$2M == USD 2M", a == c)


def test_date_range_reformatting():
    a = extract_numeric_multiset("Worked there 2018-2021.")
    b = extract_numeric_multiset("Worked there 2018 to 2021.")
    c = extract_numeric_multiset("Worked there 2018 – 2021.")
    check("2018-2021 == 2018 to 2021", a == b)
    check("2018-2021 == 2018 – 2021", a == c)


def test_date_years_not_double_counted_as_bare_numbers():
    counter = extract_numeric_multiset("Employed 2018-2021 managing 2018 accounts.")
    # The range should mask 2018/2021 as date_year; the second standalone 2018
    # (a distinct occurrence, different context) should still be caught as a
    # bare number since only the range's own span gets masked.
    date_years = {v for (k, v) in counter if k == "date_year"}
    check("range endpoints tagged as date_year", date_years == {2018.0, 2021.0})
    numbers = {v for (k, v) in counter if k == "number"}
    check("second standalone 2018 still counted as bare number", 2018.0 in numbers)


def test_month_year_range_reformatting():
    # Extremely common in this corpus: "01/2010 to 03/2013" reformatted with
    # month precision dropped should not manufacture a numeric diff.
    a = extract_numeric_multiset("Sales Manager 01/2010 to 03/2013.")
    b = extract_numeric_multiset("Sales Manager 2010-2013.")
    check("01/2010 to 03/2013 == 2010-2013 (month precision dropped)", a == b)

    c = extract_numeric_multiset("Sales Manager 01/2010 to Current.")
    d = extract_numeric_multiset("Sales Manager since 2010.")
    # "since 2010" has no range keyword, so 2010 falls through as a bare
    # "number" rather than "date_year" - this is a known residual limitation
    # (documented), not something this test should paper over.
    check("open-ended range still tags start year as date_year",
          ("date_year", 2010.0) in c)


def test_month_name_range_equivalent_to_numeric():
    # Observed in real data: the AI polisher rewrites "07/2007 - 03/2010" as
    # "July 2007 - March 2010". Both must normalize identically.
    a = extract_numeric_multiset("Role held 07/2007 - 03/2010.")
    b = extract_numeric_multiset("Role held July 2007 - March 2010.")
    check("MM/YYYY range == Month YYYY range", a == b)

    c = extract_numeric_multiset("Role held 01/2010 to Current.")
    d = extract_numeric_multiset("Role held January 2010 - Present.")
    check("open-ended MM/YYYY range == open-ended Month YYYY range", c == d)


def test_standalone_month_year_no_bare_digit_leak():
    counter = extract_numeric_multiset("Started role 05/2015 and never left.")
    numbers = {v for (k, v) in counter if k == "number"}
    check("month digit (05) does not leak into bare numbers", 5.0 not in numbers)
    date_years = {v for (k, v) in counter if k == "date_year"}
    check("standalone MM/YYYY still tags the year", 2015.0 in date_years)


def test_multiplier():
    a = extract_numeric_multiset("Grew revenue 3x.")
    counter_keys = {(k, v) for k, v in a}
    check("3x tagged as multiplier", ("multiplier", 3.0) in counter_keys)


def test_percentage():
    a = extract_numeric_multiset("Increased sales 15%.")
    b = extract_numeric_multiset("Increased sales 15 percent.")
    check("15% == 15 percent", a == b)


def test_diff_added_and_changed():
    orig = extract_numeric_multiset("Raised $2M for the team of 5.")
    pol = extract_numeric_multiset("Raised $3M for the team of 5 and grew 2x.")
    result = diff_numbers(orig, pol)
    check("numbers_added counts the new 2x multiplier", result["numbers_added"] == 1)
    check("numbers_changed counts $2M -> $3M as one change", result["numbers_changed"] == 1)
    check("numbers_removed is zero (team of 5 unchanged)", result["numbers_removed"] == 0)


def test_scoped_terms_case_insensitive_dedup():
    # IT domain list has "AWS" (capitalized); software_development has "aws"
    # (lowercase) - both exist in ALL_SKILLS as separate literal entries, and
    # extract_skills can return both if text contains "aws" case-insensitively.
    # Scoping to INFORMATION-TECH's mapped domain ("it") must still catch the
    # match regardless of which casing extract_skills happened to return.
    text = "Experienced with aws cloud infrastructure."
    scores = score_terms(text, text, "INFORMATION-TECH")
    check("aws (lowercase in text) matched under IT scope via case-insensitive lookup",
          len(scores["orig_terms"]) > 0)


if __name__ == "__main__":
    test_comma_number()
    test_currency_forms()
    test_date_range_reformatting()
    test_date_years_not_double_counted_as_bare_numbers()
    test_month_year_range_reformatting()
    test_month_name_range_equivalent_to_numeric()
    test_standalone_month_year_no_bare_digit_leak()
    test_multiplier()
    test_percentage()
    test_diff_added_and_changed()
    test_scoped_terms_case_insensitive_dedup()
    print("\nAll tests passed.")
