#!/usr/bin/env python3
"""merge_csv.py

Merge two CSV files by the 'file' column and write a combined CSV.
By default the script performs an outer join and preserves original
column names. When a column exists in both inputs the value from the
first CSV (A) will be used if non-empty, otherwise the value from B.

Usage:
    python3 /Users/sylviadong/Documents/Resume_Bias/Bias_code/merge_csv.py /Users/sylviadong/Documents/new_ent_results_losing/metric_scores_A.csv /Users/sylviadong/Documents/new_metrics_results_losing/metric_scores_A.csv -o new_merged_losing.csv --join outer

Options:
    --join {outer,inner,left,right}  Join type (default: outer)
    -o, --out OUTPUT                 Output CSV path (default: merged.csv)

The script uses only the standard library (csv) so it works without
extra dependencies.
"""

import argparse
import csv
import os
import sys
from collections import OrderedDict


def read_csv_map(path, key_col='file'):
    """Read CSV into an OrderedDict mapping key_col -> row dict.
    If duplicates of key_col appear, keep the first and warn.
    """
    mapping = OrderedDict()
    try:
        with open(path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            if key_col not in reader.fieldnames:
                raise ValueError(f"Key column '{key_col}' not found in {path}. Fields: {reader.fieldnames}")
            for i, row in enumerate(reader, start=1):
                key = row.get(key_col, '')
                if key in mapping:
                    # warn once
                    if mapping[key] is not None:
                        print(f"Warning: duplicate key '{key}' in {path} at row {i}; keeping first occurrence", file=sys.stderr)
                        # mark that we've warned/seen duplicate
                        mapping[key] = mapping[key]
                    continue
                mapping[key] = row
    except UnicodeDecodeError:
        # try latin-1 fallback
        with open(path, 'r', encoding='latin-1', errors='ignore') as f:
            reader = csv.DictReader(f)
            if key_col not in reader.fieldnames:
                raise ValueError(f"Key column '{key_col}' not found in {path}. Fields: {reader.fieldnames}")
            for i, row in enumerate(reader, start=1):
                key = row.get(key_col, '')
                if key in mapping:
                    if mapping[key] is not None:
                        print(f"Warning: duplicate key '{key}' in {path} at row {i}; keeping first occurrence", file=sys.stderr)
                    continue
                mapping[key] = row
    return mapping


def merge_maps(map_a, map_b, key_col='file', join='outer'):
    """Merge two maps of file -> row dict according to join type.
    Keeps original column names. When a column exists in both inputs, the
    value from the first CSV (A) is used if non-empty; otherwise the value
    from the second CSV (B) is used. Returns (fieldnames, rows_list) ready
    to be written by csv.DictWriter.
    """
    keys_a = list(map_a.keys())
    keys_b = list(map_b.keys())
    if join == 'inner':
        keys = [k for k in keys_a if k in map_b]
    elif join == 'left':
        keys = keys_a[:]
    elif join == 'right':
        keys = keys_b[:]
    else:  # outer
        # preserve order: keys from A in original order, then keys from B not in A
        keys = keys_a[:]
        for k in keys_b:
            if k not in map_a:
                keys.append(k)

    # collect fieldnames from both, excluding key_col; preserve order from A then B
    cols_a = []
    cols_b = []
    if map_a:
        sample_a = next(iter(map_a.values()))
        cols_a = [c for c in sample_a.keys() if c != key_col]
    if map_b:
        sample_b = next(iter(map_b.values()))
        cols_b = [c for c in sample_b.keys() if c != key_col]

    # merged columns: start with cols_a then add cols_b that are not already present
    merged_cols = []
    for c in cols_a:
        if c not in merged_cols:
            merged_cols.append(c)
    for c in cols_b:
        if c not in merged_cols:
            merged_cols.append(c)

    out_cols = [key_col] + merged_cols

    rows = []
    for k in keys:
        row = {key_col: k}
        a_row = map_a.get(k)
        b_row = map_b.get(k)
        for c in merged_cols:
            val = ''
            if a_row and c in a_row and a_row.get(c, '') != '':
                val = a_row.get(c, '')
            elif b_row and c in b_row:
                val = b_row.get(c, '')
            row[c] = val
        rows.append(row)

    return out_cols, rows


def write_csv(path, fieldnames, rows):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def main():
    parser = argparse.ArgumentParser(description='Merge two CSVs by the "file" column')
    parser.add_argument('csv_a', help='First CSV file (left / A)')
    parser.add_argument('csv_b', help='Second CSV file (right / B)')
    parser.add_argument('-o', '--out', default='merged.csv', help='Output CSV path')
    parser.add_argument('--join', choices=['outer', 'inner', 'left', 'right'], default='outer', help='Join type (default: outer)')
    parser.add_argument('--key', default='file', help='Key column name to join on (default: file)')

    args = parser.parse_args()

    map_a = read_csv_map(args.csv_a, key_col=args.key)
    map_b = read_csv_map(args.csv_b, key_col=args.key)

    fieldnames, rows = merge_maps(map_a, map_b, key_col=args.key, join=args.join)

    write_csv(args.out, fieldnames, rows)
    print(f"Wrote merged CSV to {args.out} (join={args.join}, rows={len(rows)})")


if __name__ == '__main__':
    main()
