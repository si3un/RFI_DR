#!/usr/bin/env python3
"""
Migration script that converts the CSV image-path column to paths "relative" to data_root.

[Background]
  The path column of the existing CSV (e.g. pathstr) contained hard-coded absolute paths.
      /mnt/data/osteoxray/data_nonorm/1024/.../xxx.png
  This breaks every path on servers where the data is mounted elsewhere.
  Stripping the data_root prefix with this script to make paths "relative",
      data_nonorm/1024/.../xxx.png
  means only the config's data.data_root needs to be adjusted per server, so
  the same CSV/code can be reused on any server.

[Usage]
  # Save the result to a separate file (keeps the original; recommended)
  python scripts/migrate_csv_to_relative.py \
      --csv /path/to/total_data.csv \
      --column pathstr \
      --data-root /mnt/data/osteoxray

  # Overwrite the original (a .bak backup is created automatically)
  python scripts/migrate_csv_to_relative.py \
      --csv /path/to/total_data.csv --column pathstr \
      --data-root /mnt/data/osteoxray --inplace

  # Also check that the converted paths actually exist
  python scripts/migrate_csv_to_relative.py ... --check-exists

  * Uses only the standard library, so it runs anywhere without extra installs.
"""
import argparse
import csv
import os
import shutil
import sys


def to_relative(path, data_root):
    """
    Convert an absolute path to a path relative to data_root.
      - returns it unchanged if already relative
      - returns None for absolute paths outside data_root (marks as not convertible)
    """
    if not path:
        return path
    p = os.path.normpath(path)
    root = os.path.normpath(data_root)
    if not os.path.isabs(p):
        return path  # already relative
    if p == root or p.startswith(root + os.sep):
        return os.path.relpath(p, root)
    return None  # outside data_root -> not convertible


def main():
    ap = argparse.ArgumentParser(description="Convert a CSV path column to paths relative to data_root")
    ap.add_argument('--csv', required=True, help='input CSV path')
    ap.add_argument('--column', default='pathstr', help='name of the column holding image paths')
    ap.add_argument('--data-root', required=True, help='top-level folder that relative paths are based on')
    ap.add_argument('--out', default=None, help='output CSV path (default: <csv>.relative.csv)')
    ap.add_argument('--inplace', action='store_true', help='overwrite the original (creates a .bak backup)')
    ap.add_argument('--check-exists', action='store_true',
                    help='check that converted paths actually exist under data_root')
    args = ap.parse_args()

    out = args.csv if args.inplace else (args.out or (args.csv + '.relative.csv'))

    with open(args.csv, newline='') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        if fieldnames is None or args.column not in fieldnames:
            sys.exit(f"[Error] Column '{args.column}' not found in CSV. Available columns: {fieldnames}")
        rows = list(reader)

    converted = skipped_relative = outside = missing = 0
    for row in rows:
        orig = row[args.column]
        rel = to_relative(orig, args.data_root)
        if rel is None:
            outside += 1
            continue  # keep the original
        if rel == orig:
            skipped_relative += 1
            continue
        if args.check_exists and not os.path.exists(os.path.join(args.data_root, rel)):
            missing += 1
        row[args.column] = rel
        converted += 1

    if args.inplace and os.path.exists(args.csv):
        shutil.copy2(args.csv, args.csv + '.bak')

    with open(out, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Done -> {out}")
    print(f"  converted: {converted} | already relative: {skipped_relative} | "
          f"outside data_root (not converted): {outside} | missing files: {missing}")
    if outside:
        print("  Warning: some absolute paths are outside data_root. Check that --data-root is correct.")


if __name__ == '__main__':
    main()
