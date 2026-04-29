#!/usr/bin/env python3
"""
gen_web_data.py — Generate lightweight metadata JSON for the Bones-Seed web viewer.

Usage:
  python gen_web_data.py \
      --input  ../../artifacts/bones-seed/metadata/seed_metadata_v004.csv \
      --output web/metadata.json

The output is a compact JSON file (~3–5 MB vs. 140 MB for the full CSV) that the
GitHub Pages viewer (web/index.html) can load directly from any static host.

Typical workflow:
  1. Run this script to generate metadata.json
  2. Host metadata.json (+ motion CSV files + G1 FBX) on Hugging Face Datasets or GitHub LFS
  3. Open web/index.html, fill in the URLs, and share the resulting URL hash
"""
import argparse
import json
import sys
from pathlib import Path

# Try pandas first; fall back to csv stdlib
try:
    import pandas as pd
    USE_PANDAS = True
except ImportError:
    import csv
    USE_PANDAS = False


DISPLAY_COLS = [
    "move_name",
    "package",
    "category",
    "move_duration_frames",
    "is_mirror",
    "actor_gender",
    "actor_height",
    "content_short_description",
    "move_g1_path",
    "content_type_of_movement",
    "content_body_position",
]


def main():
    ap = argparse.ArgumentParser(description="Generate compact metadata JSON for the web viewer")
    ap.add_argument("--input",  "-i", required=True,  help="Path to seed_metadata_vXXX.csv")
    ap.add_argument("--output", "-o", required=True,  help="Output JSON path (e.g. web/metadata.json)")
    ap.add_argument("--pretty",       action="store_true", help="Pretty-print JSON (larger, human-readable)")
    ap.add_argument("--cols",         nargs="*", default=None,
                    help="Override which columns to include (defaults to DISPLAY_COLS)")
    args = ap.parse_args()

    inp = Path(args.input)
    out = Path(args.output)
    cols = args.cols or DISPLAY_COLS

    if not inp.exists():
        print(f"ERROR: input file not found: {inp}", file=sys.stderr)
        sys.exit(1)

    out.parent.mkdir(parents=True, exist_ok=True)

    print(f"Reading {inp} …")

    if USE_PANDAS:
        rows, filters = _load_pandas(inp, cols)
    else:
        rows, filters = _load_csv(inp, cols)

    result = {
        "version":    inp.stem,
        "total":      len(rows),
        "cols":       cols,
        "filters":    filters,
        "rows":       rows,
    }

    indent = 2 if args.pretty else None
    separators = None if args.pretty else (",", ":")
    raw = json.dumps(result, ensure_ascii=False, indent=indent, separators=separators)
    out.write_text(raw, encoding="utf-8")

    size_mb = out.stat().st_size / 1024 / 1024
    print(f"Written {len(rows):,} rows → {out}  ({size_mb:.2f} MB)")
    print(f"Columns: {cols}")
    print(f"Filter options: {list(filters.keys())}")


def _load_pandas(inp, cols):
    import pandas as pd

    # Only read columns we need; fall back gracefully for missing ones
    available = pd.read_csv(inp, nrows=0).columns.tolist()
    read_cols  = [c for c in cols if c in available]
    missing    = [c for c in cols if c not in available]
    if missing:
        print(f"WARNING: columns not found, skipped: {missing}", file=sys.stderr)

    df = pd.read_csv(inp, usecols=read_cols, dtype=str)
    df = df[read_cols]  # preserve order

    # Numeric coerce
    for c in ("move_duration_frames",):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)

    # Build filter options from actual data
    frames_col = df["move_duration_frames"] if "move_duration_frames" in df.columns else None
    filters = {
        "packages":   _uniq(df, "package"),
        "categories": _uniq(df, "category"),
        "movTypes":   _uniq(df, "content_type_of_movement"),
        "bodyPos":    _uniq(df, "content_body_position"),
        "genders":    _uniq(df, "actor_gender"),
        "heights":    _uniq(df, "actor_height"),
        "max_frames": int(frames_col.max()) if frames_col is not None else 0,
    }

    # Convert to compact row arrays
    rows = []
    for tup in df.itertuples(index=False, name=None):
        rows.append(list(tup))

    return rows, filters


def _uniq(df, col):
    if col not in df.columns:
        return []
    return sorted(df[col].dropna().unique().tolist())


def _load_csv(inp, cols):
    """Fallback implementation using stdlib csv (no pandas)."""
    import csv

    rows = []
    filters_sets = {
        "packages":   set(), "categories": set(), "movTypes": set(),
        "bodyPos":    set(), "genders":    set(), "heights":  set(),
    }
    max_frames = 0
    col_idx = None

    with open(inp, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if col_idx is None:
                col_idx = True  # just a flag
            r = []
            for c in cols:
                val = row.get(c, "")
                if c == "move_duration_frames":
                    try:
                        val = int(float(val))
                        if val > max_frames:
                            max_frames = val
                    except (ValueError, TypeError):
                        val = 0
                r.append(val)
            rows.append(r)
            ci = {c: cols.index(c) for c in cols}
            for k, col in [("packages","package"),("categories","category"),
                           ("movTypes","content_type_of_movement"),
                           ("bodyPos","content_body_position"),
                           ("genders","actor_gender"),("heights","actor_height")]:
                if col in cols:
                    v = row.get(col, "")
                    if v:
                        filters_sets[k].add(v)

    filters = {k: sorted(v) for k, v in filters_sets.items()}
    filters["max_frames"] = max_frames
    return rows, filters


if __name__ == "__main__":
    main()
