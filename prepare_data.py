"""
ViMedAgent — Data preparation
Đọc file VM14K gốc → filter → tạo pilot_100.jsonl + vm14k_clean.jsonl

Usage:
    # Nếu có 1 file gốc
    python prepare_data.py --input data-processed-shuffled0.jsonl

    # Nếu muốn chỉ định output dir
    python prepare_data.py --input data-processed-shuffled0.jsonl --outdir .
"""

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


PILOT_N = 100
PILOT_DIST = {"Easy": 33, "Medium": 50, "Hard": 17}   # Challenging+Hard → Hard


def load_jsonl(path: str) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def normalize_difficulty(d: str) -> str:
    return "Hard" if d in ("Challenging", "Hard") else d


def prepare(input_path: str, outdir: str, seed: int = 42):
    random.seed(seed)

    print(f"Loading: {input_path}")
    rows = load_jsonl(input_path)
    print(f"  Total rows: {len(rows)}")

    # ── Stats before filter ───────────────────────────────
    opt_counts = Counter(len(r["options"]) for r in rows)
    print(f"  Option count distribution: {dict(opt_counts)}")

    diff_dist = Counter(r["difficulty_level"] for r in rows)
    print(f"  Difficulty distribution: {dict(diff_dist)}")

    # ── Filter: 4-option MCQ only ─────────────────────────
    clean = [r for r in rows if len(r["options"]) == 4]
    print(f"\nAfter filter (4-option only): {len(clean)} câu")

    # ── Verify required fields ────────────────────────────
    required = {"id", "question", "options", "answer", "difficulty_level"}
    missing_fields = [r["id"] for r in clean
                      if not required.issubset(r.keys())]
    if missing_fields:
        print(f"  WARNING: {len(missing_fields)} rows missing fields: {missing_fields[:5]}")

    # ── Save vm14k_clean.jsonl ────────────────────────────
    clean_path = Path(outdir) / "vm14k_clean.jsonl"
    with open(clean_path, "w", encoding="utf-8") as f:
        for r in clean:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Saved: {clean_path}  ({len(clean)} câu)")

    # ── Stratified pilot sample ───────────────────────────
    by_diff = defaultdict(list)
    for r in clean:
        key = normalize_difficulty(r["difficulty_level"])
        by_diff[key].append(r)

    print(f"\nStratified pool — Easy:{len(by_diff['Easy'])} "
          f"Medium:{len(by_diff['Medium'])} Hard:{len(by_diff['Hard'])}")

    pilot = []
    for diff, n in PILOT_DIST.items():
        pool = by_diff[diff]
        if len(pool) < n:
            print(f"  WARNING: only {len(pool)} {diff} samples, needed {n}")
            n = len(pool)
        pilot.extend(random.sample(pool, n))

    random.shuffle(pilot)

    pilot_path = Path(outdir) / "pilot_100.jsonl"
    with open(pilot_path, "w", encoding="utf-8") as f:
        for r in pilot:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    pilot_diff = Counter(normalize_difficulty(r["difficulty_level"]) for r in pilot)
    print(f"Saved: {pilot_path}  ({len(pilot)} câu)")
    print(f"  Pilot distribution: {dict(pilot_diff)}")

    # ── Topic distribution summary ────────────────────────
    topics = []
    for r in clean:
        t = r.get("medical_topic", [])
        if isinstance(t, list):
            topics.extend(t)
        else:
            topics.append(t)
    top10 = Counter(topics).most_common(10)
    print(f"\nTop-10 topics in clean set:")
    for topic, count in top10:
        print(f"  {topic:<35} {count:>5}")

    print(f"\nDone. Files ready in: {outdir}/")
    print(f"  pilot_100.jsonl   → test pipeline")
    print(f"  vm14k_clean.jsonl → full experiments")


def main():
    parser = argparse.ArgumentParser(description="ViMedAgent data preparation")
    parser.add_argument("--input",  required=True,
                        help="Path to raw VM14K JSONL file")
    parser.add_argument("--outdir", default=".",
                        help="Output directory (default: current)")
    parser.add_argument("--seed",   type=int, default=42)
    args = parser.parse_args()
    prepare(args.input, args.outdir, args.seed)


if __name__ == "__main__":
    main()
