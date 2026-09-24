"""
ViMedAgent — Data preparation
Reads raw VM14K JSONL → filters → creates pilot_100.jsonl + vm14k_clean.jsonl

Filters applied:
  1. 4-option MCQ only
  2. Records where ALL medical_topic values are non-medical noise → removed

Usage:
    python -m data.prepare_data --input /path/to/raw.jsonl
    python -m data.prepare_data --input /path/to/raw.jsonl --no-figures
"""

import argparse
import json
import random
import sys
import os
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PILOT_N    = 100
PILOT_DIST = {"Easy": 33, "Medium": 50, "Hard": 17}

NOISE_TOPICS = {
    "Chemistry", "Analytical Chemistry", "Botany", "Physics", "FDA",
    "Sociology", "Demographics", "Industrial Chemistry", "Health Economics",
    "Entomology", "Classification", "Risk Assessment", "Research",
    "Clinical Research", "Health Administration", "Health System",
    "Health Policy",
}


def load_jsonl(path: str) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def norm_diff(d: str) -> str:
    return "Hard" if d in ("Challenging", "Hard") else d


def get_topics(r: dict) -> list[str]:
    t = r.get("medical_topic", [])
    ts = t if isinstance(t, list) else [t]
    return [x for x in ts if x]


def is_all_noise(r: dict) -> bool:
    topics = get_topics(r)
    return bool(topics) and all(t in NOISE_TOPICS for t in topics)


def prepare(input_path: str, outdir: str, seed: int = 42) -> list[dict]:
    random.seed(seed)
    print(f"Loading: {input_path}")
    rows  = load_jsonl(input_path)
    print(f"  Total: {len(rows)}")

    clean = [r for r in rows if len(r.get("options", [])) == 4]
    print(f"After 4-option filter: {len(clean)}")

    clean = [r for r in clean if not is_all_noise(r)]
    print(f"After noise-topic filter: {len(clean)}")

    Path(outdir).mkdir(parents=True, exist_ok=True)

    clean_path = Path(outdir) / "vm14k_clean.jsonl"
    with open(clean_path, "w", encoding="utf-8") as f:
        for r in clean:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Saved: {clean_path}  ({len(clean)} records)")

    by_diff = defaultdict(list)
    for r in clean:
        by_diff[norm_diff(r["difficulty_level"])].append(r)

    pilot = []
    for diff, n in PILOT_DIST.items():
        pool = by_diff[diff]
        n    = min(n, len(pool))
        pilot.extend(random.sample(pool, n))
    random.shuffle(pilot)

    pilot_path = Path(outdir) / "pilot_100.jsonl"
    with open(pilot_path, "w", encoding="utf-8") as f:
        for r in pilot:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Saved: {pilot_path}  ({len(pilot)} records)")
    print(f"  Distribution: {dict(Counter(norm_diff(r['difficulty_level']) for r in pilot))}")

    return clean


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",      required=True)
    parser.add_argument("--outdir",     default="data")
    parser.add_argument("--seed",       type=int, default=42)
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args()

    clean = prepare(args.input, args.outdir, args.seed)
    print(f"\nDone. Files written to: {args.outdir}/")


if __name__ == "__main__":
    main()
