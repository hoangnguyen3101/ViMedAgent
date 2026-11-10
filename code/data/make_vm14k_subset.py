"""Fixed, stratified VM14K subset for the consultation ablations (baseline MedAgents, A, B, CoT-SC).

Rule (re-running always yields the same ID list):
  1. Population: data/vm14k_clean.jsonl with one ID per exact content (question + options),
     the smallest ID as a string, and every duplicate group whose gold answers disagree dropped.
     This is the same set as the sensitivity set of metrics/generate_paper_tables.dataset_audit
     (10,139 IDs), so no question appears twice and every gold label is well defined.
  2. Stratum = difficulty on the paper's 3-level scale (metrics.io.DIFFICULTY_MAP:
     Medium -> Moderate, Challenging/Hard -> Hard) x first listed specialty (medical_topic[0]).
     VM14K items carry 1-6 specialties; the first one stands for the item.
  3. Proportional allocation; fractions rounded by largest remainder (ties broken by stratum
     key), so the total is exactly N.
  4. Within each stratum the IDs are sorted and drawn with random.Random(seed).sample.

Outputs, next to the source file:
  <stem>.jsonl          the selected source lines, byte for byte, in source order (use as --dataset)
  <stem>_ids.txt        the fixed ID list, one per line, sorted
  <stem>.manifest.json  rule, seed, source SHA-256, allocation and full-vs-subset marginals

Run from code/:
  python data/make_vm14k_subset.py                 # writes data/vm14k_sub2000.*
The stem must start with "vm14k_sub" so the runner routes results to results_ablation/.
An existing ID list with different content is never overwritten unless --force is given.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_DIR))

from metrics.io import DIFFICULTY_MAP  # noqa: E402

DEFAULT_SEED = 42  # same seed as the first experiment run (config.SEEDS[0])


def content_key(r: dict) -> str:
    # Same key as generate_paper_tables.dataset_audit / load_content_keys.
    return json.dumps([r["question"], r["options"]], ensure_ascii=False)


def stratum(r: dict) -> tuple[str, str]:
    diff = DIFFICULTY_MAP.get(str(r.get("difficulty_level")), "Unknown")
    topics = r.get("medical_topic") or []
    return diff, (str(topics[0]) if topics else "Unknown")


def allocate(sizes: dict[tuple, int], n: int) -> dict[tuple, int]:
    total = sum(sizes.values())
    exact = {k: v * n / total for k, v in sizes.items()}
    alloc = {k: int(x) for k, x in exact.items()}
    left = n - sum(alloc.values())
    for k in sorted(sizes, key=lambda k: (-(exact[k] - alloc[k]), k))[:left]:
        alloc[k] += 1
    return alloc


def shares(rows: list[dict], key) -> dict[str, float]:
    c = Counter()
    for r in rows:
        for v in key(r):
            c[v] += 1
    tot = sum(c.values())
    return {k: round(100 * v / tot, 2) for k, v in c.most_common()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--source", type=Path, default=CODE_DIR / "data" / "vm14k_clean.jsonl")
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--stem", default=None, help="output stem; default vm14k_sub<N>")
    ap.add_argument("--keep-duplicates", action="store_true",
                    help="sample from every row instead of the one-per-content, gold-consistent set")
    ap.add_argument("--force", action="store_true", help="overwrite an existing, different ID list")
    a = ap.parse_args()

    stem = a.stem or f"vm14k_sub{a.n}"
    if not stem.lower().startswith("vm14k_sub"):
        sys.exit("--stem must start with 'vm14k_sub' (results routing, see config.get_results_dir)")
    out_dir = a.source.parent
    out_jsonl, out_ids, out_meta = (out_dir / f"{stem}.jsonl", out_dir / f"{stem}_ids.txt",
                                    out_dir / f"{stem}.manifest.json")

    raw = a.source.read_bytes()
    lines = [ln for ln in raw.decode("utf-8").splitlines(keepends=True) if ln.strip()]
    rows = [json.loads(ln) for ln in lines]
    ids = [str(r["id"]) for r in rows]
    if len(set(ids)) != len(ids):
        sys.exit("source has duplicate IDs")

    groups = defaultdict(list)
    for r in rows:
        groups[content_key(r)].append(r)
    if a.keep_duplicates:
        population = {str(r["id"]) for r in rows}
    else:
        population = {min(str(x["id"]) for x in g) for g in groups.values()
                      if len({x.get("answer") for x in g}) == 1}
    by_id = {str(r["id"]): r for r in rows}
    pop_rows = [by_id[i] for i in ids if i in population]
    if a.n > len(pop_rows):
        sys.exit(f"--n {a.n} exceeds the population ({len(pop_rows)})")

    strata = defaultdict(list)
    for r in pop_rows:
        strata[stratum(r)].append(str(r["id"]))
    alloc = allocate({k: len(v) for k, v in strata.items()}, a.n)
    rng = random.Random(a.seed)
    chosen = set()
    for k in sorted(strata):  # fixed order: the draw sequence depends only on the seed
        chosen.update(rng.sample(sorted(strata[k]), alloc[k]))
    assert len(chosen) == a.n

    id_list = sorted(chosen)
    id_text = "".join(i + "\n" for i in id_list)
    if out_ids.exists() and out_ids.read_text(encoding="utf-8") != id_text and not a.force:
        sys.exit(f"{out_ids.name} already exists with a different ID list; refusing to overwrite "
                 f"(the list is meant to stay fixed). Use --force only if you really mean it.")

    sub_lines = [ln for ln, i in zip(lines, ids) if i in chosen]
    sub_rows = [by_id[i] for i in ids if i in chosen]
    marg = {
        "difficulty_3level": (lambda r: [stratum(r)[0]]),
        "difficulty_raw": (lambda r: [str(r.get("difficulty_level"))]),
        "first_specialty": (lambda r: [stratum(r)[1]]),
        "all_specialty_labels": (lambda r: [str(t) for t in (r.get("medical_topic") or [])]),
        "gold_letter": (lambda r: [str(r.get("answer"))]),
    }
    manifest = {
        "purpose": "Fixed VM14K subset for the consultation ablations: baseline MedAgents (Qwen), "
                   "A (heterogeneous panel), B (no report), CoT-SC (Qwen).",
        "script": "code/data/make_vm14k_subset.py",
        "source": str(a.source.relative_to(CODE_DIR)).replace("\\", "/"),
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "source_rows": len(rows),
        "population": ("all rows" if a.keep_duplicates else
                       "one ID per exact content (min string ID), conflicting-gold groups dropped"),
        "population_size": len(pop_rows),
        "n": a.n, "seed": a.seed,
        "stratum": "DIFFICULTY_MAP(difficulty_level) x medical_topic[0]",
        "allocation": "proportional, largest remainder",
        "n_strata": len(strata), "n_strata_sampled": sum(1 for v in alloc.values() if v),
        "ids_sha256": hashlib.sha256(id_text.encode("utf-8")).hexdigest(),
        "files": {"jsonl": out_jsonl.name, "ids": out_ids.name},
        "allocation_by_stratum": {f"{d} | {t}": {"population": len(strata[(d, t)]), "sampled": alloc[(d, t)]}
                                  for d, t in sorted(strata)},
        "marginals_pct": {name: {"population": shares(pop_rows, f), "subset": shares(sub_rows, f)}
                          for name, f in marg.items()},
    }
    out_jsonl.write_text("".join(sub_lines), encoding="utf-8", newline="")
    out_ids.write_text(id_text, encoding="utf-8", newline="")
    out_meta.write_text(json.dumps(manifest, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"population {len(pop_rows):,} -> {a.n:,} IDs over {manifest['n_strata_sampled']}/{len(strata)} strata")
    print(f"wrote {out_jsonl.name}, {out_ids.name}, {out_meta.name}  (ids_sha256 {manifest['ids_sha256'][:12]})")


if __name__ == "__main__":
    main()
