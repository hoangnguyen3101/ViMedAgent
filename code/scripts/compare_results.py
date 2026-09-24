"""
ViMedAgent v2 — compare two result sets (e.g. Ollama backend vs SGLang backend).

For a (framework, dataset) pair, loads every seed file of model A and model B
under results(_test)/ and prints per-seed accuracy, mean±std, null rate, and
the per-question agreement between the two backends on matched (id, seed).

Usage:
    python scripts/compare_results.py --framework zero_shot --dataset pilot_100 \
        --a ollama2-qwen3-32b --b sglang-qwen3-32b
"""
import argparse
import glob
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from infrastructure.config import FRAMEWORK_TIER, RESULTS_DIR, RESULTS_TEST_DIR


def load(framework, model, dataset):
    safe = model.replace("/", "_").replace(".", "_")
    root = RESULTS_TEST_DIR if dataset.startswith("pilot") else RESULTS_DIR
    pat = os.path.join(root, FRAMEWORK_TIER[framework], framework,
                       f"{framework}__{safe}__{dataset}__s*.jsonl")
    out = {}
    for path in sorted(glob.glob(pat)):
        seed = int(path.rsplit("__s", 1)[1].split(".")[0])
        rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
        out[seed] = {r["id"]: r for r in rows}
    return out


def summarize(name, by_seed):
    accs = []
    print(f"  {name}")
    for seed, rows in sorted(by_seed.items()):
        n = len(rows)
        c = sum(1 for r in rows.values() if r.get("correct"))
        nulls = sum(1 for r in rows.values() if r.get("answer") is None)
        calls = sum(r.get("api_calls", 1) for r in rows.values())
        acc = 100 * c / n if n else 0
        accs.append(acc)
        print(f"    seed {seed}: n={n:5d} acc={acc:5.1f}% null={nulls} api_calls={calls}")
    if accs:
        sd = statistics.stdev(accs) if len(accs) > 1 else 0.0
        print(f"    mean = {statistics.mean(accs):.2f}% ± {sd:.2f}%  ({len(accs)} seeds)")
    return accs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--framework", default="zero_shot")
    ap.add_argument("--dataset", default="pilot_100")
    ap.add_argument("--a", default="ollama2-qwen3-32b")
    ap.add_argument("--b", default="sglang-qwen3-32b")
    args = ap.parse_args()

    A = load(args.framework, args.a, args.dataset)
    B = load(args.framework, args.b, args.dataset)
    print(f"\n{args.framework} × {args.dataset}\n")
    accA = summarize(args.a, A)
    print()
    accB = summarize(args.b, B)

    if accA and accB:
        d = statistics.mean(accB) - statistics.mean(accA)
        print(f"\n  Δ mean acc (B − A) = {d:+.2f} pp")

    common_seeds = sorted(set(A) & set(B))
    if common_seeds:
        agree = tot = both_c = 0
        for s in common_seeds:
            for qid, ra in A[s].items():
                rb = B[s].get(qid)
                if rb is None:
                    continue
                tot += 1
                agree += ra.get("answer") == rb.get("answer")
                both_c += bool(ra.get("correct")) and bool(rb.get("correct"))
        if tot:
            print(f"  per-question agreement on {tot} matched (id, seed): "
                  f"{100*agree/tot:.1f}% same answer | both correct {100*both_c/tot:.1f}%")


if __name__ == "__main__":
    main()
