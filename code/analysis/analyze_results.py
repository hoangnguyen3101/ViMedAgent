"""
ViMedAgent -- Result analysis and table generation
Reads all per-seed JSONL files, aggregates, prints Tables I-VII from paper.

Usage:
    python -m analysis.analyze_results --dataset pilot_100
    python -m analysis.analyze_results --dataset vm14k_clean
    python -m analysis.analyze_results --dataset vm14k_clean --results-dir results
"""

import argparse
import json
import os
from collections import Counter, defaultdict
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from infrastructure.config import (
    SEEDS, TIER1_FRAMEWORKS, TIER2_FRAMEWORKS, TIER3_FRAMEWORKS,
    ALL_FRAMEWORKS, MODEL_NAMES, EN_REFERENCE, FRAMEWORK_TIER,
)
from metrics.compute import (
    compute_pass_at_1, compute_pass_at_k, compute_mean_std,
    compute_collaboration_gain, compute_nmar, compute_icr,
    compute_cld, fisher_test_vs_reference, mcnemar_test,
)

# Pass@k is only valid for these frameworks:
#   cot_sc:          k=5 independent CoT sampling paths
#   voting_ensemble: k=3 independent shuffled runs
# For all other frameworks individual_votes = agent votes within 1 run.
PASS_AT_K_MAP = {"cot_sc": 5, "voting_ensemble": 3}

# Frameworks with inter-agent votes (eligible for NMAR/ICR).
# Tier 2 (voting_ensemble) uses shuffled runs, not agent votes — excluded.
MAS_FRAMEWORKS = TIER3_FRAMEWORKS


# ── Data loading ──────────────────────────────────────────

def _path(results_dir, framework, model, dataset, seed):
    safe = model.replace("/", "_").replace(".", "_")
    return os.path.join(
        results_dir,
        FRAMEWORK_TIER[framework],
        framework,
        f"{framework}__{safe}__{dataset}__s{seed}.jsonl",
    )


def load_seed(results_dir, framework, model, dataset, seed) -> list:
    path = _path(results_dir, framework, model, dataset, seed)
    if not os.path.exists(path):
        return []
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                out.append(json.loads(line))
            except Exception:
                pass
    return out


def load_all_seeds(results_dir, framework, model, dataset) -> list:
    """Return list of per-seed result lists (only non-empty seeds)."""
    return [
        r for seed in SEEDS
        if (r := load_seed(results_dir, framework, model, dataset, seed))
    ]


def load_flat(results_dir, framework, model, dataset) -> list:
    """All seeds concatenated -- for NMAR/ICR computation."""
    out = []
    for seed in SEEDS:
        out.extend(load_seed(results_dir, framework, model, dataset, seed))
    return out


def get_agg(results_dir, framework, model, dataset):
    """Return mean+/-std dict or None if missing."""
    per_seed = [compute_pass_at_1(r)
                for r in load_all_seeds(results_dir, framework, model, dataset)]
    if not per_seed:
        return None
    return compute_mean_std(per_seed)


def best_baseline(results_dir, model, dataset) -> float:
    """
    CG baseline = max Pass@1 mean among zero_shot and cot.
    These are the strongest no-communication methods.
    """
    best = 0.0
    for fw in ("zero_shot", "cot"):
        m = get_agg(results_dir, fw, model, dataset)
        if m:
            best = max(best, m["overall"]["mean"])
    return best


# ── Formatting ────────────────────────────────────────────

def ms(m) -> str:
    if m is None:
        return "---"
    return f"{m['mean']:.2f}+/-{m['std']:.2f}"


def diff_ms(m, d) -> str:
    if m is None:
        return "---"
    v = m.get("by_difficulty", {}).get(d)
    if v is None:
        return "---"
    return f"{v['mean']:.1f}+/-{v['std']:.1f}"


def cg_str(cg) -> str:
    return "---" if cg is None else f"{cg:+.2f}"


# ── Table I — Main performance ────────────────────────────

def table1(results_dir, dataset):
    W = 108
    print(f"\n{'='*W}")
    print("  TABLE I -- Main Performance Results (mean+/-std over seeds, %)")
    print(f"{'='*W}")
    print(
        f"  {'Framework':<22} {'Model':<20} "
        f"{'Easy':>14} {'Medium':>14} {'Hard':>14} "
        f"{'Overall':>14} {'P@k':>8} {'k':>2} {'CG(pp)':>8}"
    )

    tiers = [
        ("Tier 1 -- Single-LLM Prompting Methods", TIER1_FRAMEWORKS),
        ("Tier 2 -- Ensemble",                     TIER2_FRAMEWORKS),
        ("Tier 3 -- Multi-Agent Systems",           TIER3_FRAMEWORKS),
    ]

    for tier_label, fws in tiers:
        print(f"\n  -- {tier_label} --")
        for fw in fws:
            for model in MODEL_NAMES:
                m = get_agg(results_dir, fw, model, dataset)
                if m is None:
                    print(f"  [missing] {fw} x {model}")
                    continue

                base = best_baseline(results_dir, model, dataset)
                cg   = (compute_collaboration_gain(m["overall"]["mean"], base)
                        if fw not in TIER1_FRAMEWORKS else None)

                # Pass@k only for valid frameworks
                pk_str, k_str = "---", "--"
                if fw in PASS_AT_K_MAP:
                    k    = PASS_AT_K_MAP[fw]
                    flat = load_flat(results_dir, fw, model, dataset)
                    if flat:
                        pk = compute_pass_at_k(flat, k=k)
                        pk_str = f"{pk['overall']:.2f}%" if pk["n_total"] > 0 else "---"
                        k_str  = str(k)

                print(
                    f"  {fw:<22} {model:<20} "
                    f"{diff_ms(m,'Easy'):>14} "
                    f"{diff_ms(m,'Medium'):>14} "
                    f"{diff_ms(m,'Hard'):>14} "
                    f"{ms(m):>14} "
                    f"{pk_str:>8} {k_str:>2} "
                    f"{cg_str(cg):>8}"
                )
    print()


# ── Table II -- VI variants vs originals ──────────────────

def table2(results_dir, dataset):
    print(f"\n{'='*88}")
    print("  TABLE II -- Dataset-Aware VI Variants vs Originals (mean delta, pp)")
    print(f"{'='*88}")

    pairs = [
        ("medagents_vi", "medagents"),
        ("mdagents_vi",  "mdagents"),
    ]
    for vi_fw, orig_fw in pairs:
        print(f"\n  {vi_fw} minus {orig_fw}")
        print(
            f"  {'Model':<22} {'Easy delta':>12} {'Med delta':>12} "
            f"{'Hard delta':>12} {'Overall delta':>14} {'McNemar p':>10}"
        )
        for model in MODEL_NAMES:
            m_vi   = get_agg(results_dir, vi_fw,   model, dataset)
            m_orig = get_agg(results_dir, orig_fw, model, dataset)
            if m_vi is None or m_orig is None:
                print(f"  {model:<22} [missing data]")
                continue

            flat_vi   = load_flat(results_dir, vi_fw,   model, dataset)
            flat_orig = load_flat(results_dir, orig_fw, model, dataset)
            mc = (mcnemar_test(flat_vi, flat_orig)
                  if flat_vi and flat_orig else {})

            def delta(d):
                vi_m   = m_vi.get("by_difficulty",   {}).get(d, {}).get("mean", 0)
                orig_m = m_orig.get("by_difficulty", {}).get(d, {}).get("mean", 0)
                return f"{vi_m - orig_m:+.2f}"

            ov_d  = m_vi["overall"]["mean"] - m_orig["overall"]["mean"]
            p_str = f"{mc.get('p_value', 1):.4f}" if mc else "---"

            print(
                f"  {model:<22} "
                f"{delta('Easy'):>12} {delta('Medium'):>12} {delta('Hard'):>12} "
                f"{ov_d:>+14.2f} {p_str:>10}"
            )
    print()


# ── Table III -- Cross-lingual failure mode analysis ──────

def table3(results_dir, dataset):
    print(f"\n{'='*105}")
    print("  TABLE III -- Cross-Lingual Failure Mode Analysis (NMAR/ICR, Tier 3 only)")
    print(f"{'='*105}")
    print(
        f"  {'Framework':<22} {'Model':<20} "
        f"{'NMAR_VI':>8} {'NMAR_EN':>8} {'CLD_N':>7} "
        f"{'ICR_VI':>8} {'ICR_EN':>8} {'CLD_I':>7} {'Fisher_p':>9} {'sig':>4}"
    )
    print(f"  {'-'*103}")

    for model in MODEL_NAMES:
        print(f"\n  Model: {model}")
        baseline_flat = load_flat(results_dir, "zero_shot", model, dataset)

        for fw in MAS_FRAMEWORKS:
            flat = load_flat(results_dir, fw, model, dataset)
            if not flat:
                continue

            # compute_nmar now returns (rate, count, eligible)
            nmar_vi, nmar_count, nmar_eligible = compute_nmar(flat)
            icr_vi = compute_icr(flat, baseline_flat) if baseline_flat else 0.0

            ref    = EN_REFERENCE.get(fw, {})
            nmar_en = ref.get("nmar")
            icr_en  = ref.get("icr")

            cld_n = compute_cld(nmar_vi, nmar_en) if nmar_en is not None else None
            cld_i = compute_cld(icr_vi,  icr_en)  if icr_en  is not None else None

            # Fisher test: use actual eligible count, not total records
            fisher = fisher_test_vs_reference(
                nmar_count, nmar_eligible, nmar_en or 0
            )
            sig_str = "*" if fisher.get("significant") else ""

            print(
                f"  {fw:<22} {model:<20} "
                f"{nmar_vi:>7.1f}% "
                f"{(str(nmar_en)+'%' if nmar_en else '---'):>8} "
                f"{(f'{cld_n:+.1f}' if cld_n is not None else '---'):>7} "
                f"{icr_vi:>7.1f}% "
                f"{(str(icr_en)+'%' if icr_en else '---'):>8} "
                f"{(f'{cld_i:+.1f}' if cld_i is not None else '---'):>7} "
                f"{fisher.get('p_value', '---'):>9.4f} "
                f"{sig_str:>4}"
            )
    print()


# ── Table IV -- Failure mode taxonomy ────────────────────

def table4(results_dir, dataset):
    print(f"\n{'='*65}")
    print("  TABLE IV -- Failure Mode Taxonomy (%, pooled over seeds)")
    print(f"{'='*65}")
    print(f"  {'Framework':<22} {'Model':<20} {'NMAR':>8} {'ICR':>8}")
    print(f"  {'-'*62}")

    for model in MODEL_NAMES:
        baseline_flat = load_flat(results_dir, "zero_shot", model, dataset)
        for fw in MAS_FRAMEWORKS:
            flat = load_flat(results_dir, fw, model, dataset)
            if not flat:
                continue
            nmar, _, _ = compute_nmar(flat)
            icr        = compute_icr(flat, baseline_flat) if baseline_flat else 0.0
            print(f"  {fw:<22} {model:<20} {nmar:>7.1f}% {icr:>7.1f}%")
    print()


# ── Table V -- Per-specialty analysis ────────────────────

def table5(results_dir, dataset, top_n=10):
    print(f"\n{'='*88}")
    print(f"  TABLE V -- Per-Specialty Analysis (top {top_n}, llama-3.3-70b, mean over seeds)")
    print(f"{'='*88}")

    model      = "llama-3.3-70b"
    mdvi_flat  = load_flat(results_dir, "mdagents_vi", model, dataset)

    # Use best single-LLM baseline (max of zero_shot, cot) per paper Section III.E
    base_fw = "zero_shot"
    best    = 0.0
    for fw in ("zero_shot", "cot"):
        m = get_agg(results_dir, fw, model, dataset)
        if m and m["overall"]["mean"] > best:
            best   = m["overall"]["mean"]
            base_fw = fw
    base_flat = load_flat(results_dir, base_fw, model, dataset)

    if not base_flat:
        print(f"  [missing baseline {model}]")
        return

    # Topic frequency
    counts: Counter = Counter()
    for r in base_flat:
        topics = r.get("topic") or ["Unknown"]
        if isinstance(topics, str):
            topics = [topics]
        for t in topics:
            counts[t or "Unknown"] += 1
    top_topics = [t for t, _ in counts.most_common(top_n)]

    def topic_acc(flat, topic):
        sub = [
            r for r in flat
            if (isinstance(r.get("topic"), list) and topic in r["topic"])
            or r.get("topic") == topic
        ]
        if not sub:
            return None, 0
        c = sum(1 for r in sub if r.get("correct"))
        return c / len(sub) * 100, len(sub)

    print(
        f"  {'Specialty':<35} {'Best Prompt':>11} {'MDAgents-VI':>12} "
        f"{'CG(pp)':>8} {'n':>5}"
    )
    print(f"  {'-'*75}")
    for topic in top_topics:
        s_acc, n = topic_acc(base_flat, topic)
        m_acc, _ = topic_acc(mdvi_flat, topic) if mdvi_flat else (None, 0)
        cg = (compute_collaboration_gain(m_acc, s_acc)
              if s_acc is not None and m_acc is not None else None)
        print(
            f"  {topic:<35} "
            f"{(f'{s_acc:.1f}%' if s_acc is not None else '---'):>10} "
            f"{(f'{m_acc:.1f}%' if m_acc is not None else '---'):>12} "
            f"{(f'{cg:+.1f}pp' if cg is not None else '---'):>8} "
            f"{n:>5}"
        )
    print()


# ── Table VI -- Deployment guidelines ────────────────────

def table6(results_dir, dataset):
    print(f"\n{'='*95}")
    print("  TABLE VI -- Condition-Specific Deployment Guidelines")
    print(f"{'='*95}")
    rows = [
        ("Easy + strong model (>70% zero-shot)", "Negative",    "Low",       "1x",   "Zero-shot / CoT"),
        ("Easy + weak model (<55% zero-shot)",   "~Zero",       "Medium",    "3x",   "Voting Ensemble"),
        ("Medium, standard specialty",           "~Zero",       "Medium",    "4-8x", "CoT or MedAgents-VI"),
        ("Hard + strong model",                  "Positive",    "High",      "4-7x", "MDAgents-VI"),
        ("Hard + weak model",                    "Negative",    "Very High", "4-7x", "Zero-shot"),
        ("Eastern Med / Psychiatry",             "Negative",    "Very High", "Any",  "Zero-shot"),
        ("High-stakes, need ensemble",           "Positive",    "Medium",    "3x",   "Voting Ensemble"),
    ]
    print(
        f"  {'Condition':<42} {'CG':>9} {'NMAR risk':>10} "
        f"{'Cost':>6}  {'Recommendation'}"
    )
    print(f"  {'-'*93}")
    for cond, cg, nmar, cost, rec in rows:
        print(f"  {cond:<42} {cg:>9} {nmar:>10} {cost:>6}  {rec}")
    print()


# ── Table VII -- API cost proxy ───────────────────────────

def table7(results_dir, dataset):
    print(f"\n{'='*72}")
    print("  TABLE VII -- API Cost Proxy (avg calls/question, pooled over seeds)")
    print(f"{'='*72}")
    print(f"  {'Framework':<22} {'Model':<20} {'Avg calls':>12} {'Cost x':>8}")
    print(f"  {'-'*65}")

    for fw in ALL_FRAMEWORKS:
        for model in MODEL_NAMES:
            flat = load_flat(results_dir, fw, model, dataset)
            if not flat:
                continue
            calls = [r.get("api_calls", 1) for r in flat]
            avg   = sum(calls) / len(calls)
            print(f"  {fw:<22} {model:<20} {avg:>12.1f} {avg:>7.1f}x")
    print()


# ── Main ──────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="ViMedAgent result analysis")
    parser.add_argument("--dataset",     default="pilot_100",
                        help="Dataset stem (e.g. pilot_100 or vm14k_clean)")
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args()

    print(f"\nViMedAgent Analysis")
    print(f"  Dataset     : {args.dataset}")
    print(f"  Results dir : {args.results_dir}")
    print(f"  Seeds       : {SEEDS}")

    table1(args.results_dir, args.dataset)
    table2(args.results_dir, args.dataset)
    table3(args.results_dir, args.dataset)
    table4(args.results_dir, args.dataset)
    table5(args.results_dir, args.dataset)
    table6(args.results_dir, args.dataset)
    table7(args.results_dir, args.dataset)


if __name__ == "__main__":
    main()
