"""
ViMedAgent -- Metrics
Pass@1, Pass@k, mean+/-std (sample std), CG, NMAR, ICR, CLD, Fisher, McNemar
"""

import math
from collections import defaultdict, Counter
from scipy.stats import binomtest, fisher_exact


# ── Pass@1 ────────────────────────────────────────────────
def compute_pass_at_1(results: list) -> dict:
    """
    Standard accuracy. Works for all frameworks.
    Uses 'answer' (final answer) vs 'gold'.
    """
    total, correct = 0, 0
    by_diff  = defaultdict(lambda: {"total": 0, "correct": 0})
    by_topic = defaultdict(lambda: {"total": 0, "correct": 0})

    for r in results:
        is_correct = (r.get("answer") == r.get("gold"))
        d      = _norm_diff(r.get("difficulty", "Unknown"))
        topics = r.get("topic") or ["Unknown"]
        if isinstance(topics, str):
            topics = [topics] if topics else ["Unknown"]
        topics = [t or "Unknown" for t in topics]

        total   += 1
        correct += is_correct
        by_diff[d]["total"]   += 1
        by_diff[d]["correct"] += is_correct
        for t in topics:
            by_topic[t]["total"]   += 1
            by_topic[t]["correct"] += is_correct

    return {
        "overall":       _pct(correct, total),
        "n_total":       total,
        "n_correct":     correct,
        "by_difficulty": {k: {"acc": _pct(v["correct"], v["total"]), **v}
                          for k, v in by_diff.items()},
        "by_topic":      {k: {"acc": _pct(v["correct"], v["total"]), **v}
                          for k, v in by_topic.items()},
    }


# ── Pass@k ────────────────────────────────────────────────
def compute_pass_at_k(results: list, k: int) -> dict:
    """
    Pass@k: fraction of questions where at least 1 of k independent
    attempts is correct.

    ONLY valid for:
      cot_sc:          individual_votes = k CoT sampling paths (k=5)
      voting_ensemble: individual_votes = k shuffled runs (k=3)

    For MAS frameworks, individual_votes = agent votes within 1 run
    and must NOT be used for Pass@k.

    n_total=0 signals no valid data -- caller should display "---".
    """
    total, correct = 0, 0
    by_diff = defaultdict(lambda: {"total": 0, "correct": 0})

    for r in results:
        votes = r.get("individual_votes") or []
        gold  = r.get("gold")
        if not votes or gold is None:
            continue
        valid = [v for v in votes if v in ("A", "B", "C", "D")]
        if not valid:
            continue

        any_correct = any(v == gold for v in valid[:k])
        d = _norm_diff(r.get("difficulty", "Unknown"))

        total   += 1
        correct += any_correct
        by_diff[d]["total"]   += 1
        by_diff[d]["correct"] += any_correct

    return {
        "overall":       _pct(correct, total),
        "n_total":       total,
        "n_correct":     correct,
        "by_difficulty": {dk: {"acc": _pct(v["correct"], v["total"]), **v}
                          for dk, v in by_diff.items()},
    }


# ── Mean +/- Std across seeds ─────────────────────────────
def compute_mean_std(per_seed_metrics: list) -> dict:
    """
    Aggregate per-seed Pass@1 dicts into mean +/- std.

    Uses sample standard deviation (divides by n-1, not n) consistent
    with MDAgents and MedAgentBoard reporting convention for n=3 seeds.
    Falls back to std=0 when n=1 (cannot compute sample std).

    per_seed_metrics: list of compute_pass_at_1() return dicts.
    Returns dict with same structure but each numeric value replaced by
    {"mean": float, "std": float}.
    """
    if not per_seed_metrics:
        return {}

    n = len(per_seed_metrics)

    def _agg(values: list) -> dict:
        mean = sum(values) / n
        # Sample std (n-1): correct statistical convention for small n
        if n > 1:
            std = math.sqrt(sum((v - mean) ** 2 for v in values) / (n - 1))
        else:
            std = 0.0
        return {"mean": round(mean, 4), "std": round(std, 4)}

    overall_vals = [m["overall"] for m in per_seed_metrics]

    # Aggregate by_difficulty
    all_diffs = set()
    for m in per_seed_metrics:
        all_diffs.update(m.get("by_difficulty", {}).keys())
    by_diff_agg = {}
    for d in sorted(all_diffs):
        vals = [m["by_difficulty"].get(d, {}).get("acc", 0.0)
                for m in per_seed_metrics]
        by_diff_agg[d] = _agg(vals)

    return {
        "overall":       _agg(overall_vals),
        "n_seeds":       n,
        "n_total":       per_seed_metrics[0].get("n_total", 0),
        "by_difficulty": by_diff_agg,
    }


# ── Collaboration Gain ────────────────────────────────────
def compute_collaboration_gain(mas_acc: float, baseline_acc: float) -> float:
    """CG = Acc(MAS) - Acc(best no-communication method), in pp."""
    return round(mas_acc - baseline_acc, 4)


# ── NMAR ─────────────────────────────────────────────────
def compute_nmar(results: list) -> tuple:
    """
    Negative Majority Assimilation Rate.
    Among questions where the correct answer is a minority opinion,
    the fraction where the final answer followed the (wrong) majority.

    Returns (rate_pct, nmar_count, eligible_count) so callers can
    use the counts directly for Fisher's exact test.
    """
    nmar_count = 0
    eligible   = 0

    for r in results:
        votes = (r.get("individual_votes")
                 or r.get("round2_answers")
                 or r.get("final_votes") or [])
        final = r.get("answer")
        gold  = r.get("gold")
        if not votes or final is None or gold is None:
            continue
        valid = [v for v in votes if v in ("A", "B", "C", "D")]
        if not valid:
            continue

        majority         = Counter(valid).most_common(1)[0][0]
        minority_correct = (gold in valid) and (majority != gold)
        if minority_correct:
            eligible += 1
            if final == majority and final != gold:
                nmar_count += 1

    rate = _pct(nmar_count, eligible)
    return rate, nmar_count, eligible


# ── ICR ───────────────────────────────────────────────────
def compute_icr(mas_results: list, baseline_results: list) -> float:
    """
    Ineffective Collaboration Rate.
    Fraction of questions where MAS final answer == zero_shot answer.
    Matched by question ID.
    """
    base_by_id = {r["id"]: r for r in baseline_results if "id" in r}
    same, total = 0, 0
    for m in mas_results:
        b = base_by_id.get(m.get("id"))
        if b is None:
            continue
        total += 1
        if m.get("answer") == b.get("answer"):
            same += 1
    return _pct(same, total)


# ── CLD ───────────────────────────────────────────────────
def compute_cld(vi_rate: float, en_rate: float) -> float:
    """Cross-Lingual Delta = VI_rate - EN_reference. Positive = amplified."""
    return round(vi_rate - en_rate, 4)


# ── Fisher's exact test for CLD significance ─────────────
def fisher_test_vs_reference(
    vi_count: int,
    vi_eligible: int,
    en_rate_pct: float,
) -> dict:
    """
    Test whether VI NMAR rate differs significantly from EN reference.
    Uses Fisher's exact test with a synthetic EN 2x2 table of equal size.

    Args:
        vi_count:    number of NMAR cases in VI
        vi_eligible: number of eligible cases in VI (denominator of NMAR)
        en_rate_pct: EN reference NMAR rate in percent (e.g. 18.3)
    """
    if vi_eligible == 0 or en_rate_pct is None:
        return {"p_value": 1.0, "significant": False}

    en_count = round((en_rate_pct / 100.0) * vi_eligible)
    table    = [
        [vi_count,               vi_eligible - vi_count],
        [en_count,               vi_eligible - en_count],
    ]
    _, p_value = fisher_exact(table)
    return {
        "p_value":     round(float(p_value), 6),
        "significant": p_value < 0.05,
    }


# ── McNemar test ──────────────────────────────────────────
def mcnemar_test(results_a: list, results_b: list) -> dict:
    """
    Exact McNemar test on matched pairs (by id).
    b = A wrong, B correct; c = A correct, B wrong.
    """
    b_by_id = {r["id"]: r for r in results_b if "id" in r}
    b_count, c_count = 0, 0
    for a in results_a:
        b = b_by_id.get(a.get("id"))
        if b is None:
            continue
        ca = (a.get("answer") == a.get("gold"))
        cb = (b.get("answer") == b.get("gold"))
        if not ca and cb:
            b_count += 1
        elif ca and not cb:
            c_count += 1
    n = b_count + c_count
    if n == 0:
        return {"b": 0, "c": 0, "p_value": 1.0, "significant": False}
    p_value = float(binomtest(b_count, n, 0.5, alternative="two-sided").pvalue)
    return {
        "b": b_count, "c": c_count,
        "p_value": round(p_value, 6), "significant": p_value < 0.05,
    }


# ── Helpers ───────────────────────────────────────────────
def _norm_diff(d: str) -> str:
    return "Hard" if d in ("Challenging", "Hard") else (d or "Unknown")


def _pct(num: int, den: int) -> float:
    return num / den * 100 if den else 0.0
