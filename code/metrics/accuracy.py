"""Pass@1, Pass@k, tổng hợp qua seed, Collaboration Gain."""

from __future__ import annotations

from collections import defaultdict

from infrastructure.config import TIER1_FRAMEWORKS
from .io import Cell, DIFFICULTY_ORDER, OPTIONS
from .stats import mean_std

#: v6 Section III-F chú thích CG so với max(Zero-shot, CoT)
CG_BASELINE_PAPER = ("zero_shot", "cot")

#: Pass@k chỉ hợp lệ khi individual_votes là các lần thử ĐỘC LẬP
PASS_AT_K = {"cot_sc": 5, "voting_ensemble": 3}


def pass_at_1(records: list[dict]) -> dict:
    by_diff = defaultdict(lambda: [0, 0])
    n = c = 0
    for r in records:
        ok = r["_correct"]
        n += 1
        c += ok
        d = by_diff[r["_difficulty"]]
        d[0] += 1
        d[1] += ok
    return {"overall": c / n * 100 if n else 0.0, "n": n, "n_correct": c,
            "by_difficulty": {k: {"acc": v[1] / v[0] * 100 if v[0] else 0.0,
                                  "n": v[0]} for k, v in by_diff.items()}}


def pass_at_k(records: list[dict], k: int) -> dict:
    n = c = 0
    for r in records:
        votes = [v for v in (r.get("individual_votes") or []) if v in OPTIONS]
        if not votes or r.get("gold") is None:
            continue
        n += 1
        c += any(v == r["gold"] for v in votes[:k])
    return {"pass_at_k": c / n * 100 if n else 0.0, "k": k, "n": n}


def cell_accuracy(cell: Cell) -> dict | None:
    per_seed = cell.per_seed(only_complete=True)
    if not per_seed:
        return None
    stats = [pass_at_1(rs) for rs in per_seed]
    m, s = mean_std(x["overall"] for x in stats)
    out = {"framework": cell.framework, "model": cell.model,
           "n_seeds": len(per_seed), "n_per_seed": stats[0]["n"],
           "overall_mean": m, "overall_std": s, "by_difficulty": {}}
    for d in DIFFICULTY_ORDER:
        vals = [x["by_difficulty"][d]["acc"] for x in stats if d in x["by_difficulty"]]
        if vals:
            dm, ds = mean_std(vals)
            out["by_difficulty"][d] = {
                "mean": dm, "std": ds,
                "n": stats[0]["by_difficulty"].get(d, {}).get("n", 0)}
    k = PASS_AT_K.get(cell.framework)
    if k:
        km, ks = mean_std(pass_at_k(rs, k)["pass_at_k"] for rs in per_seed)
        out["pass_at_k"] = {"k": k, "mean": km, "std": ks}
    return out


def avg_api_calls(cell: Cell) -> float | None:
    recs = cell.flat(True) or cell.flat(False)
    vals = [r["api_calls"] for r in recs if r.get("api_calls") is not None]
    return sum(vals) / len(vals) if vals else None


def cg_baselines(acc_by_cell: dict, model: str) -> dict:
    """Hai mốc CG.

    v6 viết 'best Tier-1' nhưng chú thích 'max(Zero-shot, CoT)' — khác nhau
    vì Tier 1 còn CoT-SC và Self-Refine. Trả cả hai; caller so sánh và gắn cờ.
    """
    def best_of(fws):
        cand = [(fw, a["overall_mean"]) for (fw, m), a in acc_by_cell.items()
                if m == model and fw in fws and a]
        return max(cand, key=lambda x: x[1]) if cand else (None, None)

    fw_p, v_p = best_of(CG_BASELINE_PAPER)
    fw_a, v_a = best_of(TIER1_FRAMEWORKS)
    return {"paper_fw": fw_p, "paper_val": v_p,
            "all_t1_fw": fw_a, "all_t1_val": v_a}
