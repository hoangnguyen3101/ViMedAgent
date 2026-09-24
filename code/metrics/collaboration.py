"""NMAR và ICR.

NMAR suy biến với framework lấy đa số CỨNG (voting_ensemble, cot_sc): đáp án
cuối luôn bằng phe đa số theo định nghĩa thuật toán nên NMAR = 100% bất kể
model. Hàm tự phát hiện và trả cờ `degenerate`; caller có trách nhiệm không
trình bày giá trị suy biến như kết quả thực nghiệm.
"""

from __future__ import annotations

from collections import Counter

from .io import OPTIONS, voting_answers
from .stats import pct


def _eligible(r: dict):
    """Trả (votes, majority) nếu câu đủ điều kiện xét NMAR, ngược lại None."""
    votes = voting_answers(r)
    if len(votes) < 2 or r.get("answer") not in OPTIONS or r.get("gold") is None:
        return None
    return votes, Counter(votes).most_common(1)[0][0]


def nmar(records: list[dict]) -> dict:
    hits = eligible = follow = cases = 0
    for r in records:
        e = _eligible(r)
        if e is None:
            continue
        votes, majority = e
        cases += 1
        follow += (r["answer"] == majority)
        if r["gold"] in votes and majority != r["gold"]:
            eligible += 1
            if r["answer"] == majority and r["answer"] != r["gold"]:
                hits += 1
    return {"nmar": pct(hits, eligible), "hits": hits, "eligible": eligible,
            "majority_follow_rate": pct(follow, cases),
            "degenerate": cases > 0 and follow == cases,
            "n_with_votes": cases}


def nmar_flags(records: list[dict]) -> dict:
    """{(id, seed): bool} trên tập đủ điều kiện — cho McNemar ở Table VI-a."""
    out = {}
    for r in records:
        e = _eligible(r)
        if e is None:
            continue
        votes, majority = e
        if r["gold"] in votes and majority != r["gold"]:
            out[r["_key"]] = (r["answer"] == majority and r["answer"] != r["gold"])
    return out


def _zs_index(zero_shot: list[dict]):
    exact = {r["_key"]: r for r in zero_shot}
    by_id = {}
    for r in sorted(zero_shot, key=lambda x: x.get("seed") or 0):
        by_id.setdefault(r.get("id"), r)
    return exact, by_id


def icr(mas: list[dict], zero_shot: list[dict]) -> dict:
    """Tỷ lệ câu mà đáp án cuối của MAS trùng zero_shot cùng model.

    Ghép theo (id, seed); nếu thiếu seed tương ứng thì lùi về ghép theo id
    và đếm số lần phải lùi, để người đọc biết ICR tính trên cơ sở nào.
    """
    exact, by_id = _zs_index(zero_shot)
    same = total = fallback = 0
    for m in mas:
        b = exact.get(m["_key"])
        if b is None:
            b = by_id.get(m.get("id"))
            if b is None:
                continue
            fallback += 1
        total += 1
        same += (m.get("answer") == b.get("answer"))
    return {"icr": pct(same, total), "matched": total, "seed_fallback": fallback}


def icr_flags(mas: list[dict], zero_shot: list[dict]) -> dict:
    exact, by_id = _zs_index(zero_shot)
    out = {}
    for m in mas:
        b = exact.get(m["_key"]) or by_id.get(m.get("id"))
        if b is not None:
            out[m["_key"]] = (m.get("answer") == b.get("answer"))
    return out
