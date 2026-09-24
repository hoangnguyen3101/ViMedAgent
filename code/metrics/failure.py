"""Bốn chỉ số failure mode (paper v6 Section III-F, Table III).

Hai mẫu số khác nhau, có chủ đích:
  - Flawed Consensus / Minority Suppression / Information Loss nói về quan hệ
    giữa NHIỀU agent, không định nghĩa được khi câu chỉ có 1 agent trả lời.
    Chia cho `n_wrong_multi_agent`. (MDAgents có ~82% câu đi nhánh PCC một
    agent; chia cho toàn bộ câu sai sẽ đẩy flawed consensus lên >90% thuần
    tạo tác số học, vì một agent thì luôn là "đa số".)
  - Ineffective Collaboration và Overall Fail chia cho toàn bộ.

Information Loss là CẬN DƯỚI: định nghĩa của paper gồm cả "đáp án đúng được
nêu trong reasoning trace", nhưng log không lưu `raw` nên chỉ tính được phần
"đáp án đúng nằm trong đáp án pre-synthesis của agent".
"""

from __future__ import annotations

from collections import Counter

from .io import OPTIONS, pre_synthesis_answers
from .stats import pct

INFO_LOSS_NOTE = ("cận dưới — chỉ tính đáp án agent; "
                  "reasoning trace không được lưu trong log")


def failure_modes(records: list[dict], zero_shot: list[dict]) -> dict:
    zs = {r["_key"]: r.get("answer") for r in zero_shot}
    zs_by_id = {}
    for r in sorted(zero_shot, key=lambda x: x.get("seed") or 0):
        zs_by_id.setdefault(r.get("id"), r.get("answer"))

    n = len(records)
    n_wrong = n_multi = n_any = 0
    flawed = suppressed = ineffective = info_loss = 0

    for r in records:
        if r.get("gold") is None or r["_correct"]:
            continue
        n_wrong += 1
        final, gold = r.get("answer"), r["gold"]

        agents = pre_synthesis_answers(r)
        if agents:
            n_any += 1
        if len(agents) >= 2:
            n_multi += 1
            majority, k = Counter(agents).most_common(1)[0]
            if final in OPTIONS and majority == final and k * 2 > len(agents):
                flawed += 1
            if gold in agents:
                suppressed += 1
                info_loss += 1

        z = zs.get(r["_key"], zs_by_id.get(r.get("id")))
        if z is not None and z == final:
            ineffective += 1

    return {
        "n": n, "n_wrong": n_wrong,
        "n_wrong_multi_agent": n_multi,
        "n_wrong_with_agent_answers": n_any,
        "multi_agent_share_of_wrong": pct(n_multi, n_wrong),
        "overall_fail_pct": pct(n_wrong, n),
        "flawed_consensus_pct": pct(flawed, n_multi),
        "minority_suppression_pct": pct(suppressed, n_multi),
        "info_loss_pct": pct(info_loss, n_multi),
        "ineffective_collab_pct": pct(ineffective, n_wrong),
        "info_loss_note": INFO_LOSS_NOTE,
    }
