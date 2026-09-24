"""Kiểm tra tiền đề trước khi tin bất kỳ bảng nào.

Ba loại kiểm tra, mỗi loại chặn một cách hiểu sai cụ thể:

1. `answer_distribution` / `flag_parser_skew` — phát hiện lỗi trích xuất đáp án.
   experiment/001-tier1-fix-parse-answer đã chứng minh một lỗi regex đẩy đáp án
   lệch ~+11pp mà Pass@1 vẫn trông hợp lý. Mốc so sánh là zero_shot cùng model:
   nó sinh đúng 1 ký tự nên không đi qua regex fallback, phần lệch của nó là
   position bias thật. Chữ ký quyết định là PRECISION ĐẢO NGƯỢC — đáp án được
   dự đoán nhiều nhất lại là đáp án đúng ít nhất.

2. `nmar_is_degenerate` — xem metrics/collaboration.py.

3. `gs_applicability` — +GS chỉ tính lại offline được khi bước cuối THỰC SỰ là
   bỏ phiếu. Kiểm bằng cách dựng lại phiếu với trọng số đều rồi so với đáp án
   đã lưu; khớp thấp nghĩa là có synthesizer LLM ở bước cuối, thay bằng
   weighted vote sẽ ĐỔI THUẬT TOÁN chứ không phải thêm một repair.
"""

from __future__ import annotations

from collections import Counter

from .io import OPTIONS, agent_roles

#: dưới ngưỡng này phân bố đáp án là nhiễu thuần, không chẩn đoán được
MIN_RECORDS = 500
#: phần lệch vượt quá mốc tự nhiên bao nhiêu thì đáng nghi (pp)
EXCESS_SKEW_PP = 8.0
#: tỷ lệ tái tạo phiếu tối thiểu để coi bước cuối là bỏ phiếu thuần
VOTE_RECONSTRUCTION_MIN = 0.95


def answer_distribution(records: list[dict]) -> dict:
    n = len(records)
    if n == 0:
        return {"n": 0}
    gold = Counter(r.get("gold") for r in records)
    pred = Counter(r.get("answer") for r in records)
    prec = {}
    for L in OPTIONS:
        sel = [r for r in records if r.get("answer") == L]
        prec[L] = (sum(1 for r in sel if r.get("gold") == L) / len(sel) * 100
                   if sel else None)
    pred_pct = {L: pred.get(L, 0) / n * 100 for L in OPTIONS}
    gold_pct = {L: gold.get(L, 0) / n * 100 for L in OPTIONS}
    skew = {L: pred_pct[L] - gold_pct[L] for L in OPTIONS}
    top = max(OPTIONS, key=lambda L: pred_pct[L])
    have = {L: v for L, v in prec.items() if v is not None}
    worst = min(have, key=have.get) if have else None
    return {
        "n": n,
        "null_pct": sum(1 for r in records if r.get("answer") is None) / n * 100,
        "pass1": sum(1 for r in records if r["_correct"]) / n * 100,
        "pred_pct": pred_pct, "gold_pct": gold_pct, "skew": skew,
        "precision": prec,
        "max_skew_option": max(OPTIONS, key=lambda L: skew[L]),
        "max_skew_pp": max(skew.values()),
        "top_pred_option": top,
        "inverted_precision": top == worst,
    }


def flag_parser_skew(dist: dict, natural_skew_pp: float | None) -> tuple[str, str]:
    """Trả (mức cảnh báo, lý do). natural_skew_pp lấy từ zero_shot cùng model."""
    if dist.get("n", 0) == 0:
        return "no data", "không có bản ghi"
    if dist["n"] < MIN_RECORDS:
        return "too small", f"chỉ {dist['n']} bản ghi — chưa chẩn đoán được"

    reasons = []
    if dist["null_pct"] >= 2.0:
        reasons.append(f"null {dist['null_pct']:.1f}%")
    base = natural_skew_pp if natural_skew_pp is not None else 5.0
    excess = dist["max_skew_pp"] - base
    if excess >= EXCESS_SKEW_PP:
        reasons.append(f"lệch về '{dist['max_skew_option']}' "
                       f"+{dist['max_skew_pp']:.1f}pp (mốc tự nhiên {base:.1f}pp)")
        if dist["inverted_precision"]:
            reasons.append("precision đảo ngược")
            return "SUSPECT", "; ".join(reasons)
    return ("WARN", "; ".join(reasons)) if reasons else ("ok", "")


def gs_applicability(records: list[dict], reconstruct_fn, sample: int = 3000) -> dict:
    """reconstruct_fn(record, down_weight=1.0) -> đáp án dựng lại với trọng số đều."""
    checked = match = with_roles = 0
    for r in records[:sample]:
        from .io import voting_answers
        if len(voting_answers(r)) < 2:
            continue
        checked += 1
        match += (reconstruct_fn(r, 1.0) == r.get("answer"))
        with_roles += bool(agent_roles(r))

    rate = match / checked if checked else 0.0
    role_rate = with_roles / checked if checked else 0.0
    reasons = []
    if checked == 0:
        reasons.append("không có câu nào có >= 2 phiếu agent")
    else:
        if rate < VOTE_RECONSTRUCTION_MIN:
            reasons.append(f"bước cuối không phải bỏ phiếu thuần "
                           f"(tái tạo khớp {rate*100:.1f}%)")
        if role_rate < 0.5:
            reasons.append(f"không suy được chuyên khoa của agent "
                           f"({role_rate*100:.1f}% câu có roles)")
    return {"applicable": not reasons, "reason": "; ".join(reasons),
            "vote_reconstruction_pct": round(rate * 100, 1),
            "roles_available_pct": round(role_rate * 100, 1),
            "n_checked": checked}


def has_confidence(records: list[dict]) -> bool:
    """+CW cần agent tự chấm confidence; trường này chưa tồn tại trong log."""
    for r in records[:200]:
        if r.get("agent_confidences"):
            return True
        if any("confidence" in (s or {}) for s in (r.get("audit_trail") or [])):
            return True
    return False
