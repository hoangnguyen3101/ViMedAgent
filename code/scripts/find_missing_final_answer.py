"""
Lọc các câu mà output của model KHÔNG có dòng "Final answer: X".

"Thiếu" được định nghĩa bằng đúng regex mà parse_answer() dùng làm ưu tiên đầu
(infrastructure/llm_client.py) — nên đây chính là những câu mà đáp án phải lấy
từ fallback yếu (**X**, (X), chữ cái đứng một mình cuối cùng…), hoặc ra None.

Đọc audit_trail[*].raw của file kết quả (runner ghi raw cho mọi framework từ
26/09/2026). Framework nhiều mẫu (cot_sc, voting_ensemble) được xét TỪNG mẫu.
Chỉ đọc, không sửa file kết quả — an toàn cả khi file đang được runner ghi tiếp.

Usage (chạy từ code/):
    python scripts/find_missing_final_answer.py \
        results/tier1/cot/cot__litellm-llama3_3-70b-awq__vm14k_clean__s42.jsonl
    # tuỳ chọn: --out <file.json>  --dataset data/vm14k_clean.jsonl  --max-tokens 512
"""
import argparse
import json
import os
import re
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from infrastructure.config import MAX_TOKENS_LONG
from infrastructure.llm_client import parse_answer

# Giữ đồng bộ với parse_answer() — nhánh "Final answer: B".
FINAL_RE = re.compile(r'final\s+answer[:\s]*\**\s*([ABCD])\b', re.IGNORECASE)
THINK_RE = re.compile(r'<think>.*?</think>', re.DOTALL)


def has_final_answer(raw: str) -> bool:
    return bool(raw) and bool(FINAL_RE.search(THINK_RE.sub('', raw)))


def load_jsonl(path: str) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass    # dòng cuối có thể đang được runner ghi dở
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results", help="file kết quả .jsonl")
    ap.add_argument("--out", help="file .json đầu ra (mặc định: results_audit/<tên file>__missing_final_answer.json)")
    ap.add_argument("--dataset", default="data/vm14k_clean.jsonl",
                    help="để gắn kèm câu hỏi + đáp án (bỏ qua nếu không có)")
    ap.add_argument("--max-tokens", type=int, default=MAX_TOKENS_LONG,
                    help="tokens_out >= giá trị này coi là bị cắt (mặc định MAX_TOKENS_LONG)")
    args = ap.parse_args()

    rows = load_jsonl(args.results)
    questions = {}
    if args.dataset and os.path.exists(args.dataset):
        questions = {r["id"]: r for r in load_jsonl(args.dataset)}

    missing, n_steps, no_raw = [], 0, 0
    for r in rows:
        steps = [s for s in (r.get("audit_trail") or []) if isinstance(s, dict) and "raw" in s]
        if not steps:
            no_raw += 1
            continue
        bad = []
        for i, s in enumerate(steps):
            n_steps += 1
            if not has_final_answer(s.get("raw") or ""):
                bad.append({
                    "step_index": i,
                    "step": s.get("step"),
                    "parsed_by_fallback": parse_answer(s.get("raw") or ""),
                    "raw": s.get("raw"),
                })
        if not bad:
            continue
        q = questions.get(r["id"], {})
        missing.append({
            "id": r["id"],
            "seed": r.get("seed"),
            "framework": r.get("framework"),
            "model": r.get("model"),
            "difficulty": r.get("difficulty"),
            "topic": r.get("topic"),
            "gold": r.get("gold"),
            "answer": r.get("answer"),
            "correct": r.get("correct"),
            "tokens_out": r.get("tokens_out"),
            # tokens_out là tổng mọi lời gọi của câu — chỉ đúng nghĩa "bị cắt" khi 1 lời gọi/câu
            "truncated": (r.get("n_llm_calls") == 1 and (r.get("tokens_out") or 0) >= args.max_tokens),
            "question": q.get("question"),
            "options": q.get("options"),
            "steps_missing_final_answer": bad,
        })

    out = args.out
    if not out:
        stem = os.path.basename(args.results).removesuffix(".jsonl")
        out = os.path.join("results_audit", f"{stem}__missing_final_answer.json")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)

    summary = {
        "source": os.path.abspath(args.results),
        "n_records": len(rows),
        "n_records_without_raw": no_raw,
        "n_steps_checked": n_steps,
        "n_records_missing_final_answer": len(missing),
        "n_truncated": sum(m["truncated"] for m in missing),
        "n_answer_none": sum(m["answer"] is None for m in missing),
        "accuracy_of_missing": (sum(bool(m["correct"]) for m in missing) / len(missing)) if missing else None,
        "by_difficulty": dict(Counter(m["difficulty"] for m in missing)),
    }
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "records": missing}, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if no_raw:
        print(f"!! {no_raw} bản ghi không có raw (chạy trước khi runner lưu raw) — không kiểm được")
    print(f"→ {out}")


if __name__ == "__main__":
    main()
