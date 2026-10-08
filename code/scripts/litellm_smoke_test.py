"""
ViMedAgent v2 — smoke test cho model phục vụ qua LiteLLM (provider "litellm").

Chạy TRƯỚC mỗi đợt thí nghiệm trên một model/server mới. Bắt đúng hai lỗi đã
gặp với Llama-3.3-70B trên GX10 ngày 26/09/2026 — cả hai trả HTTP 200 và
không lỗi ở runner, nên chỉ lộ ra khi đọc nội dung:
  - chat template sai → model viết tiếp prompt thay vì trả lời
  - radix cache hỏng  → lần gọi ĐẦU đúng, các lần lặp cùng prefix ra rác

Kiểm tra:
  1. /v1/models có model_id mà config.py mong đợi
  2. 3 prompt có đáp án xác định × 3 vòng lặp, T=0 → phải đúng và GIỐNG NHAU mọi vòng
  3. prompt zero_shot thật trên N câu pilot → tỉ lệ parse được 1 ký tự
  4. thông lượng: CONC request CoT song song → token/s tổng, số câu cắt ở max_tokens

Usage (trong env có openai, sau khi nạp .env):
    set -a; . /home/ailab/A2I/.env; set +a
    python scripts/litellm_smoke_test.py --model litellm-qwen2.5-72b
"""
import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from infrastructure.config import MODELS, PILOT_FILE, get_api_key, get_base_url
from infrastructure.llm_client import parse_answer
from frameworks.tier1.zero_shot import build_prompt as zs_prompt
from frameworks.tier1.cot import build_prompt as cot_prompt

# (tên, messages, hàm kiểm đáp án) — câu trả lời ngắn, xác định, không phụ thuộc ngôn ngữ
PROBES = [
    ("2+2", [{"role": "user", "content": "What is 2+2? Reply with the number only."}],
     lambda t: t.strip().rstrip(".") == "4"),
    ("paris+system", [{"role": "system", "content": "You are a medical expert."},
                      {"role": "user", "content": "What is the capital of France? Answer in one word."}],
     lambda t: t.strip().rstrip(".").lower() == "paris"),
    ("zero_shot VI", zs_prompt("Thủ đô của Việt Nam là?", ["Huế", "Hà Nội", "Đà Nẵng", "Sài Gòn"]),
     lambda t: parse_answer(t) == "B"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--n-pilot", type=int, default=20)
    ap.add_argument("--concurrency", type=int, default=32)
    args = ap.parse_args()

    cfg = MODELS[args.model]
    assert cfg["provider"] == "litellm", "script này cho model provider=litellm"
    from openai import OpenAI
    client = OpenAI(api_key=get_api_key("litellm"), base_url=get_base_url("litellm"), timeout=900)
    mid = cfg["model_id"]
    fail = False

    served = [m.id for m in client.models.list().data]
    print(f"[1] {get_base_url('litellm')} phục vụ: {served}")
    if mid not in served:
        print(f"    !! model_id {mid!r} không có — sửa tên model group trong LiteLLM hoặc config.py")
        sys.exit(1)

    print(f"[2] {len(PROBES)} prompt × {args.rounds} vòng, T=0 (bắt chat template + radix cache)")
    for name, msgs, ok in PROBES:
        outs = []
        for rnd in range(args.rounds):
            r = client.chat.completions.create(model=mid, messages=msgs, temperature=0, max_tokens=20)
            outs.append((r.choices[0].message.content or "", r.choices[0].finish_reason, r.usage.prompt_tokens))
        good = all(ok(o[0]) for o in outs)
        same = len({o[0] for o in outs}) == 1
        fail |= not (good and same)
        print(f"    {'OK ' if good and same else '!! '} {name:13s} pt={outs[0][2]:<4} "
              f"{[o[0][:30] for o in outs]} finish={[o[1] for o in outs]}")
        if good and not same:
            print("       !! đúng nhưng khác nhau giữa các vòng ở T=0 — nghi radix cache")

    samples = [json.loads(l) for l in open(PILOT_FILE, encoding="utf-8")]
    n = min(args.n_pilot, len(samples))
    def zs(s):
        r = client.chat.completions.create(model=mid, messages=zs_prompt(s["question"], s["options"]),
                                           temperature=0.3, max_tokens=16)
        return parse_answer(r.choices[0].message.content)
    with ThreadPoolExecutor(args.concurrency) as p:
        ans = list(p.map(zs, samples[:n]))
    nulls = sum(a is None for a in ans)
    fail |= nulls > n // 10
    print(f"[3] zero_shot {n} câu pilot: đúng {sum(a == s['answer'] for a, s in zip(ans, samples))}/{n} | null {nulls}")

    k = min(args.concurrency, len(samples))
    def cot(s):
        r = client.chat.completions.create(model=mid, messages=cot_prompt(s["question"], s["options"]),
                                           temperature=0.3, max_tokens=512)
        return r.usage.completion_tokens, r.choices[0].finish_reason, parse_answer(r.choices[0].message.content)
    t = time.time()
    with ThreadPoolExecutor(k) as p:
        out = list(p.map(cot, samples[:k]))
    dt = time.time() - t
    tok = sum(o[0] for o in out)
    print(f"[4] cot {k} song song: {dt:.0f}s | {tok/dt:.0f} tok/s tổng | "
          f"cắt ở 512: {sum(o[1] == 'length' for o in out)} | null {sum(o[2] is None for o in out)}")

    print("FAIL — đừng chạy benchmark" if fail else "OK")
    sys.exit(1 if fail else 0)


if __name__ == "__main__":
    main()
