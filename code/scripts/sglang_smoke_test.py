"""
ViMedAgent v2 — SGLang smoke test.

Checks, against the running server (SGLANG_BASE_URL):
  1. /v1/models lists the model_id the config expects
  2. think=False really disables Qwen3 thinking (no <think> block, short reply)
  3. the exact zero_shot prompt returns a single A/B/C/D on 3 real pilot questions
  4. throughput: N concurrent zero_shot calls (default 32) → requests/sec

Usage:
    python scripts/sglang_smoke_test.py --model sglang-qwen3-32b [--concurrency 32]
"""
import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from infrastructure.config import MODELS, get_base_url, PILOT_FILE
from infrastructure.llm_client import call_llm, parse_answer
from frameworks.tier1.zero_shot import build_prompt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="sglang-qwen3-32b")
    ap.add_argument("--concurrency", type=int, default=32)
    args = ap.parse_args()

    cfg = MODELS[args.model]
    assert cfg["provider"] == "sglang", "smoke test is for sglang-* models"
    base = get_base_url("sglang")

    # 1. server up + model name matches
    from openai import OpenAI
    client = OpenAI(api_key="sglang", base_url=base, timeout=60)
    served = [m.id for m in client.models.list().data]
    print(f"[1] server {base} serves: {served}")
    if cfg["model_id"] not in served:
        print(f"    !! config model_id {cfg['model_id']!r} not served — fix MODEL= in serve_sglang.sh or config.py")
        sys.exit(1)

    # 2. thinking really off
    msgs = [{"role": "user", "content": "What is 2+2? Reply with the number only."}]
    t0 = time.time()
    off = call_llm(args.model, msgs, max_tokens=64, think=False)
    dt_off = time.time() - t0
    print(f"[2] think=False → {off!r}  ({dt_off:.1f}s)  has_think_tag={'<think>' in off}")
    if "<think>" in off:
        print("    !! <think> block leaked with think=False — chat_template_kwargs not honoured")
        sys.exit(1)

    # 3. zero_shot prompt on real pilot questions
    samples = [json.loads(l) for l in open(PILOT_FILE, encoding="utf-8")][:3]
    for s in samples:
        raw = call_llm(args.model, build_prompt(s["question"], s["options"]), think=False)
        print(f"[3] {s['id']}: raw={raw[:40]!r} parsed={parse_answer(raw)} gold={s['answer']}")

    # 4. throughput
    samples = [json.loads(l) for l in open(PILOT_FILE, encoding="utf-8")][:args.concurrency]
    def one(s):
        return parse_answer(call_llm(args.model, build_prompt(s["question"], s["options"]), think=False))
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        outs = list(pool.map(one, samples))
    dt = time.time() - t0
    nulls = sum(o is None for o in outs)
    print(f"[4] {len(samples)} zero_shot calls @ concurrency={args.concurrency}: "
          f"{dt:.1f}s → {len(samples)/dt:.2f} req/s | null={nulls}")
    print("OK")


if __name__ == "__main__":
    main()
