"""
ViMedAgent v2 — LLM Client (Ollama local + SGLang)

Ollama: local API via http://localhost:11434/v1  (OpenAI-compatible) or the
        native /api/chat endpoint when think=False/True is requested
SGLang: local OpenAI-compatible server (default http://127.0.0.1:30000/v1),
        https://github.com/sgl-project/sglang — continuous batching +
        RadixAttention prefix cache. Started with scripts/serve_sglang.sh.

All three use the OpenAI SDK — only base_url / api_key / extra_body differ.
"""

import json
import re
import threading
import time
import sys
import os
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from infrastructure.config import (
    MODELS, get_api_key, get_base_url, OLLAMA_PROVIDERS, REQUEST_TIMEOUT,
    MAX_RETRIES, RETRY_DELAY, DEFAULT_TEMPERATURE, MAX_TOKENS_LONG,
)


# ── Token accounting (thread-local) ───────────────────────
# call_llm trả về str, và 10 framework đang dựa vào chữ ký đó. Để đếm token mà
# không phải sửa cả 10, mỗi lời gọi cộng dồn usage vào một bộ đếm THEO THREAD.
# Runner chạy mỗi câu hỏi trong đúng một worker thread (không framework nào tự
# sinh thread — đã kiểm), nên "một câu hỏi = một thread" là bất biến đúng:
#     reset_usage()            # đầu câu
#     ... framework gọi call_llm nhiều lần ...
#     get_usage()              # cuối câu → tokens_in / tokens_out của riêng câu đó
_usage = threading.local()


def reset_usage() -> None:
    """Xoá bộ đếm của thread hiện tại. Runner gọi ở đầu mỗi câu hỏi."""
    _usage.tokens_in = 0
    _usage.tokens_out = 0
    _usage.n_calls = 0
    _usage.n_missing = 0


def get_usage() -> dict:
    """Tổng token của thread hiện tại kể từ reset_usage().

    n_missing đếm số lời gọi KHÔNG kèm usage (server cũ, lỗi mạng). Nếu > 0 thì
    tokens_in/out là cận dưới — phải kiểm cột này trước khi báo cáo chi phí.
    """
    return {
        "tokens_in":  getattr(_usage, "tokens_in", 0),
        "tokens_out": getattr(_usage, "tokens_out", 0),
        "n_llm_calls": getattr(_usage, "n_calls", 0),
        "n_usage_missing": getattr(_usage, "n_missing", 0),
    }


def _add_usage(tin, tout) -> None:
    _usage.n_calls = getattr(_usage, "n_calls", 0) + 1
    if tin is None and tout is None:
        _usage.n_missing = getattr(_usage, "n_missing", 0) + 1
        return
    _usage.tokens_in  = getattr(_usage, "tokens_in", 0)  + int(tin or 0)
    _usage.tokens_out = getattr(_usage, "tokens_out", 0) + int(tout or 0)


# ── Shared OpenAI clients ─────────────────────────────────
# v2: one client per base_url instead of a new client (and a new HTTP
# connection pool) per call. With 32+ worker threads hammering SGLang the
# per-call construction was measurable and leaked sockets. The OpenAI client
# is thread-safe. SDK retries are disabled — the retry loop in call_llm owns
# that policy so behaviour is identical across providers.
_CLIENTS: dict = {}
_CLIENTS_LOCK = threading.Lock()


def _get_client(api_key: str, base_url: str):
    key = (api_key, base_url)
    client = _CLIENTS.get(key)
    if client is None:
        with _CLIENTS_LOCK:
            client = _CLIENTS.get(key)
            if client is None:
                from openai import OpenAI
                client = OpenAI(
                    api_key=api_key, base_url=base_url,
                    timeout=REQUEST_TIMEOUT, max_retries=0,
                )
                _CLIENTS[key] = client
    return client


def _call_ollama_native(
    base_url: str, model_id: str, messages: list[dict],
    temperature: float, max_tokens: int, think: bool,
) -> str:
    """
    Ollama's native /api/chat endpoint (not the OpenAI-compatible /v1 one).
    Needed because /v1/chat/completions silently ignores "think": false, so
    Qwen3/DeepSeek-R1 always run their hidden <think> reasoning pass first —
    burning most of the token budget (and wall time) on reasoning that gets
    thrown away, and sometimes exhausting max_tokens before ever emitting the
    actual answer (see zero_shot's ~42% null rate on qwen3-32b at MAX_TOKENS_LONG).
    """
    url = base_url.removesuffix("/v1") + "/api/chat"
    payload = {
        "model": model_id,
        "messages": messages,
        "stream": False,
        "think": think,
        "options": {"temperature": temperature, "num_predict": max_tokens},
    }
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as resp:
        data = json.loads(resp.read())
    # Ollama native đặt tên khác OpenAI: prompt_eval_count / eval_count.
    _add_usage(data.get("prompt_eval_count"), data.get("eval_count"))
    return data["message"]["content"].strip()


def _sglang_extra_body(think: bool | None) -> dict:
    """
    SGLang honours HF chat-template kwargs per request. For hybrid-reasoning
    models (Qwen3, DeepSeek-R1-Distill on Qwen) `enable_thinking=False` makes
    the template emit an empty <think></think> block so the model answers
    directly — the exact equivalent of Ollama's "think": false. Models whose
    template has no such kwarg (Llama) simply ignore it.
    """
    if think is None:
        return {}
    return {"chat_template_kwargs": {"enable_thinking": bool(think)}}


def call_llm(
    model_name: str,
    messages: list[dict],
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = MAX_TOKENS_LONG,
    think: bool | None = None,
) -> str:
    """
    Call Ollama / SGLang. Returns text response.
    Retries on rate limits with exponential backoff.
    Ollama/SGLang do not enforce rate limits so backoff is only for connection errors.

    think: only meaningful for hybrid-reasoning models (Qwen3, DeepSeek-R1).
    None (default) leaves the server's default thinking behavior untouched.
    Pass False to force a direct answer with no hidden <think> pass:
      - Ollama  → routed through native /api/chat (see _call_ollama_native)
      - SGLang  → extra_body.chat_template_kwargs.enable_thinking=False
    """
    cfg      = MODELS[model_name]
    model_id = cfg["model_id"]
    provider = cfg["provider"]
    api_key  = get_api_key(provider)
    base_url = get_base_url(provider)

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            if provider in OLLAMA_PROVIDERS and think is not None:
                return _call_ollama_native(
                    base_url, model_id, messages, temperature, max_tokens, think,
                )

            client = _get_client(api_key, base_url)
            kwargs = dict(
                model=model_id,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            if provider == "sglang":
                extra = _sglang_extra_body(think)
                if extra:
                    kwargs["extra_body"] = extra
            # hoangnv: Ollama's /v1/chat/completions (OpenAI-compatible) endpoint
            # silently ignores extra_body options like num_ctx, so context window
            # is instead baked into the Ollama model tag itself (see config.py
            # comment on "ollama-qwen3-32b" / qwen3-32b-ctx8k Modelfile).
            resp = client.chat.completions.create(**kwargs)
            u = getattr(resp, "usage", None)
            _add_usage(getattr(u, "prompt_tokens", None) if u else None,
                       getattr(u, "completion_tokens", None) if u else None)
            content = resp.choices[0].message.content
            return (content or "").strip()

        except Exception as e:
            err  = str(e)
            wait = RETRY_DELAY * attempt

            if "rate_limit" in err.lower() or "429" in err:
                wait = 30 * attempt
                print(f"  [{provider}] rate-limited — waiting {wait}s "
                      f"(attempt {attempt}/{MAX_RETRIES})")
            else:
                print(f"  [{provider}] {model_name} attempt {attempt}/{MAX_RETRIES}: "
                      f"{err[:100]}")

            if attempt < MAX_RETRIES:
                time.sleep(wait)
            else:
                raise

    return ""


def parse_answer(text: str) -> str | None:
    """
    Extract A/B/C/D from model output.
    Returns None if no valid letter found.
    Priority: exact single char → labeled → bold → parenthesised → standalone.

    Strips <think>...</think> blocks first (Qwen3, DeepSeek-R1 extended thinking)
    so inner mentions of A/B/C/D in reasoning don't pollute matching.
    """
    if not text:
        return None
    text = text.strip()

    # Strip extended thinking blocks before any pattern matching
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
    if not text:
        return None

    # Exact single character
    if text in ("A", "B", "C", "D"):
        return text

    # "Final answer: B" (all frameworks that ask for CoT-style reasoning end
    # with this exact literal marker — see cot/cot_sc/self_refine/medagents/
    # mdagents prompts). Checked first and anchored on "final answer"
    # specifically, since it is the one unambiguous conclusion marker.
    m = re.search(r'final\s+answer[:\s]*\**\s*([ABCD])\b', text, re.IGNORECASE)
    if m:
        return m.group(1).upper()

    # Bold **B**
    m = re.search(r'\*\*([ABCD])\*\*', text)
    if m:
        return m.group(1).upper()

    # Parenthesised (C)
    m = re.search(r'\(([ABCD])\)', text)
    if m:
        return m.group(1).upper()

    # hoangnv: weak fallback for free-text reasoning that never emits the
    # literal "Final answer:" marker. "is" was dropped from the trigger
    # words — under IGNORECASE it matches the indefinite article in plain
    # English sentences like "The patient is a 62-year-old..." and silently
    # extracts "A", regardless of the model's actual conclusion (confirmed:
    # this skewed CoT's predicted-answer distribution to ~39% "A" vs a ~29%
    # "A" gold rate, while zero_shot's single-char output — unaffected by
    # this fallback — matched gold's distribution almost exactly).
    # Last match is used instead of first: reasoning walks through each
    # option in order (mentioning "answer"/"option"/"đáp án" for each one),
    # so the *last* mention is closer to the actual conclusion than the
    # first.
    matches = list(re.finditer(
        r'(?:answer|đáp án|đáp|option)[:\s]+([ABCD])\b', text, re.IGNORECASE,
    ))
    if matches:
        return matches[-1].group(1).upper()

    # Last standalone letter mention (closest to the conclusion)
    matches = list(re.finditer(r'\b([ABCD])\b', text))
    if matches:
        return matches[-1].group(1).upper()

    return None
