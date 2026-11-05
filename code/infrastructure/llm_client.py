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
from infrastructure.choices import choice_letters
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


ANSWER_PARSER_VERSION = "explicit-answer-v1"


def parse_answer(text: str, n_options: int = 4) -> str | None:
    """
    Extract a label from model output (A-D, or A-E for five options).
    Prefer a final-answer declaration, then consistent answer declarations or
    standalone answer lines. Conflicting declarations, out-of-range labels and
    option discussion without an answer return None. Never consult the gold.
    """
    letters = choice_letters(n_options)
    if not text:
        return None
    # An unfinished thinking block contains no visible final answer either.
    text = re.sub(r'<think>.*?(?:</think>|$)', '', text,
                  flags=re.DOTALL | re.IGNORECASE).strip()
    if not text:
        return None
    plain = re.sub(r'[*_`]', '', text)
    # Capture even out-of-range letters so "Final answer: E" on a four-option
    # question cannot silently fall back to an A-D mention in the explanation.
    label = r'[\(\[\"\']?([A-Za-z])[\)\]\"\']?(?!\w)'

    def value(match) -> str | None:
        letter = match.group(1)
        suffix = plain[match.end():]
        # "answer is a beta-blocker" contains an article, not a label.
        if letter == 'a' and re.match(r'\s+\w', suffix):
            return None
        # An unresolved "A or B" / "A/B" is not a single selected option.
        if re.match(r'\s*(?:or\b|and\b|/|,)\s*\(?[A-Za-z]\)?(?!\w)',
                    suffix, re.IGNORECASE):
            return None
        return letter.upper() if letter.upper() in letters else None

    final = list(re.finditer(
        r'\bfinal\s+answer\b\s*(?:(?:is|would\s+be)\b\s*)?[:：=\-]?\s*' + label,
        plain, re.IGNORECASE,
    ))
    if final:
        # An explicit final answer may revise an earlier provisional answer.
        return value(final[-1])

    candidates = []
    declarations = re.finditer(
        r'(?:\b(?:the\s+)?(?:(?:correct|best)\s+)?answer\b'
        r'\s*(?:(?:is|would\s+be)\b\s*)?[:：=\-]?\s*'
        r'|đáp\s+án\s*(?:đúng\s*)?(?:là\s*)?[:：=\-]?\s*'
        r'|\bconclusion\b\s*[:：=\-]?\s*)' + label,
        plain, re.IGNORECASE,
    )
    for match in declarations:
        # Reject prose articles without blocking a later real declaration.
        if match.group(1) == 'a' and re.match(r'\s+\w', plain[match.end():]):
            continue
        candidates.append(value(match))

    # Bare labels on the first/last non-empty line are direct answers. Labels
    # such as "Option D (...)" inside an explanation are deliberately ignored.
    lines = [line.strip() for line in plain.splitlines() if line.strip()]
    for line in (lines[0], lines[-1]):
        match = re.fullmatch(label + r'[.!]?', line)
        if match:
            letter = match.group(1).upper()
            candidates.append(letter if letter in letters else None)

    # Explicit concluding choices can expose contradictory model output:
    # "The correct answer is B. ... Therefore, option A ..." is ambiguous.
    conclusions = re.finditer(
        r'\b(?:therefore|thus|hence|in\s+conclusion)\b\s*[:,]?\s*'
        r'(?:option\s+|I\s+(?:choose|select)\s+)' + label,
        plain, re.IGNORECASE,
    )
    for match in conclusions:
        suffix = plain[match.end():]
        if re.match(r'\s*(?:\([^)]*\)\s*)?(?:(?:is|would\s+be)\s+)?'
                    r'(?:not|incorrect|wrong|excluded|ruled\s+out)\b', suffix, re.IGNORECASE):
            continue
        candidates.append(value(match))

    if candidates:
        return candidates[0] if None not in candidates and len(set(candidates)) == 1 else None

    # Retain unambiguous formatted labels, without choosing the first one in
    # an option-by-option discussion. No arbitrary last-letter fallback.
    formatted = re.findall(r'\*\*([A-Z])\*\*|\(([A-Z])\)', text)
    labels = {bold or paren for bold, paren in formatted}
    if len(labels) == 1 and labels <= set(letters):
        return labels.pop()
    return None
