"""
ViMedAgent v2 — Central configuration

v2 adds the "sglang" provider (https://github.com/sgl-project/sglang): an
OpenAI-compatible server with continuous batching + RadixAttention prefix
caching. Same frameworks, same prompts, same result format as v1 — only the
inference backend (and the client-side concurrency) changes.

Edit N_RUNS and SEEDS to control how many times each experiment runs.
All other settings are referenced by the rest of the codebase via imports.
"""

import os

# ── API ───────────────────────────────────────────────────
# Doc tu bien moi truong, KHONG bao gio ghi khoa vao file nay.
#   export GROQ_API_KEY=...
# 6 khoa Groq tung bi hardcode o day (da comment) bi xoa ngay 24/09/2026 —
# chung phai duoc COI LA DA LO va thu hoi tren console, vi da nam plaintext
# tren dia. Chuoi rong lam get_api_key("groq") tra ve "" thay vi NameError.
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")

# ── Models ────────────────────────────────────────────────
MODELS: dict[str, dict] = {

    # ── Groq (cloud) ──────────────────────────────────────
    "llama-3.3-70b": {
        "provider": "groq",
        "model_id": "llama-3.3-70b-versatile",
    },
    "llama-3.1-8b": {
        "provider": "groq",
        "model_id": "llama-3.1-8b-instant",
    },
    "qwen3-32b": {
        "provider": "groq",
        "model_id": "qwen/qwen3-32b",
    },
    # "deepseek-r1-70b": {
    #     "provider": "groq",
    #     "model_id": "deepseek-r1-distill-llama-70b",
    # },

    # ── Local (Ollama) ─────────────────────────────────────
    # Install: https://ollama.com/download
    # Pull:    ollama pull <model_id>
    #
    # hoangnv: all four models below point at "*-ctx8k" tags instead of the base
    # tag. Ollama's default num_ctx (the model's full trained context, e.g. 40960
    # for qwen3:32b) reserves a KV cache too large to fit alongside the weights on
    # the 48GB GPU, forcing most layers onto CPU (measured: qwen3:32b at 23%/77%
    # CPU/GPU, ~76s/call). Per-request "options.num_ctx" is silently ignored by
    # Ollama's /v1/chat/completions (OpenAI-compatible) endpoint this client uses,
    # so num_ctx has to be baked into a custom tag instead, once per base model:
    #   ollama create <model>-ctx8k -f infrastructure/ollama_modelfiles/Modelfile.<model>
    # (each Modelfile is just "FROM <base tag>" + "PARAMETER num_ctx 8192"). Confirmed
    # for qwen3:32b: qwen3-32b-ctx8k runs 100% GPU, 29GB, ~6-7s/call (~12x faster).
    # The other three base models aren't pulled on this machine yet — pull + run the
    # `ollama create` above the first time you actually use one of them.
    "ollama-llama3.1-8b": {
        "provider": "ollama",
        "model_id": "llama3.1-8b-ctx8k",    # ~5GB VRAM base
    },
    "ollama-qwen3-14b": {
        "provider": "ollama",
        "model_id": "qwen3-14b-ctx8k",      # ~9GB VRAM base
    },
    "ollama-qwen3-32b": {
        "provider": "ollama",
        "model_id": "qwen3-32b-ctx8k",      # ~12GB weights (Q4_K_M) + capped KV cache
    },
    "ollama-deepseek-r1-14b": {
        "provider": "ollama",
        "model_id": "deepseek-r1-14b-ctx8k",  # ~9GB VRAM base, has <think> blocks
    },
    "ollama-llama3.3-70b": {
        "provider": "ollama",
        # ctx4k (not ctx8k): measured worst-case prompt across vm14k_clean for
        # medagents is ~2475 tokens (longest question + max-budget filler at
        # every accumulation step) — 4096 covers that with margin. ctx8k made
        # the model 57GB (>48GiB card), forcing 13%/87% CPU/GPU split and
        # >300s timeouts per call; ctx4k fits 100% GPU (measured 49GB, ~0.3s/call).
        "model_id": "llama3.3-70b-ctx4k",
    },

    # ── Ollama, hoangnv's own instance (port 11435) ────────
    # Separate `ollama serve` under /storage/hoangnv/ollama-local, independent
    # OLLAMA_MAX_LOADED_MODELS from the shared server on 11434 (owned by cuongvt).
    # Lets a 2nd model load and run truly in parallel when VRAM allows, instead of
    # the two servers fighting over the shared server's single-model slot.
    "ollama2-qwen3-32b": {
        "provider": "ollama2",
        "model_id": "qwen3-32b-ctx8k",
    },

    # ── SGLang (v2) ────────────────────────────────────────
    # One `sglang` server (scripts/serve_sglang.sh) serves ONE model at a time
    # on SGLANG_BASE_URL (default http://127.0.0.1:30000/v1). model_id must be
    # the exact --model-path the server was started with (SGLang's default
    # served-model-name is the model path). All entries here are HF repo ids,
    # downloaded on first use into the HF cache (HF_HOME).
    #
    # Thinking (Qwen3 / DeepSeek-R1) is disabled per request through
    # extra_body.chat_template_kwargs.enable_thinking — see llm_client.py — so
    # the think=False contract of every framework is honoured exactly like the
    # Ollama native /api/chat path.
    "sglang-qwen3-32b": {
        "provider": "sglang",
        # AWQ int4 (~19GB weights). Same 4-bit class as Ollama's qwen3:32b
        # Q4_K_M tag, so results stay comparable with the v1 tables; validated
        # on pilot_100 (exp 007). Deliberately the only Qwen3-32B variant here —
        # FP8/bf16 would score differently for reasons unrelated to the frameworks.
        "model_id": "Qwen/Qwen3-32B-AWQ",
    },
    "sglang-qwen3-14b": {
        "provider": "sglang",
        "model_id": "Qwen/Qwen3-14B",
    },
    "sglang-llama3.1-8b": {
        "provider": "sglang",
        "model_id": "meta-llama/Llama-3.1-8B-Instruct",
    },
    "sglang-llama3.3-70b": {
        "provider": "sglang",
        # AWQ int4 (~37GB weights) — 48GB card with a small KV cache, or --tp 2.
        "model_id": "casperhansen/llama-3.3-70b-instruct-awq",
    },
    "sglang-deepseek-r1-14b": {
        "provider": "sglang",
        "model_id": "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
    },
    "sglang-medgemma-4b": {
        "provider": "sglang",
        # MedGemma 4B instruction-tuned (Google Health AI Developer Foundations),
        # nền Gemma 3 4B + tháp thị giác SigLIP → repo là image-text-to-text, SGLang
        # nạp qua gemma3_mm. Ta chỉ dùng phần text; tháp thị giác vẫn chiếm chỗ.
        #
        # KHÁC MỌI MODEL KHÁC TRONG BẢNG NÀY, ba điểm phải khai trong Methods:
        #   1. bf16, KHÔNG lượng tử hoá — 4B mà ép 4-bit thì mất chất lượng nhiều
        #      hơn hẳn 32B/70B. Nên đây là model duy nhất không cùng hạng 4-bit.
        #   2. Chuyên biệt y khoa (các model khác là phổ thông) — muốn tách được
        #      ảnh hưởng của việc tinh chỉnh y khoa thì cần thêm google/gemma-3-4b-it
        #      làm đối chứng cùng cỡ, cùng họ.
        #   3. Tinh chỉnh y khoa của MedGemma chủ yếu trên dữ liệu TIẾNG ANH, còn
        #      VM14K là tiếng Việt. Điểm thấp có thể do ngôn ngữ chứ không do y khoa.
        #
        # Repo bị khoá (gated=auto): phải bấm đồng ý điều khoản tại
        #   https://huggingface.co/google/medgemma-4b-it
        # rồi dùng token có quyền (HF_TOKEN hoặc ~/.cache/huggingface/token).
        # Trọng số 8,01 GB.
        "model_id": "google/medgemma-4b-it",
    },
}


# Providers that speak Ollama's native /api/chat (needed for think=False).
# Single source of truth — exp 004 root cause was this tuple living in 3 places.
OLLAMA_PROVIDERS = ("ollama", "ollama2")

# SGLang server endpoint. Override per machine without editing code:
#   export SGLANG_BASE_URL=http://<host>:<port>/v1
SGLANG_BASE_URL = os.environ.get("SGLANG_BASE_URL", "http://127.0.0.1:30000/v1")
SGLANG_API_KEY  = os.environ.get("SGLANG_API_KEY", "sglang")   # only if server started with --api-key


def get_api_key(provider: str) -> str:
    if provider == "groq":
        return GROQ_API_KEY
    if provider in OLLAMA_PROVIDERS:
        return "ollama"     # Ollama does not require a real API key
    if provider == "sglang":
        return SGLANG_API_KEY
    raise ValueError(f"Unknown provider: {provider}")


def get_base_url(provider: str) -> str:
    if provider == "groq":
        return "https://api.groq.com/openai/v1"
    if provider == "ollama":
        return "http://localhost:11434/v1"
    if provider == "ollama2":
        # hoangnv's own Ollama instance (port 11435, /storage/hoangnv/ollama-local),
        # independent OLLAMA_MAX_LOADED_MODELS from the shared server on 11434 —
        # lets a second model run in parallel when VRAM allows (see experiment log).
        return "http://127.0.0.1:11435/v1"
    if provider == "sglang":
        return SGLANG_BASE_URL
    raise ValueError(f"Unknown provider: {provider}")


# ── Paths ─────────────────────────────────────────────────
DATA_DIR    = "data"
PILOT_FILE  = "data/pilot_100.jsonl"
FULL_FILE   = "data/vm14k_clean.jsonl"
RESULTS_DIR      = "results"
RESULTS_TEST_DIR = "results_test"   # dataset stems starting with "pilot" go here instead

# ── Inference ─────────────────────────────────────────────
# DEFAULT_TEMPERATURE=0.3 uniformly across all frameworks (paper Section III.E).
# CoT-SC is the only exception: T=0.5 for sampling diversity (paper Section III.B).
DEFAULT_TEMPERATURE = 0.3
COT_SC_TEMPERATURE  = 0.5   # CoT-SC only — paper explicitly states T=0.5
MAX_RETRIES         = 3
RETRY_DELAY         = 10    # seconds before retry on rate limit

# hoangnv: harness previously called the LLM strictly one-at-a-time. The ollama
# service already allows OLLAMA_NUM_PARALLEL=4 concurrent requests, so running
# workers serially left the GPU mostly idle. Match client-side concurrency to that.
NUM_WORKERS = 4

# v2: SGLang does continuous batching server-side, so the client should keep
# many requests in flight — throughput scales almost linearly until the KV
# cache is full. 32 is a safe default for a 32B-AWQ model on one 48GB card;
# raise on bigger cards. Override without editing: export VIMED_NUM_WORKERS=64
NUM_WORKERS_BY_PROVIDER = {
    "groq":   4,
    "ollama": NUM_WORKERS,
    "ollama2": NUM_WORKERS,
    "sglang": 32,
}


def get_num_workers(model_name: str) -> int:
    env = os.environ.get("VIMED_NUM_WORKERS")
    if env:
        return max(1, int(env))
    provider = MODELS[model_name]["provider"]
    return NUM_WORKERS_BY_PROVIDER.get(provider, NUM_WORKERS)

# Per-request timeout (seconds). Under heavy batching a single request may
# queue behind others, so this is generous; the retry loop handles the rest.
REQUEST_TIMEOUT = int(os.environ.get("VIMED_REQUEST_TIMEOUT", "300"))

# ── Token budgets ─────────────────────────────────────────
MAX_TOKENS_SHORT   = 16     # single-char decision / synthesizer output
MAX_TOKENS_ROLES   = 512    # expert role listing — large enough for Qwen3/DeepSeek-R1 <think> blocks
MAX_TOKENS_COMPLEX = 8      # complexity classification (Low/Moderate/High)
MAX_TOKENS_MEDIUM  = 300    # report summarization, consultation, critique
MAX_TOKENS_LONG    = 512    # CoT reasoning, self-refine initial/refine
MAX_TOKENS_AGENT   = 600    # per-expert full analysis
MAX_TOKENS_DEBATE  = 400    # MAD debate rounds

# ── Multi-run settings ────────────────────────────────────
# Each experiment is run N_RUNS times with different random seeds.
# Results from all runs are aggregated to produce mean ± std in tables.
N_RUNS = 3
SEEDS  = [42, 43, 44]  # len must equal N_RUNS

# ── Framework tiers ───────────────────────────────────────
# Tier 1: Single-LLM prompting methods (no inter-agent communication)
TIER1_FRAMEWORKS = ["zero_shot", "cot", "cot_sc", "self_refine"]

# Tier 2: Ensemble (single model, shuffle diversity, no communication)
TIER2_FRAMEWORKS = ["voting_ensemble"]

# Tier 3: Multi-Agent Systems (inter-agent communication)
TIER3_FRAMEWORKS = [
    "medagents", "medagents_vi",
    "mdagents",  "mdagents_vi",
    "mad_debate",
]

ALL_FRAMEWORKS = TIER1_FRAMEWORKS + TIER2_FRAMEWORKS + TIER3_FRAMEWORKS

# framework -> "tier1"/"tier2"/"tier3", used to place result files under
# results(_test)/tier{N}/<framework>/ instead of a flat directory.
FRAMEWORK_TIER = {
    **{fw: "tier1" for fw in TIER1_FRAMEWORKS},
    **{fw: "tier2" for fw in TIER2_FRAMEWORKS},
    **{fw: "tier3" for fw in TIER3_FRAMEWORKS},
}

# ── Experiment matrix ─────────────────────────────────────
# Each (framework, model) pair is run N_RUNS times.
MODEL_NAMES = list(MODELS.keys())

EXPERIMENT_MATRIX: list[tuple[str, str]] = [
    (fw, model)
    for fw in ALL_FRAMEWORKS
    for model in MODEL_NAMES
]

# ── EN reference values (MedAgentAudit [5]) ──────────────
# Used to compute Cross-Lingual Delta (CLD).
EN_REFERENCE = {
    "medagents":       {"nmar": 18.3, "icr": 42.1},
    "medagents_vi":    {"nmar": 18.3, "icr": 42.1},  # same base architecture
    "mdagents":        {"nmar": 17.1, "icr": 38.4},
    "mdagents_vi":     {"nmar": 17.1, "icr": 38.4},
    "mad_debate":      {"nmar": 16.9, "icr": 44.2},
    "voting_ensemble": {"nmar": None, "icr": None},   # N/A for ensemble
}