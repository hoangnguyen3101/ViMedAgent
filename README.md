# ViMedAgent v2 — SGLang backend

ViMedAgent benchmarks single-LLM prompting against multi-agent systems on Vietnamese medical
multiple-choice questions (VM14K, 11,094 questions). **v2** is identical to v1 in every framework,
prompt, seed, result format and metric — it only adds an **SGLang** inference backend
(continuous batching + RadixAttention prefix cache), so v2 results compare directly with v1.

- Benchmark, frameworks and metrics in detail: [`code/README.md`](code/README.md)
- Backend setup and validation log: [`code/experiment/007-sglang-backend-setup/README.md`](code/experiment/007-sglang-backend-setup/README.md)

All commands below run from `code/` unless stated otherwise.

---

## 1. Requirements

| | |
|---|---|
| OS | Linux |
| GPU | NVIDIA, driver ≥ 525. Compute capability ≥ 7.5 (Turing or newer) |
| Python | conda, or Python ≥ 3.10 |
| Disk | 4–40 GB per model (see §3), plus up to ~300 MB of results per model × framework (3 seeds; `mad_debate` is the largest) |
| Hugging Face | an account and a read token, for the gated models in §3 |

One SGLang server serves **one model**. The experiment runner is a separate, lightweight
HTTP client and can run on the same machine or another one.

---

## 2. Install

### 2.1 Client (the experiment runner)

```bash
git clone https://github.com/hoangnguyen3101/ViMedAgent.git
cd ViMedAgent/code
pip install -r requirements-client.txt       # openai, scipy, numpy — no torch needed
```

### 2.2 SGLang server

**Standard path — Linux x86_64 + NVIDIA GPU:**

```bash
bash scripts/x86_cuda/install_sglang.sh      # creates conda env "sglang" with sglang[all]
```

Tested with **SGLang 0.5.9**. The script installs into its own env, so it does not touch the
client environment.

> **GPUs older than Ampere (compute capability < 8.0, e.g. Turing / Quadro RTX 8000 / T4).**
> The `sgl-kernel` wheels contain no code for these GPUs. `serve_sglang.sh` detects this and starts
> the server through `scripts/x86_cuda/sglang_turing_launch.py`, which swaps in JIT, Triton and
> PyTorch fallbacks. Nothing to configure, but it needs a CUDA toolkit ≥ 12 with `nvcc` and
> g++ ≥ 10 (the installer adds GCC 13 to the env). Check the fallbacks before serving:
>
> ```bash
> conda run -n sglang python scripts/x86_cuda/sglang_turing_probe.py
> ```

**aarch64 + Blackwell (NVIDIA DGX Spark / ASUS Ascent GX10, GB10):** PyPI wheels do not apply
(no `sm_121` code). Use the pre-built container instead:

```bash
bash scripts/install_gx10.sh                 # client deps + docker pull + model pre-download
```

A native (no-Docker) install for GB10 is in the [appendix](#appendix--gb10-native-install-no-docker).

---

## 3. Download models

Each `--model` key in [`infrastructure/config.py`](code/infrastructure/config.py) maps to one
Hugging Face repo. The server must serve **exactly** that repo id.

| `--model` key | Hugging Face repo | Weights | Gated |
|---|---|---|---|
| `sglang-qwen3-32b` | `Qwen/Qwen3-32B-AWQ` | ~19 GB, AWQ int4 | no |
| `sglang-llama3.3-70b` | `casperhansen/llama-3.3-70b-instruct-awq` | ~40 GB, AWQ int4 | no |
| `sglang-qwen3-14b` | `Qwen/Qwen3-14B` | ≈ 30 GB, bf16 | no |
| `sglang-llama3.1-8b` | `meta-llama/Llama-3.1-8B-Instruct` | ≈ 16 GB, bf16 | **yes** |
| `sglang-deepseek-r1-14b` | `deepseek-ai/DeepSeek-R1-Distill-Qwen-14B` | ≈ 30 GB, bf16 | no |
| `sglang-gemma3-4b-awq` | `gaunernst/gemma-3-4b-it-int4-awq` | 3.8 GB, AWQ int4 | no |
| `sglang-gemma3-4b-w4a16` | `RedHatAI/gemma-3-4b-it-quantized.w4a16` | ~4.8 GB, W4A16 | no — **needs GPU ≥ 8.0** |
| `sglang-gemma3-4b` | `google/gemma-3-4b-it` | 8.0 GB, bf16 | **yes** (manual) |
| `sglang-medgemma-4b` | `google/medgemma-4b-it` | 8.0 GB, bf16 | **yes** (auto) |
| `sglang-medgemma-4b-w4a16` | self-quantized, see §8 | ~4 GB, W4A16 | — |

For gated repos, accept the license on the model's Hugging Face page first, then:

```bash
hf auth login                                # or: export HF_TOKEN=hf_...
hf download Qwen/Qwen3-32B-AWQ               # repeat for each model you need
```

On GB10, download through the container instead: `MODEL=<repo> bash scripts/install_gx10.sh`.

> Never put a token in a file inside the repo. `.gitignore` already blocks `.env`, `*.key`, `*.pem`.

---

## 4. Start the server

Run it in `tmux` — it stays in the foreground.

```bash
tmux new -s sglang
MODEL=Qwen/Qwen3-32B-AWQ MEM_FRACTION=0.65 bash scripts/x86_cuda/serve_sglang.sh
```

Wait for `The server is fired up and ready to roll!`, then detach with `Ctrl-b d`.

**Options** (environment variables, all optional):

| Variable | Default | Meaning |
|---|---|---|
| `MODEL` | `Qwen/Qwen3-32B-AWQ` | Hugging Face repo id or local path |
| `PORT` | `30000` | the runner reaches it at `http://127.0.0.1:$PORT/v1` |
| `MEM_FRACTION` | `0.85` | share of **total** GPU memory for weights + KV cache. Lower it on a shared GPU |
| `MAX_RUNNING` | `64` | max sequences batched together |
| `CONTEXT_LEN` | `8192` | context window, same for every model |
| `DTYPE` | `auto` | `auto` = fp16 below compute capability 8.0, bf16 otherwise |
| `ATTN_BACKEND` | `auto` | `auto` = triton below 8.0 |
| `EXTRA_ARGS` | — | passed unchanged to `sglang.launch_server` |

Pick `MEM_FRACTION` from the memory a model needs: `MEM_FRACTION ≈ needed_GB / card_GB`.

### Per-model recipes

**Qwen3-32B-AWQ** — needs ~28 GB (19 GB weights + KV):
```bash
MODEL=Qwen/Qwen3-32B-AWQ MEM_FRACTION=0.65 MAX_RUNNING=32 bash scripts/x86_cuda/serve_sglang.sh
```

**Llama-3.3-70B-AWQ** — needs ~50–60 GB, so a single 48 GB card is not enough once anything
else holds memory. Use a larger GPU or GB10.

**Gemma 3 / MedGemma 4B** — see §4.1 if your GPU is older than Ampere. On compute capability
≥ 8.0, the defaults work:
```bash
MODEL=RedHatAI/gemma-3-4b-it-quantized.w4a16 MEM_FRACTION=0.35 bash scripts/x86_cuda/serve_sglang.sh
```

**On GB10** use `serve_gx10.sh`, where memory is an absolute `MEM_GB` instead of a fraction:
```bash
MODEL=Qwen/Qwen3-32B-AWQ MEM_GB=27 bash scripts/serve_gx10.sh
MODEL=casperhansen/llama-3.3-70b-instruct-awq MEM_GB=60 bash scripts/serve_gx10.sh
```

### 4.1 Gemma 3 on GPUs older than Ampere (compute capability < 8.0)

Gemma 3 is designed for **bf16**, which these GPUs do not have. Three things are required
**together**; with any one missing, the server crashes on the first request:

```bash
MODEL=gaunernst/gemma-3-4b-it-int4-awq \
DTYPE=float32 ATTN_BACKEND=triton \
EXTRA_ARGS='--prefill-attention-backend torch_native --decode-attention-backend triton' \
MEM_FRACTION=0.35 MAX_RUNNING=32 \
  bash scripts/x86_cuda/serve_sglang.sh
```

| | Why |
|---|---|
| `DTYPE=float32` | In fp16 Gemma's logits overflow → softmax produces `NaN` → the sampler asserts and the scheduler dies. fp32 is the only dtype both supported and wide enough |
| prefill `torch_native` | In fp32, Triton's prefill kernel needs 69,760 B of shared memory with Gemma's `head_dim=256`; these GPUs have 65,536 B |
| decode `triton` | Do **not** use `torch_native` for decode: it loops over every request in Python and has no CUDA-graph support — ~4× slower (77 vs 301.6 tok/s measured) |

The launcher also applies two runtime patches automatically (they only take effect when
`float32` is requested explicitly, so fp16 runs of other models are unaffected):

- SGLang 0.5.9 forces fp16 on every GPU below 8.0, even when `--dtype float32` is passed —
  the stated reason is missing bf16 support, but fp32 is fully supported. The patch keeps an
  explicit `float32`.
- SGLang's AWQ only allows fp16 activations. The compute path is already dtype-generic, so the
  patch allows fp32 → **W4A32**: 4-bit weights, fp32 matmul. Uses 6.1 GB.

`RedHatAI/gemma-3-4b-it-quantized.w4a16` **cannot** run on these GPUs: compressed-tensors refuses
to load below compute capability 8.0. Use `gaunernst/gemma-3-4b-it-int4-awq` instead.

### 4.2 Remote server through a LiteLLM proxy (optional)

The `litellm-*` keys call a model served on another machine through a LiteLLM proxy instead of a
local SGLang server:

```bash
export LITELLM_BASE_URL=http://<proxy-host>:<port>/v1
export LITELLM_API_KEY=<key>
python scripts/litellm_smoke_test.py --model litellm-qwen2.5-72b   # check before every batch of runs
```

The `model_id` of a `litellm-*` key must match the `model_name` in the proxy's `config.yaml`
exactly.

---

## 5. Run experiments

Point the runner at the server and check it before running anything long:

```bash
export SGLANG_BASE_URL=http://127.0.0.1:30000/v1
python scripts/sglang_smoke_test.py --model sglang-qwen3-32b
```

### One framework

```bash
# quick check on 100 questions — writes to results_test/
python -m runner.run_experiments --dataset data/pilot_100.jsonl  --framework zero_shot --model sglang-qwen3-32b

# full benchmark, 11,094 questions × 3 seeds — writes to results/
python -m runner.run_experiments --dataset data/vm14k_clean.jsonl --framework zero_shot --model sglang-qwen3-32b
```

### All ten frameworks

| Tier | Framework | LLM calls / question |
|---|---|---|
| 1 — single LLM | `zero_shot` | 1 |
| | `cot` | 1 |
| | `cot_sc` | 5 |
| | `self_refine` | 2–5 (stops early) |
| 2 — ensemble | `voting_ensemble` | 3 |
| 3 — multi-agent | `medagents` | 9 |
| | `medagents_vi` | 8 |
| | `mdagents` | 2 / 4 / 7 (by routed difficulty) |
| | `mdagents_vi` | 1 / 3 / 6 (difficulty read from the dataset) |
| | `mad_debate` | ≤ 9 (stops on consensus) |

```bash
mkdir -p ../logs
for fw in zero_shot cot cot_sc self_refine voting_ensemble medagents medagents_vi mdagents mdagents_vi mad_debate; do
  PYTHONUNBUFFERED=1 python -m runner.run_experiments \
    --dataset data/vm14k_clean.jsonl --framework $fw --model sglang-qwen3-32b
done 2>&1 | tee -a ../logs/full_run.log
```

Or let a script wait for the server and then start one framework — useful after a reboot or a
model swap:

```bash
bash scripts/wait_and_run.sh <framework> <model-key> <workers>
# e.g.  bash scripts/wait_and_run.sh cot sglang-qwen3-32b 32
```

### Runner options

| Option | Meaning |
|---|---|
| `--workers N` | concurrent requests. Default 32 for SGLang. Keep it ≤ the server's `MAX_RUNNING` |
| `--no-resume` | ignore existing results and start over |
| `--all` | every framework × every model in `config.py` (rarely what you want) |

| Environment variable | Meaning |
|---|---|
| `SGLANG_BASE_URL` | server address, default `http://127.0.0.1:30000/v1` |
| `VIMED_REQUEST_TIMEOUT` | seconds per request, default 300. Raise it (e.g. 900) when the GPU is shared |
| `VIMED_NUM_WORKERS` | default for `--workers` |
| `PYTHONUNBUFFERED=1` | without it, progress lines piped to `tee` sit in an 8 KB buffer and can be lost |

### Resuming and stopping

- **Resume** — re-run the same command. Results are appended per question, and finished
  questions are skipped.
- **Stop** — the runner catches the first `Ctrl-C` of a seed and moves on to the next seed. To stop
  the whole run, press `Ctrl-C` until the prompt returns (once per remaining seed), or `kill` the
  process. Nothing already written is lost.
- **Do not swap the model on a running server.** The runner does not check which model the server
  is serving. Stop the job first, change the server, run the smoke test again.

### Output

```
results/tier{1,2,3}/<framework>/<framework>__<model>__<dataset>__s<seed>.jsonl
results/tier{1,2,3}/<framework>/<framework>__<model>__<dataset>__s<seed>.jsonl.meta.json
```

Datasets whose name starts with `pilot` go to `results_test/` instead. Each `.meta.json` records the
run configuration (model id, provider, temperature, token limits, host, and every restart).

---

## 6. Check results

After each run:

```bash
python - results/tier1/zero_shot/zero_shot__sglang-qwen3-32b__vm14k_clean__s42.jsonl <<'PY'
import json, sys
rows = [json.loads(l) for l in open(sys.argv[1], encoding="utf-8") if l.strip()]
print("rows           :", len(rows), "(expect 11094)")
print("duplicate ids  :", len(rows) - len({r["id"] for r in rows}))
print("api==n_llm     :", all(r.get("api_calls") == r.get("n_llm_calls") for r in rows))
print("usage missing  :", sum(r.get("n_usage_missing", 0) for r in rows))
print("unparsed       :", sum(1 for r in rows if not r.get("answer")))
print("Pass@1         : %.2f%%" % (100 * sum(bool(r["correct"]) for r in rows) / len(rows)))
PY
```

All four must hold: full row count, no duplicate ids, `api_calls == n_llm_calls` (two independent
counters — a mismatch means a call went unrecorded), and `usage missing = 0` (otherwise token
counts are only a lower bound and must not be used for cost).

Compare two models on the same framework and dataset:

```bash
python scripts/compare_results.py --framework zero_shot --dataset pilot_100 --a <model-A> --b <model-B>
```

> **Statistics:** at `zero_shot` the three seeds often give *identical* answers (one-token answers
> sampled at T = 0.3 are nearly deterministic). The statistical unit is therefore the **question**,
> not the (question, seed) pair. Counting all three seeds as independent inflates significance.

---

## 7. Notes for a Methods section

- **Quantization.** Every model runs 4-bit weight-only, but the schemes differ: AWQ (`sglang-*`
  32B/70B), GGUF Q4_K_M (Ollama, v1), GPTQ (W4A16, Gemma on GPUs ≥ 8.0), and **W4A32** for Gemma
  on older GPUs (§4.1).
- **Context window** is 8,192 for every SGLang model.
- **Temperature** is fixed per framework, not per model: 0.3 everywhere, except 0.5 for `cot_sc`
  and 0.0 for the `self_refine` critique step.
- **Truncation.** Generation limits are shared by all models (e.g. 512 tokens for `cot`). Verbose
  models are cut off before stating an answer, and an unparsed answer is scored wrong. Measured on
  Gemma-3-4B `cot`: 37.1 % of answers hit the limit, and 98.7 % of unparsed answers were truncated.
  Report the truncation rate per model.
- **Frameworks are re-implementations**, not exact replications of the authors' code — see
  [`code/README.md`](code/README.md).

---

## 8. MedGemma 4-bit (self-quantized)

No 4-bit MedGemma that follows a published recipe exists on Hugging Face. To keep the
MedGemma ↔ Gemma-3 pair fair, MedGemma is quantized with **the same recipe RedHatAI used** for
`RedHatAI/gemma-3-4b-it-quantized.w4a16`:

```bash
conda create -n quant python=3.11 -y && conda activate quant
pip install "llmcompressor>=0.6" datasets "transformers>=4.50" accelerate
python scripts/quantize_w4a16.py --model google/medgemma-4b-it --out ~/models/medgemma-4b-it-quantized.w4a16
```

Recommended check: quantize `google/gemma-3-4b-it` with the same script and compare it with the
RedHatAI model on `pilot_100` — per-question agreement should be ~90 % or higher, otherwise the
recipe was not reproduced. The output folder's `vimed_quantization.json` records the recipe and
library versions. The result is W4A16, so it needs a GPU with compute capability ≥ 8.0 to serve.

---

## 9. Why SGLang

| | Ollama (v1) | SGLang (v2) |
|---|---|---|
| Parallelism | fixed slots (`OLLAMA_NUM_PARALLEL=4`) | continuous batching, 32–64+ sequences per batch |
| Repeated prefixes (system prompts, multi-agent steps) | recomputed each time | cached (RadixAttention) |
| Disable Qwen3 thinking | native `/api/chat` with `"think": false` | `chat_template_kwargs.enable_thinking=false` |
| Context | baked into the model tag | `--context-length` |

Validated on `pilot_100` (Qwen3-32B, 3 seeds): v1 Ollama 60.33 ± 0.58 %, v2 SGLang 61.33 ± 1.15 %,
87.7 % per-question agreement — within the difference expected between two 4-bit formats.
`zero_shot` on 100 questions: ~7.5 min → 12 s.

## 10. Changes from v1

- `infrastructure/config.py` — `sglang` and `litellm` providers and their models; one tuple for Ollama
  providers; per-provider worker counts; `REQUEST_TIMEOUT`. The Groq provider was removed.
- `infrastructure/llm_client.py` — shared, thread-safe OpenAI client per base URL; disables thinking
  through `chat_template_kwargs`; records per-question token usage.
- `runner/run_experiments.py` — `--workers`; records `tokens_in`, `tokens_out`, `n_llm_calls`,
  `n_usage_missing`; writes a `.meta.json` per run.
- `frameworks/` — every LLM step now stores its full `raw` output. Prompts and logic are unchanged.
- `scripts/` — install, serve, smoke-test, compare, wait-and-run, quantize.

---

## Appendix — GB10 native install (no Docker)

From the NVIDIA forum thread *"Run SGLang in Spark"*:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv ~/.sglang --python 3.12 && source ~/.sglang/bin/activate
export TORCH_CUDA_ARCH_LIST=12.1a TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas
uv pip install sgl-kernel --prerelease=allow --index-url https://docs.sglang.ai/whl/cu130/
uv pip install sglang --prerelease=allow
uv pip install --force-reinstall torch torchvision torchaudio triton --index-url https://download.pytorch.org/whl/cu130
uv pip install flashinfer-python openai scipy ninja
python scripts/x86_cuda/sglang_turing_launch.py --model-path Qwen/Qwen3-32B-AWQ --host 127.0.0.1 \
    --port 30000 --context-length 8192 --mem-fraction-static 0.23 --max-running-requests 32 \
    --attention-backend triton --trust-remote-code
```

The launcher probes the kernels first and changes nothing if `sgl-kernel` was built for `sm_121`.
