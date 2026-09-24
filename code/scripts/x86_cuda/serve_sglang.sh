#!/usr/bin/env bash
# ViMedAgent v2 — launch the SGLang OpenAI-compatible server.
#
# Usage (run inside tmux, it stays in the foreground):
#   bash scripts/x86_cuda/serve_sglang.sh                         # Qwen/Qwen3-32B-AWQ on :30000
#   MODEL=Qwen/Qwen3-14B bash scripts/x86_cuda/serve_sglang.sh    # any HF repo id in config.py
#
# Env knobs (all optional):
#   MODEL            HF repo id / local path   (default Qwen/Qwen3-32B-AWQ)
#   PORT             default 30000              → SGLANG_BASE_URL=http://127.0.0.1:$PORT/v1
#   HOST             default 127.0.0.1
#   ENV_NAME         conda env with sglang      (default sglang)
#   CONTEXT_LEN      default 8192  — same cap as the Ollama *-ctx8k tags used in v1
#   MEM_FRACTION     fraction of GPU memory SGLang may reserve (weights + KV cache).
#                    default 0.85; lower it when the GPU is shared (e.g. 0.55).
#   MAX_RUNNING      max concurrent sequences in a batch (default 64)
#   TP               tensor parallel size (default 1)
#   DTYPE            auto | float16 | bfloat16  (auto: fp16 if sm<80 else bf16)
#   ATTN_BACKEND     auto | flashinfer | triton | fa3 (auto: triton if sm<80)
#   CUDA_HOME        CUDA toolkit >= 12 with nvcc — only needed on sm<80 for the
#                    JIT kernels; auto-detected from /usr/local/cuda-12* if unset
#   EXTRA_ARGS       anything else passed straight to sglang.launch_server
#
# DGX Spark / ASUS Ascent GX10 (GB10, sm121, aarch64): the x86 conda env does not
#   transfer. Either use scripts/serve_gx10.sh (pre-built GB10 image,
#   recommended) or a native venv built per README "GX10" section, then run this
#   script with ENV_NAME pointing at it. 128GB unified memory fits Qwen3-32B in
#   bf16 (MODEL=Qwen/Qwen3-32B) or FP8 (MODEL=Qwen/Qwen3-32B-FP8).
#
# Pre-Ampere GPUs (sm70/sm75 — V100, T4, Quadro RTX 8000, RTX 20xx):
#   sgl-kernel wheels have no sm75 code, so the server is started through
#   scripts/x86_cuda/sglang_turing_launch.py, which swaps in JIT/Triton/PyTorch
#   fallbacks (see that file). Needs ninja + a C++20-capable g++ (>= 10) —
#   scripts/x86_cuda/install_sglang.sh installs GCC 13 into the conda env for this.
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen3-32B-AWQ}"
PORT="${PORT:-30000}"
HOST="${HOST:-127.0.0.1}"
ENV_NAME="${ENV_NAME:-sglang}"
CONTEXT_LEN="${CONTEXT_LEN:-8192}"
MEM_FRACTION="${MEM_FRACTION:-0.85}"
MAX_RUNNING="${MAX_RUNNING:-64}"
TP="${TP:-1}"
DTYPE="${DTYPE:-auto}"
ATTN_BACKEND="${ATTN_BACKEND:-auto}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDA_BASE="$(conda info --base)"
ENV_BIN="$CONDA_BASE/envs/$ENV_NAME/bin"
PY="$ENV_BIN/python"
[ -x "$PY" ] || { echo "conda env '$ENV_NAME' not found — run scripts/x86_cuda/install_sglang.sh first"; exit 1; }
export PATH="$ENV_BIN:$PATH"     # ninja, conda g++

# Compute capability without creating a CUDA context.
SM="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '. ' || true)"
SM="${SM:-80}"

if [ "$DTYPE" = "auto" ]; then
  if [ "$SM" -lt 80 ]; then DTYPE=float16; else DTYPE=bfloat16; fi
fi
if [ "$ATTN_BACKEND" = "auto" ]; then
  if [ "$SM" -lt 80 ]; then
    ATTN_BACKEND=triton          # Turing/Volta: no FlashInfer/FA3 kernels
  elif [ "$SM" -ge 120 ]; then
    ATTN_BACKEND=triton          # GB10 / RTX 50xx (sm120/121): triton is what the
                                 # DGX Spark community images recommend; flashinfer
                                 # works on newer builds — override with ATTN_BACKEND=flashinfer
  else
    ATTN_BACKEND=flashinfer      # Ampere / Ada / Hopper
  fi
fi

LAUNCHER=(-m sglang.launch_server)
COMPAT_ARGS=()
if [ "$SM" -lt 80 ]; then
  LAUNCHER=("$HERE/sglang_turing_launch.py")
  COMPAT_ARGS=(--sampling-backend pytorch)
  # nvcc >= 12 for the JIT kernels (the system default may be an old toolkit)
  if [ -z "${CUDA_HOME:-}" ]; then
    for d in /usr/local/cuda-12.9 /usr/local/cuda-12.8 /usr/local/cuda-12.6 /usr/local/cuda-12.4 /usr/local/cuda-12 /usr/local/cuda; do
      if [ -x "$d/bin/nvcc" ] && "$d/bin/nvcc" --version | grep -q "release 12"; then CUDA_HOME="$d"; break; fi
    done
  fi
  if [ -n "${CUDA_HOME:-}" ]; then
    export CUDA_HOME; export PATH="$CUDA_HOME/bin:$PATH"
  else
    echo "[serve] WARNING: no CUDA 12 toolkit found — JIT norm/rope will fail and fall back to PyTorch ops (slower)"
  fi
  # C++20-capable host compiler for nvcc (conda-forge gxx from install_sglang.sh)
  if [ -x "$ENV_BIN/x86_64-conda-linux-gnu-g++" ]; then
    export CXX="$ENV_BIN/x86_64-conda-linux-gnu-g++"
    export NVCC_PREPEND_FLAGS="-ccbin $CXX ${NVCC_PREPEND_FLAGS:-}"
  fi
  echo "[serve] sm$SM < 80 → compat launcher (JIT norm/rope, Triton AWQ, PyTorch sampling)"
  echo "[serve] CUDA_HOME=${CUDA_HOME:-<none>}  CXX=${CXX:-<system>}"
fi

echo "[serve] model=$MODEL  port=$PORT  sm=$SM  dtype=$DTYPE  attn=$ATTN_BACKEND"
echo "[serve] ctx=$CONTEXT_LEN  mem_fraction=$MEM_FRACTION  max_running=$MAX_RUNNING  tp=$TP"
echo "[serve] clients should set: export SGLANG_BASE_URL=http://$HOST:$PORT/v1"

exec "$PY" "${LAUNCHER[@]}" \
  --model-path "$MODEL" \
  --host "$HOST" --port "$PORT" \
  --dtype "$DTYPE" \
  --attention-backend "$ATTN_BACKEND" \
  --context-length "$CONTEXT_LEN" \
  --mem-fraction-static "$MEM_FRACTION" \
  --max-running-requests "$MAX_RUNNING" \
  --tp "$TP" \
  --trust-remote-code \
  "${COMPAT_ARGS[@]}" \
  $EXTRA_ARGS
