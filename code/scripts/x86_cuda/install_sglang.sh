#!/usr/bin/env bash
# ViMedAgent v2 — install SGLang into its own conda env.
#
# Usage:  bash scripts/install_sglang.sh [env_name]      (default: sglang)
#
# Requirements: conda, NVIDIA driver >= 525 (CUDA 12.x wheels use minor-version
# compatibility), internet. Tested on driver 550.163 / Quadro RTX 8000 (sm75)
# and expected to work unchanged on Ampere+ (A100/A6000/H100).
# Installs into a separate env so the v1 "agent" env stays untouched.
set -euo pipefail

# GB10 boxes (DGX Spark / ASUS Ascent GX10) are aarch64 + sm_121: PyPI x86 wheels do not
# apply and sgl-kernel from PyPI has no sm_121 code. Use the pre-built container instead.
if [ "$(uname -m)" = "aarch64" ] && [ "${FORCE_NATIVE:-0}" != "1" ]; then
  echo "[install] aarch64 detected (GB10 / DGX Spark / GX10?)."
  echo "          Do NOT use this script here. Serve with:  bash scripts/serve_gx10.sh"
  echo "          and install only the client:            bash scripts/install_gx10.sh"
  echo "          (native venv recipe: README section 'GX10'; set FORCE_NATIVE=1 to bypass this check)"
  exit 2
fi

ENV_NAME="${1:-sglang}"
PY_VER="${PY_VER:-3.11}"
CONDA_BASE="$(conda info --base)"
PY="$CONDA_BASE/envs/$ENV_NAME/bin/python"

if [ ! -x "$PY" ]; then
  echo "[install] creating conda env $ENV_NAME (python $PY_VER)"
  conda create -y -n "$ENV_NAME" "python=$PY_VER"
fi

# uv is ~10x faster than pip at resolving the torch/flashinfer tree; fall back to pip.
PKGS=("sglang[all]" openai "huggingface_hub[cli]" ninja)
if command -v uv >/dev/null 2>&1; then
  uv pip install --python "$PY" "${PKGS[@]}"
else
  "$PY" -m pip install --upgrade pip
  "$PY" -m pip install "${PKGS[@]}"
fi

# Pre-Ampere GPUs (sm<80): sgl-kernel has no prebuilt code for them, so the
# serve script uses nvcc-JIT kernels that need a C++20 host compiler (g++>=10).
# Ubuntu 20.04 ships g++ 9 → install GCC 13 into the env (~300MB).
SM="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '. ' || true)"
if [ -n "$SM" ] && [ "$SM" -lt 80 ]; then
  echo "[install] sm$SM < 80 → installing GCC 13 toolchain into env for JIT kernels"
  conda install -y -n "$ENV_NAME" -c conda-forge "gxx_linux-64=13" "gcc_linux-64=13"
  if ! ls /usr/local/cuda-12*/bin/nvcc /usr/local/cuda/bin/nvcc >/dev/null 2>&1; then
    echo "[install] WARNING: no CUDA 12 toolkit (nvcc) found under /usr/local — JIT kernels"
    echo "          will be skipped and PyTorch fallbacks used (works, but slower)."
  fi
fi

echo
echo "[install] versions:"
"$PY" - <<'PYEOF'
import torch, sglang, openai
print("  torch  ", torch.__version__, "| cuda", torch.version.cuda, "| gpu ok:", torch.cuda.is_available())
print("  sglang ", sglang.__version__)
print("  openai ", openai.__version__)
if torch.cuda.is_available():
    cc = torch.cuda.get_device_capability(0)
    print("  device ", torch.cuda.get_device_name(0), "| sm%d%d" % cc)
    if cc < (8, 0):
        print("  NOTE: sm < 80 → serve_sglang.sh will use scripts/sglang_turing_launch.py automatically")
PYEOF
echo
echo "[install] done. Next: bash scripts/x86_cuda/serve_sglang.sh   (in a tmux window)"
echo "          optional check on sm<80: python scripts/x86_cuda/sglang_turing_probe.py"
