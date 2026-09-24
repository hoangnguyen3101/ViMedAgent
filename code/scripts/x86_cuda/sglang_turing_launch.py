"""
ViMedAgent v2 — SGLang compat launcher for pre-Ampere NVIDIA GPUs (sm70 / sm75:
V100, T4, Quadro RTX 8000, RTX 20xx).

Why this exists
---------------
Current sgl-kernel wheels ship SASS only for sm80+ (checked with cuobjdump on
sgl_kernel 0.3.21: sm_80/89/90/100/120, no PTX). On a Turing card every fused
op from sgl_kernel dies with "no kernel image is available for execution on the
device" — the first one being RMSNorm during CUDA-graph capture, which kills the
scheduler. Everything *else* SGLang needs on a dense fp16 model has a pure
PyTorch, Triton, or nvcc-JIT alternative already inside the package:

  op                        sgl_kernel (AOT)     what we use instead on sm<80
  ------------------------  -------------------  ---------------------------------------
  RMSNorm / fused add-norm  no sm75 image        sglang.jit_kernel.norm (nvcc JIT for this
                                                 GPU) if it compiles, else forward_native
  RotaryEmbedding           no sm75 image        sglang.jit_kernel.rope (nvcc JIT) if it
                                                 compiles, else forward_native
  SiluAndMul / GeluAndMul   (varies)             forward_native (torch)
  AWQ int4 dequant          no sm75 image        awq_triton.awq_dequantize_triton
  attention                 flashinfer/fa3       --attention-backend triton  (serve script)
  sampling                  flashinfer           --sampling-backend pytorch  (serve script)

The launcher does NOT edit the installed sglang package. It monkey-patches
`MultiPlatformOp.dispatch_forward` (the single place every fused layer picks its
backend) and the AWQ module's dequantize symbol, then hands over to
`sglang.launch_server` with the original CLI args. SGLang spawns its scheduler
with multiprocessing "spawn", which re-imports this file as __mp_main__, so the
patch is installed in every child automatically.

On sm80+ the launcher is a no-op pass-through (probe says sgl_kernel works).

Usage (what scripts/x86_cuda/serve_sglang.sh runs when it detects sm<80):
    python scripts/x86_cuda/sglang_turing_launch.py --model-path Qwen/Qwen3-32B-AWQ \
        --dtype float16 --attention-backend triton --sampling-backend pytorch ...

Env:
    SGLANG_TURING_COMPAT = auto (default: hook always, probe decides) | 1 (force) | 0 (never)
    SGLANG_TURING_NO_JIT = 1  → skip nvcc-JIT norm/rope, use forward_native for all
    SGLANG_TURING_AWQ_GEMM_MAX_M = 0 (default: off) → >0 enables the fused int4 Triton
                              GEMM for batches with <= M tokens; measured slower on RTX 8000
    SGLANG_TURING_AWQ_GEMM_SPLIT_K = 8
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys

logger = logging.getLogger("sglang_turing_compat")
if not logger.handlers:
    _h = logging.StreamHandler(sys.stderr)
    _h.setFormatter(logging.Formatter("[turing-compat] %(message)s"))
    logger.addHandler(_h)
    logger.setLevel(logging.INFO)


# ── decide once, without creating a CUDA context (nvidia-smi / NVML) ─────────

def _compute_capability_via_nvidia_smi() -> tuple[int, int] | None:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=20,
        ).stdout.strip().splitlines()
        vis = os.environ.get("CUDA_VISIBLE_DEVICES")
        idx = 0
        if vis and vis.split(",")[0].strip().isdigit():
            idx = int(vis.split(",")[0])
        major, minor = out[idx].strip().split(".")
        return int(major), int(minor)
    except Exception:
        return None


def compat_needed() -> bool:
    """auto (default): always install the hook — it is a no-op wherever the
    sgl_kernel probe succeeds (Ampere/Ada/Hopper, or a GB10 build of
    sgl-kernel), and only swaps kernels where the AOT wheel has no image for
    the device (Turing today; any future arch/wheel mismatch tomorrow)."""
    mode = os.environ.get("SGLANG_TURING_COMPAT", "auto").lower()
    if mode in ("0", "false", "off"):
        return False
    if mode in ("1", "true", "on"):
        return True
    cap = _compute_capability_via_nvidia_smi()
    if cap is not None and cap < (8, 0):
        logger.info(f"sm{cap[0]}{cap[1]} < sm80 → compat mode expected")
    return True


# ── kernel probes (run lazily inside the GPU process) ────────────────────────

_PROBE: dict = {}


def _clear_cuda_error() -> int:
    """Consume cudaGetLastError(). A failed sgl_kernel launch ("no kernel
    image") leaves a non-sticky error in the per-thread slot; the next kernel
    whose wrapper checks the error slot (our JIT norm) would otherwise report
    that stale error as its own failure."""
    import ctypes
    import glob
    import torch
    cands = glob.glob(os.path.join(os.path.dirname(torch.__file__), "..", "nvidia",
                                   "cuda_runtime", "lib", "libcudart.so*"))
    cands += ["libcudart.so.12", "libcudart.so"]
    for c in cands:
        try:
            return int(ctypes.CDLL(c).cudaGetLastError())
        except OSError:
            continue
    return -1


def _probe_once() -> dict:
    """Try each candidate kernel on tiny tensors. Called from the scheduler
    process at first layer construction (before any CUDA graph capture)."""
    if _PROBE:
        return _PROBE
    import torch

    dev = "cuda"
    dt = torch.get_default_dtype() if torch.get_default_dtype() in (torch.float16, torch.bfloat16) else torch.float16
    no_jit = os.environ.get("SGLANG_TURING_NO_JIT", "0") == "1"

    def ok(name, fn):
        try:
            fn()
            torch.cuda.synchronize()
            _PROBE[name] = True
        except Exception as e:  # noqa: BLE001
            _PROBE[name] = False
            logger.info(f"probe {name}: unavailable ({str(e).splitlines()[0][:120]})")
        _clear_cuda_error()          # never let one probe poison the next
        return _PROBE[name]

    x = torch.randn(4, 1024, device=dev, dtype=dt)
    w = torch.ones(1024, device=dev, dtype=dt)

    import sgl_kernel
    ok("sgl_kernel", lambda: sgl_kernel.rmsnorm(x, w, 1e-6))

    if not no_jit:
        from sglang.jit_kernel import norm as jn
        ok("jit_norm", lambda: (jn.rmsnorm(x, w, torch.empty_like(x), 1e-6),
                                jn.fused_add_rmsnorm(x.clone(), x.clone(), w, 1e-6)))
        from sglang.jit_kernel.rope import apply_rope_with_cos_sin_cache_inplace
        pos = torch.arange(4, device=dev)
        q = torch.randn(4, 8 * 128, device=dev, dtype=dt)
        k = torch.randn(4, 2 * 128, device=dev, dtype=dt)
        cache = torch.randn(64, 128, device=dev, dtype=torch.float32)
        ok("jit_rope", lambda: apply_rope_with_cos_sin_cache_inplace(pos, q, k, 128, cache, True))
    else:
        _PROBE["jit_norm"] = False
        _PROBE["jit_rope"] = False

    logger.info(f"kernel probe: {_PROBE}")
    return _PROBE


# ── the patches ──────────────────────────────────────────────────────────────

_PATCHED = {"done": False}


def _install_fallbacks():
    """Swap module-level kernel symbols. Idempotent."""
    if _PATCHED["done"]:
        return
    _PATCHED["done"] = True
    import torch

    p = _probe_once()

    # AWQ: pure Triton. Two paths, picked per call by token count M:
    #   M <= SGLANG_TURING_AWQ_GEMM_MAX_M  → awq_gemm_triton: fused int4 GEMM that
    #        reads the packed int4 weight directly (decode is weight-bandwidth
    #        bound, so this is several x faster than dequantizing 64GB of fp16
    #        per step).
    #   M  > threshold                     → awq_dequantize_triton + torch.matmul
    #        (prefill: compute bound, cuBLAS fp16 wins).
    # vLLM uses this heuristic on ROCm (threshold 256). Measured on Quadro RTX
    # 8000 (scripts/sglang_awq_gemm_bench.py, 2026-09-11) the fused Triton
    # kernel was SLOWER than dequant+cuBLAS at every M (0.6-0.8x at M<=32,
    # 0.1-0.4x above), so the default here is 0 = disabled. Left in as an
    # opt-in for other pre-Ampere cards where the balance may differ.
    try:
        import sglang.srt.layers.quantization.awq as awq
        from sglang.srt.layers.quantization.awq_triton import (
            awq_dequantize_triton, awq_gemm_triton,
        )
        awq.awq_dequantize = awq_dequantize_triton
        max_m = int(os.environ.get("SGLANG_TURING_AWQ_GEMM_MAX_M", "0"))
        split_k = int(os.environ.get("SGLANG_TURING_AWQ_GEMM_SPLIT_K", "8"))

        def apply(self, layer, x, bias=None):
            qweight, scales, qzeros = layer.qweight, layer.scales, layer.qzeros
            pack_factor = self.quant_config.pack_factor
            out_shape = x.shape[:-1] + (qweight.shape[-1] * pack_factor,)
            reshaped_x = x.reshape(-1, x.shape[-1])
            if 0 < reshaped_x.shape[0] <= max_m:
                out = awq_gemm_triton(reshaped_x.contiguous(), qweight, scales, qzeros, split_k)
            else:
                out = torch.matmul(reshaped_x, awq_dequantize_triton(qweight, scales, qzeros))
            if bias is not None:
                out.add_(bias)
            return out.reshape(out_shape)

        if max_m > 0:
            awq.AWQLinearMethod.apply = apply
        logger.info(f"awq_dequantize -> awq_dequantize_triton; fused awq_gemm_triton for M<={max_m}")
    except Exception as e:  # noqa: BLE001
        logger.warning(f"could not patch AWQ path: {e}")

    # RMSNorm: JIT kernels compiled by nvcc for this GPU, wrapped to the
    # sgl_kernel call convention used in layernorm.forward_cuda.
    if p.get("jit_norm"):
        import sglang.srt.layers.layernorm as ln
        from sglang.jit_kernel import norm as jn

        def rmsnorm(x, weight, eps=1e-6, out=None, enable_pdl=None):
            out = torch.empty_like(x) if out is None else out
            jn.rmsnorm(x, weight, out, eps)
            return out

        def fused_add_rmsnorm(x, residual, weight, eps=1e-6, enable_pdl=None):
            jn.fused_add_rmsnorm(x, residual, weight, eps)

        ln.rmsnorm = rmsnorm
        ln.fused_add_rmsnorm = fused_add_rmsnorm
        logger.info("layernorm.rmsnorm/fused_add_rmsnorm -> sglang.jit_kernel.norm")


def _choose_forward(op) -> str:
    """Return 'cuda' or 'native' for a MultiPlatformOp instance on sm<80."""
    p = _probe_once()
    cls = type(op).__name__
    if cls in ("RMSNorm",):
        # forward_cuda now calls the JIT wrappers installed above. Anything
        # else in that method (variance_size_override, batch-invariant mode)
        # already falls back to forward_native on its own.
        return "cuda" if p.get("jit_norm") else "native"
    if "Rotary" in cls:
        # forward_cuda → jit_kernel.rope (main path). The 'fallback kernel'
        # path (head_size not in 64/128/256/512) uses jit_kernel.pos_enc which
        # we did not probe — stay native there.
        if p.get("jit_rope") and not getattr(op, "use_fallback_kernel", False):
            return "cuda"
        return "native"
    # SiluAndMul, GeluAndMul, GemmaRMSNorm, QuickGELU, ... — torch is fine.
    return "native"


def install():
    from sglang.srt.layers.utils import multi_platform as mp_mod

    MultiPlatformOp = mp_mod.MultiPlatformOp
    if getattr(MultiPlatformOp, "_turing_compat_installed", False):
        return
    orig_dispatch = MultiPlatformOp.dispatch_forward

    def dispatch_forward(self):
        import torch
        if not torch.cuda.is_available():
            return orig_dispatch(self)
        _install_fallbacks()
        if _probe_once().get("sgl_kernel"):
            return orig_dispatch(self)          # Ampere+: nothing to do
        which = _choose_forward(self)
        fn = self.forward_cuda if which == "cuda" else self.forward_native
        logger.debug(f"{type(self).__name__} -> forward_{which}")
        return fn

    MultiPlatformOp.dispatch_forward = dispatch_forward
    MultiPlatformOp._turing_compat_installed = True
    logger.info("MultiPlatformOp.dispatch_forward hooked (sm<80 compat mode)")


# Module level: runs in the launcher process AND in every spawned child
# (multiprocessing 'spawn' re-imports __main__ as __mp_main__).
if compat_needed():
    os.environ["SGLANG_TURING_COMPAT"] = "1"   # children skip the nvidia-smi probe
    install()

if __name__ == "__main__":
    import runpy
    # NOTE: alter_sys must stay False. With alter_sys=True runpy swaps
    # sys.modules["__main__"] for sglang.launch_server, and multiprocessing
    # "spawn" then re-imports *that* file in the scheduler child instead of
    # this one — the hook above would never be installed where it matters.
    sys.argv = ["sglang.launch_server", *sys.argv[1:]]
    runpy.run_module("sglang.launch_server", run_name="__main__", alter_sys=False)
