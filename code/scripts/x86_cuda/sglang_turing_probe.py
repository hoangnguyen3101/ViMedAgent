"""
ViMedAgent v2 — probe which SGLang kernel paths work on this GPU.

Run inside the `sglang` env on the target machine. Prints OK/FAIL for each
kernel family the compat launcher (sglang_turing_launch.py) may need to swap:

  sgl_kernel  : prebuilt AOT kernels (sm80+ only in current wheels)
  jit norm    : sglang.jit_kernel.norm  (nvcc-compiled for this GPU at first use)
  jit rope    : sglang.jit_kernel.rope
  triton awq  : sglang.srt.layers.quantization.awq_triton (pure Triton)
"""
import time
import torch


def clear_cuda_error():
    """cudaGetLastError() via ctypes — a failed kernel launch leaves a
    non-sticky error in the per-thread slot that the *next* checked launch
    (e.g. a JIT kernel) would report as its own. Consume it."""
    import ctypes, glob, os, torch
    cands = glob.glob(os.path.join(os.path.dirname(torch.__file__), "..", "nvidia", "cuda_runtime", "lib", "libcudart.so*"))
    cands += ["libcudart.so.12", "libcudart.so"]
    for c in cands:
        try:
            lib = ctypes.CDLL(c)
            return int(lib.cudaGetLastError())
        except OSError:
            continue
    return -1


def _rms_ref(x, w, eps):
    xf = x.float()
    return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps) * w.float()).to(x.dtype)


def main():
    dev = "cuda"
    cap = torch.cuda.get_device_capability(0)
    print(f"GPU: {torch.cuda.get_device_name(0)}  sm{cap[0]}{cap[1]}  torch {torch.__version__}")
    dt = torch.float16
    H = 5120                                   # Qwen3-32B hidden size
    x = torch.randn(8, H, device=dev, dtype=dt)
    w = torch.rand(H, device=dev, dtype=dt) + 0.5
    eps = 1e-6
    ref = _rms_ref(x, w, eps)

    # 1. sgl_kernel AOT
    try:
        import sgl_kernel
        out = sgl_kernel.rmsnorm(x, w, eps)
        torch.cuda.synchronize()
        print(f"[sgl_kernel.rmsnorm]      OK  maxdiff={(out-ref).abs().max().item():.3e}")
    except Exception as e:
        print(f"[sgl_kernel.rmsnorm]      FAIL {str(e)[:90]}")
    code = clear_cuda_error()
    print(f"[clear_cuda_error]        consumed stale error code {code} (0 = none)")

    # 2. JIT norm
    try:
        from sglang.jit_kernel.norm import rmsnorm as jrms, fused_add_rmsnorm as jfused
        t0 = time.time()
        out = torch.empty_like(x)
        jrms(x, w, out, eps)
        torch.cuda.synchronize()
        print(f"[jit rmsnorm]             OK  maxdiff={(out-ref).abs().max().item():.3e}  (first call {time.time()-t0:.1f}s incl. compile)")
        xx, rr = x.clone(), torch.randn_like(x)
        ref2 = _rms_ref((x + rr), w, eps)
        t0 = time.time()
        jfused(xx, rr, w, eps)
        torch.cuda.synchronize()
        print(f"[jit fused_add_rmsnorm]   OK  maxdiff={(xx-ref2).abs().max().item():.3e}  ({time.time()-t0:.1f}s)")
    except Exception as e:
        print(f"[jit norm]                FAIL {str(e)[:200]}")

    # 3. JIT rope (neox, head 128, like Qwen3)
    try:
        from sglang.jit_kernel.rope import apply_rope_with_cos_sin_cache_inplace
        n, nq, nk, hs = 8, 64, 8, 128
        pos = torch.randint(0, 4096, (n,), device=dev)
        q = torch.randn(n, nq * hs, device=dev, dtype=dt)
        k = torch.randn(n, nk * hs, device=dev, dtype=dt)
        inv = 1.0 / (1e6 ** (torch.arange(0, hs, 2, device=dev).float() / hs))
        t = torch.arange(8192, device=dev).float()
        freqs = torch.outer(t, inv)
        cache = torch.cat([freqs.cos(), freqs.sin()], -1)  # [8192, 128] fp32

        def ref_rope(v, nh):
            v3 = v.view(n, nh, hs).float()
            c = cache[pos, : hs // 2].unsqueeze(1); s = cache[pos, hs // 2 :].unsqueeze(1)
            x1, x2 = v3[..., : hs // 2], v3[..., hs // 2 :]
            return torch.cat([x1 * c - x2 * s, x2 * c + x1 * s], -1).view(n, nh * hs).to(dt)
        rq, rk = ref_rope(q, nq), ref_rope(k, nk)
        t0 = time.time()
        apply_rope_with_cos_sin_cache_inplace(pos, q, k, hs, cache, True)
        torch.cuda.synchronize()
        print(f"[jit rope]                OK  maxdiff q={(q-rq).abs().max().item():.3e} k={(k-rk).abs().max().item():.3e}  ({time.time()-t0:.1f}s)")
    except Exception as e:
        print(f"[jit rope]                FAIL {str(e)[:200]}")

    # 4. Triton AWQ dequant vs decomposition reference
    try:
        from sglang.srt.layers.quantization.awq_triton import (
            awq_dequantize_triton, awq_dequantize_decomposition,
        )
        K, N, G = 512, 1024, 128
        qweight = torch.randint(0, 2**31 - 1, (K, N // 8), device=dev, dtype=torch.int32)
        qzeros = torch.randint(0, 2**31 - 1, (K // G, N // 8), device=dev, dtype=torch.int32)
        scales = torch.rand(K // G, N, device=dev, dtype=dt) * 0.01
        t0 = time.time()
        out = awq_dequantize_triton(qweight, scales, qzeros)
        torch.cuda.synchronize()
        ref4 = awq_dequantize_decomposition(qweight, scales, qzeros)
        print(f"[triton awq_dequantize]   OK  maxdiff={(out-ref4).abs().max().item():.3e}  ({time.time()-t0:.1f}s)")
    except Exception as e:
        print(f"[triton awq_dequantize]   FAIL {str(e)[:200]}")

    # 5. Triton silu_and_mul is not needed — native torch is fine. Check sgl_kernel one anyway.
    try:
        import sgl_kernel
        y = torch.randn(8, 2 * H, device=dev, dtype=dt)
        o = torch.empty(8, H, device=dev, dtype=dt)
        sgl_kernel.silu_and_mul(y, o); torch.cuda.synchronize()
        print("[sgl_kernel.silu_and_mul] OK")
    except Exception as e:
        print(f"[sgl_kernel.silu_and_mul] FAIL {str(e)[:90]}")


if __name__ == "__main__":
    main()
