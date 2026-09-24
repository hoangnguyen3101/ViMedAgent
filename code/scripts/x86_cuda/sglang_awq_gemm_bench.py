"""
ViMedAgent v2 — AWQ int4 matmul micro-benchmark on this GPU.

Compares the two Triton paths available on pre-Ampere cards for one AWQ
linear layer at decode-like (M small) and prefill-like (M large) shapes:
  A) dequantize whole weight to fp16 (awq_dequantize_triton) + torch.matmul
  B) fused int4 GEMM (awq_gemm_triton, reads int4 weights directly)
Prints max abs diff and time per call so the launcher's M threshold can be set.
"""
import time
import torch
from sglang.srt.layers.quantization.awq_triton import awq_dequantize_triton, awq_gemm_triton

dev, dt = "cuda", torch.float16
G = 128
shapes = [("qkv_proj  K=5120 N=10240", 5120, 10240), ("gate_up   K=5120 N=51200", 5120, 51200), ("down      K=25600 N=5120", 25600, 5120)]

def bench(fn, iters=10):
    fn(); torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(iters): fn()
    torch.cuda.synchronize()
    return (time.time() - t0) / iters * 1000

for name, K, N in shapes:
    qweight = torch.randint(0, 2**31 - 1, (K, N // 8), device=dev, dtype=torch.int32)
    qzeros = torch.randint(0, 2**31 - 1, (K // G, N // 8), device=dev, dtype=torch.int32)
    scales = (torch.rand(K // G, N, device=dev, dtype=dt) * 0.02)
    print(f"\n{name}")
    for M in (1, 8, 32, 64, 128, 256, 1024):
        x = torch.randn(M, K, device=dev, dtype=dt)
        a = lambda: torch.matmul(x, awq_dequantize_triton(qweight, scales, qzeros))
        b = lambda: awq_gemm_triton(x, qweight, scales, qzeros, 8)
        ra, rb = a(), b()
        diff = (ra.float() - rb.float()).abs().max().item()
        ta, tb = bench(a), bench(b)
        print(f"  M={M:5d}  dequant+matmul {ta:7.2f} ms | fused gemm {tb:7.2f} ms | speedup x{ta/tb:4.1f} | maxdiff {diff:.3e}")
