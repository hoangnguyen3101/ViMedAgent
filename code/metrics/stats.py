"""Kiểm định thống kê dùng chung. Hàm thuần, không I/O."""

from __future__ import annotations

import math

from scipy.stats import binomtest

Z_ALPHA_05 = 1.959963985
Z_POWER_80 = 0.8416212336


def _exact_binom_p(b: int, c: int) -> float:
    n = b + c
    return 1.0 if n == 0 else float(binomtest(b, n, 0.5, alternative="two-sided").pvalue)


def mcnemar_pairs(b: int, c: int, n_matched: int = 0) -> dict:
    p = _exact_binom_p(b, c)
    return {"b": b, "c": c, "n_matched": n_matched,
            "p_value": p, "significant": p < 0.05}


def mcnemar(a: list[dict], b: list[dict]) -> dict:
    """McNemar exact trên outcome đúng/sai, ghép theo (id, seed).

    Ghép theo (id, seed) chứ không chỉ id: với nhiều seed, ghép theo id
    sẽ ghi đè và mất cặp.
    """
    idx = {r["_key"]: r for r in b}
    nb = nc = matched = 0
    for x in a:
        y = idx.get(x["_key"])
        if y is None:
            continue
        matched += 1
        if not x["_correct"] and y["_correct"]:
            nb += 1
        elif x["_correct"] and not y["_correct"]:
            nc += 1
    return mcnemar_pairs(nb, nc, matched)


def mcnemar_flags(fa: dict, fb: dict) -> dict:
    """McNemar khi outcome là một cờ nhị phân bất kỳ (vd: có xảy ra NMAR)."""
    keys = fa.keys() & fb.keys()
    nb = sum(1 for k in keys if not fa[k] and fb[k])
    nc = sum(1 for k in keys if fa[k] and not fb[k])
    return mcnemar_pairs(nb, nc, len(keys))


def wilson_ci(k: int, n: int, z: float = Z_ALPHA_05) -> tuple[float, float]:
    """Khoảng tin cậy Wilson (%). Ổn định hơn Wald khi tỷ lệ gần 0 hoặc 1."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    ctr = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, ctr - half) * 100, min(1.0, ctr + half) * 100)


def mcnemar_mde(n_pairs: int, discordant_rate: float = 0.20) -> float:
    """Hiệu ứng nhỏ nhất phát hiện được (pp) ở power 80%, alpha 0.05.

    Báo cáo cùng p-value để 'p > 0.05' không bị đọc nhầm thành
    'không có hiệu ứng' khi thực chất là thiếu dữ liệu.
    """
    if n_pairs <= 0:
        return float("inf")
    return math.sqrt((Z_ALPHA_05 + Z_POWER_80) ** 2 * discordant_rate / n_pairs) * 100


def mean_std(values) -> tuple[float, float]:
    """Trung bình và độ lệch chuẩn MẪU (chia n-1) — quy ước cho n=3 seed."""
    vals = list(values)
    n = len(vals)
    if n == 0:
        return (0.0, 0.0)
    m = sum(vals) / n
    if n == 1:
        return (m, 0.0)
    return (m, math.sqrt(sum((v - m) ** 2 for v in vals) / (n - 1)))


def pct(num: int, den: int):
    return num / den * 100 if den else None
