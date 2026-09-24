# Experiment 002-tier2-disable-thinking — Tắt Qwen3 hidden thinking cho voting_ensemble

<!-- theo chuẩn experiment/Template/README.md -->

- **Date:** 19/07/2026
- **Author:** hoangnv (qua Claude Code)
- **Status:** done (code fix) — chưa có run thật kiểm chứng trên Qwen3-32B
- **Base model:** `ollama-qwen3-32b` (tag `qwen3-32b-ctx8k`)
- **Method:** không phải training — thay đổi inference config (tắt cờ `think` của Ollama)

## 1. Motivation / Hypothesis

Rà soát lại toàn bộ codebase sau `001-tier1-disable-thinking` và `003-tier3-disable-thinking`
để tìm các `call_llm()` còn thiếu `think=False`. Kiểm tra bằng cách so khớp số lần gọi
`call_llm(` với số lần xuất hiện `think=False` trong từng file:

```
frameworks/tier2/voting_ensemble.py:  call_llm()=1  think=False=0   <- thiếu
frameworks/tier1/cot_sc.py:            call_llm()=1  think=False=0   <- thiếu (xử lý ở 001-tier1)
frameworks/tier1/self_refine.py:       call_llm()=3  think=False=0   <- thiếu (xử lý ở 001-tier1)
frameworks/tier3/*.py:                 khớp 100%                     <- đã xong ở 003-tier3
```

`voting_ensemble.py` (Tier 2, VM14K "Ensemble" — 3 lần shuffle option rồi vote) dùng
prompt gần như y hệt `zero_shot` ("Only response with 1 character") nhưng gọi
`call_llm(model, msgs)` không truyền `think=`, nên với Qwen3-32B sẽ có cùng rủi ro
null-answer/truncation đã ghi nhận ở `001-tier1-disable-thinking`.

## 2. Setup

- **Data:** dự kiến `data/pilot_100.jsonl` (test nhanh) trước khi chạy full
  `data/vm14k_clean.jsonl`.
- **Config:** không đổi `infrastructure/config.py` — chỉ thêm `think=False` vào lời gọi
  `call_llm()` sẵn có trong `voting_ensemble.py`.
- **Δ vs previous:** đúng 1 dòng đổi trong `frameworks/tier2/voting_ensemble.py`.

## 3. What I did (log)

1. Grep so khớp `call_llm(` vs `think=False` trên toàn bộ `frameworks/tier1/`,
   `frameworks/tier2/`, `frameworks/tier3/` để tìm chỗ còn sót.
2. Sửa `frameworks/tier2/voting_ensemble.py` dòng 68:
   ```python
   raw = call_llm(model, msgs)              # trước
   raw = call_llm(model, msgs, think=False)  # sau
   ```
3. Verify cú pháp bằng `ast.parse()` — không lỗi.
4. **Chưa chạy thử** — dừng ở bước code.

## 4. Results

_(để trống — chưa có run nào)_

## 5. Error analysis

_(để trống)_

## 6. Conclusion

_(để trống)_

## 7. Next step

- Chạy `voting_ensemble x ollama-qwen3-32b` trên `pilot_100.jsonl` để kiểm chứng
  không còn null-answer, rồi chạy full `vm14k_clean.jsonl` × 3 seed.
- Cùng đợt rà soát này cũng phát hiện `cot_sc.py`/`self_refine.py` (Tier 1) còn thiếu
  `think=False` — đã sửa và ghi nhận trong
  [`../001-tier1-disable-thinking/README.md`](../001-tier1-disable-thinking/README.md)
  (mục 7 Next step) thay vì tạo experiment riêng, vì cùng phạm vi với experiment đó.

## 8. Artifacts & links

- Code thay đổi: [`frameworks/tier2/voting_ensemble.py`](../../frameworks/tier2/voting_ensemble.py)
- Related: `001-tier1-disable-thinking` (cùng đợt rà soát, xử lý `cot_sc`/`self_refine`),
  `003-tier3-disable-thinking`
