# Experiment 003-tier3-disable-thinking — Tắt Qwen3 hidden thinking cho toàn bộ Tier 3

<!-- theo chuẩn experiment/Template/README.md -->

- **Date:** 13/07/2026
- **Author:** hoangnv (qua Claude Code)
- **Status:** done (code fix) — chưa có run thật để lấy số liệu kiểm chứng, xem mục 7
- **Base model:** `ollama-qwen3-32b` (tag `qwen3-32b-ctx8k`, Qwen3 32.8B, Q4_K_M, num_ctx=8192)
- **Method:** không phải training — thay đổi inference config (tắt cờ `think` của Ollama),
  áp dụng phòng ngừa trước khi chạy, dựa trên bài học từ `001-tier1-disable-thinking`

## 1. Motivation / Hypothesis

Chuẩn bị chạy Tier 3 (`medagents`, `medagents_vi`, `mdagents`, `mdagents_vi`, `mad_debate`)
với `ollama-qwen3-32b`. `001-tier1-disable-thinking` đã chứng minh Qwen3 tự sinh khối
suy luận ẩn `<think>` trước khi trả lời, có thể ăn hết token budget và gây `null` tới
41.8% nếu không tắt (`think=False`). Trước khi đầu tư thời gian chạy Tier 3 (chậm hơn
Tier 1 nhiều lần do nhiều agent/vòng/bước), rà lại toàn bộ 5 file Tier 3 xem đã áp dụng
`think=False` chưa.

**Phát hiện:** grep `call_llm(` trong `frameworks/tier3/*.py` — **không file nào** truyền
`think=`. Đáng chú ý, [medagents.py:248](../../frameworks/tier3/medagents.py#L248) từng có
1 workaround thủ công cho đúng vấn đề này nhưng vá sai cách:
```python
raw_dec = call_llm(model, msgs, max_tokens=64)  # 64 instead of MAX_TOKENS_SHORT=16: allows <think> models to output letter
```
Comment cho thấy có nhận thức về vấn đề, nhưng tăng từ 16 lên 64 token là không đủ —
theo số liệu đo được ở `001-tier1-disable-thinking`, thinking của Qwen3 tốn 300-4096+
token, không phải vài chục token.

**Giả thuyết:** áp `think=False` cho toàn bộ `call_llm()` trong Tier 3 sẽ ngăn bug
null-answer/truncation tương tự lặp lại, trước khi tốn thời gian chạy thật.

## 2. Setup

- **Data:** chưa chạy — dự kiến `data/pilot_100.jsonl` (test nhanh) rồi `data/vm14k_clean.jsonl`
  (full) sau khi có kết quả pilot ổn.
- **Config:** không đổi tham số nào trong `infrastructure/config.py` — chỉ thêm `think=False`
  vào lời gọi `call_llm()` sẵn có ở từng file Tier 3.
- **Hardware:** 1x Quadro RTX 8000 (49GB), Ollama server local.
- **Δ vs previous:** so với `001-tier1-disable-thinking` (chỉ sửa `zero_shot.py`+`cot.py`),
  lần này mở rộng cùng 1 fix sang toàn bộ Tier 3 — không sửa hạ tầng
  (`infrastructure/llm_client.py` đã có sẵn tham số `think` từ trước, dùng lại nguyên vẹn).

## 3. What I did (log)

1. Grep toàn bộ `call_llm(` trong `frameworks/tier3/*.py` → 21 chỗ, không chỗ nào có `think=`.
2. Phát hiện workaround sai ở `medagents.py:248` (tăng `max_tokens` 16→64 thay vì tắt thinking).
3. Sửa cả 21 chỗ, thêm `think=False`:
   - `mad_debate.py`: 2 chỗ (round 1 + rounds 2+)
   - `mdagents.py`: 5 chỗ (complexity check, PCC, MDT agent×3-vote, ICT expert, ICT synthesizer)
   - `mdagents_vi.py`: 4 chỗ (PCC, MDT agent, ICT expert, ICT synthesizer)
   - `medagents.py`: 5 chỗ (expert gathering, analysis, report, consultation, decision)
   - `medagents_vi.py`: 4 chỗ (analysis, report, consultation, decision)
4. Đồng thời revert workaround ở `medagents.py:248`: `max_tokens=64` → `max_tokens=MAX_TOKENS_SHORT`
   (=16, giá trị gốc trước workaround) — lý do tồn tại của việc tăng token không còn nữa
   khi đã tắt thinking. *(Thay đổi ngoài phạm vi yêu cầu ban đầu, ghi chú lại ở đây cho rõ.)*
5. Verify cú pháp cả 5 file bằng `ast.parse()` — không lỗi.
6. **Chưa chạy thử** — dừng ở bước code, chờ user tự chạy (theo yêu cầu không tự chạy hộ).

## 4. Results

_(để trống — chưa có run nào. Điền sau khi chạy pilot_100 hoặc vm14k_clean.)_

## 5. Error analysis

_(để trống — sau khi chạy, nên kiểm tra riêng theo từng framework:)_
- null rate mỗi framework (nếu vẫn >0%, có thể do bước nào đó có prompt quá phức tạp
  khiến response vượt cả `max_tokens` dù không có thinking)
- với `mad_debate` cụ thể: đã quan sát ở pilot test với Llama-3.1-8B (không liên quan
  thinking) hiện tượng "đồng thuận giả" (3 agent hội tụ cùng đáp án sai) — hiệu ứng này
  độc lập với thinking, cần theo dõi riêng khi chạy với Qwen3-32B.
- phân bố predicted-answer vs gold (như đã làm ở `001-tier1-fix-parse-answer`) để phát
  hiện sớm nếu `parse_answer()` vẫn lệch với văn bản multi-agent — đặc biệt `mad_debate`
  không yêu cầu marker `"Final answer: X"` tường minh trong prompt (chỉ nói
  "answer (A/B/C/D)"), rủi ro cao hơn `medagents`/`mdagents` (các file này có
  `prompt_decision`/`prompt_ict_synthesizer` riêng, cần kiểm tra format yêu cầu).

## 6. Conclusion

_(để trống — điền sau khi có kết quả thật)_

## 7. Next step

- Chạy `pilot_100.jsonl` cho cả 5 framework Tier 3 với `ollama-qwen3-32b` để kiểm chứng
  giả thuyết (null rate ~0%, không có response bị `finish_reason: length`).
- Nếu ổn, chạy full `vm14k_clean.jsonl` × 3 seed.
- Kiểm tra riêng format "final answer" mà mỗi `prompt_*` trong Tier 3 yêu cầu — nếu framework
  nào không dùng đúng marker `"Final answer: X"`, `parse_answer()` sẽ phải dựa vào fallback
  yếu hơn (xem `001-tier1-fix-parse-answer`) dù đã tắt thinking.

## 8. Artifacts & links

- Code thay đổi: [`frameworks/tier3/mad_debate.py`](../../frameworks/tier3/mad_debate.py),
  [`frameworks/tier3/mdagents.py`](../../frameworks/tier3/mdagents.py),
  [`frameworks/tier3/mdagents_vi.py`](../../frameworks/tier3/mdagents_vi.py),
  [`frameworks/tier3/medagents.py`](../../frameworks/tier3/medagents.py),
  [`frameworks/tier3/medagents_vi.py`](../../frameworks/tier3/medagents_vi.py)
- Related: `001-tier1-disable-thinking`, `001-tier1-fix-parse-answer`, `003-tier3-llama8b-pilot`
