# Experiment 001-tier1-disable-thinking — Tắt Qwen3 hidden thinking cho zero_shot & CoT

<!-- theo chuẩn experiment/Template/README.md -->

- **Date:** 05/07/2026
- **Author:** hoangnv (qua Claude Code)
- **Status:** done (phần `zero_shot`) — xem follow-up ở `001-tier1-fix-parse-answer` cho `cot`
- **Base model:** `ollama-qwen3-32b` (tag `qwen3-32b-ctx8k`, Qwen3 32.8B, Q4_K_M, num_ctx=8192)
- **Method:** không phải training — thay đổi inference config (tắt cờ `think` của Ollama)

## 1. Motivation / Hypothesis

Chạy `zero_shot x ollama-qwen3-32b` (seed 42, full `vm14k_clean.jsonl`, 11094 câu) cho
`answer: null` ở **41.8%** số câu (4635/11094). Giả thuyết: Qwen3 là model
hybrid-reasoning, tự động sinh khối suy luận ẩn `<think>...</think>` trước khi trả lời —
kể cả khi prompt yêu cầu "chỉ trả lời 1 ký tự" — và khối này ăn hết ngân sách
`MAX_TOKENS_LONG=512` trước khi model kịp nhả ra đáp án, khiến response bị cắt cụt.
Nếu đúng, tắt thinking sẽ vừa hết null vừa nhanh hơn nhiều.

## 2. Setup

- **Data:** `data/vm14k_clean.jsonl` — 11094 câu hỏi trắc nghiệm y khoa tiếng Việt (A/B/C/D),
  full set (không chia train/val/test, đây là benchmark eval).
- **Config:** không có `config.yaml` training — tham số liên quan nằm trong
  `infrastructure/config.py`: `MAX_TOKENS_LONG=512`, `DEFAULT_TEMPERATURE=0.3`,
  `SEEDS=[42,43,44]`, `NUM_WORKERS=4` (khớp `OLLAMA_NUM_PARALLEL=4`).
- **Hardware:** 1x Quadro RTX 8000 (49GB), Ollama server local (`OLLAMA_FLASH_ATTENTION=1`,
  `OLLAMA_KEEP_ALIVE=60m`).
- **Command:**
  ```
  python -m runner.run_experiments --dataset data/vm14k_clean.jsonl --framework zero_shot --model ollama-qwen3-32b
  ```
- **Δ vs previous:** thêm đúng 1 tham số `think=False` vào lời gọi `call_llm()` trong
  `zero_shot.py` (và tương tự cho `cot.py`) — mọi thứ khác (prompt, `max_tokens`,
  `temperature`, seed) giữ nguyên.

## 3. What I did (log)

1. Phát hiện null rate 41.8% trên seed 42 (`results/_pre_nothink_backup/zero_shot__ollama-qwen3-32b__vm14k_clean__s42.jsonl`).
2. Test trực tiếp qua `curl` tới `/v1/chat/completions`: 1 câu hỏi đơn giản sinh ra
   315 completion token, trong đó model chỉ trả lời "A" — còn lại là `reasoning` field
   (khối thinking ẩn). Mất ~34.8s cho 1 request.
3. Test câu hỏi phức tạp hơn với `max_tokens=512` (đúng giá trị code đang dùng):
   `finish_reason: "length"`, bị cắt đúng lúc chạm giới hạn.
4. Test cực đoan: câu hỏi "Urobilinogen" cho `max_tokens=4096` — vẫn không hội tụ
   (`finish_reason: "length"`, 16675 ký tự reasoning, `content` rỗng). → không chỉ là
   vấn đề tăng token là đủ, thinking có thể loop không hội tụ với 1 số câu.
5. Thử `"think": false` qua `/v1/chat/completions` — **không có tác dụng** (Ollama âm thầm
   bỏ qua tham số này ở endpoint OpenAI-compat).
6. Thử qua endpoint gốc `/api/chat` với `"think": false` — thành công: 2 completion token
   (chỉ "A"), ~8.5s, `finish_reason: "stop"`.
7. Sửa `infrastructure/llm_client.py`: thêm `_call_ollama_native()` gọi `/api/chat`,
   thêm tham số `think: bool | None = None` vào `call_llm()` (mặc định `None` = giữ
   nguyên hành vi cũ, không ảnh hưởng framework/model khác).
8. Sửa `zero_shot.py` và `cot.py`: thêm `think=False`.
9. Dừng tiến trình đang chạy (`Ctrl-C` chỉ hủy seed hiện tại rồi tự nhảy seed kế —
   phải `kill -TERM` mới dừng hẳn), backup 3 file kết quả cũ (s42/s43/s44) vào
   `results/_pre_nothink_backup/`, chạy lại từ đầu.

## 4. Results

- **Metrics** (vs baseline — cùng model, cùng dataset, cùng seed, chỉ khác `think`):

  | Metric | think=False (mới) | think=True/mặc định (cũ, có bug) | Δ |
  |--------|---------:|---------:|---:|
  | Overall accuracy | **67.96% ± 0.01%** | ~48%* | +~20pp |
  | null rate | 0% | 41.8% | -41.8pp |
  | Thời gian / câu (đơn lẻ) | ~8.5s | ~30-35s | ~4x nhanh hơn |
  | completion tokens / câu | ~1-5 | 297-315+ (nhiều câu vượt cả 512) | |

  *con số 48% bị nhiễu nặng bởi 41.8% null (luôn tính sai), không phản ánh khả năng thật.

  Breakdown đầy đủ (think=False, 3 seed, n=11094):
  ```
  Overall : 67.96% +/- 0.01%
  Easy    : 69.51% +/- 0.23%
  Hard    : 65.41% +/- 0.25%
  Medium  : 67.59% +/- 0.12%
  ```

- **Qualitative:** so 1 câu hỏi cụ thể (Urobilinogen) — think=True không bao giờ ra
  được đáp án dù cho 4096 token; think=False ra "B" trong 9.5s (đáp án model chọn sai
  so với gold "A", nhưng ít nhất *có* đáp án thay vì `null`).

## 5. Error analysis

- Nguồn lỗi không nằm ở khả năng model, mà ở việc token budget (512) không đủ chứa
  cả khối thinking ẩn lẫn đáp án — với model hybrid-reasoning, thinking có thể tốn
  vài trăm tới vài nghìn token tùy độ khó câu hỏi, không có cách nào set `max_tokens`
  đủ lớn để an toàn 100% (đã thấy loop không hội tụ ở 4096 token).
- Lỗi chỉ xảy ra với model có thinking (Qwen3, DeepSeek-R1) — Llama-3.1-8B không có
  cơ chế này nên không bao giờ gặp bug tương tự dù chạy cùng code, cùng `max_tokens`.

## 6. Conclusion

Giả thuyết đúng: tắt thinking giải quyết cả 2 vấn đề (null-answer bug + tốc độ) trong
cùng 1 thay đổi. Ngoài ra, về mặt thiết kế nghiên cứu, tắt thinking cho tier `zero_shot`/`cot`
còn là lựa chọn *đúng đắn hơn về phương pháp luận* (không chỉ vì tốc độ): 2 framework này
là baseline "không suy luận"/"suy luận theo prompt tường minh" trong thang 3-tier
(`zero_shot → cot → multi-agent`) — để thinking ẩn bật sẽ phá vỡ construct validity của
baseline, và làm so sánh chéo Qwen3 vs Llama-3.1-8B (model không có thinking) không còn
công bằng.

Giữ hướng này (think=False) cho mọi framework Tier 1 dùng Qwen3/DeepSeek-R1 qua Ollama.

## 7. Next step

- `cot x ollama-qwen3-32b`: code đã sửa (`think=False`), nhưng trong lúc chạy lại phát
  hiện thêm 1 bug độc lập ở bước parse đáp án (accuracy 63.4% < zero_shot dù đã tắt
  thinking) → xem `001-tier1-fix-parse-answer/`. Cần backup + chạy lại `cot` sau khi
  áp dụng fix đó.
- Rà soát các model Ollama khác có thinking chưa từng chạy qua code mới
  (`ollama-qwen3-14b`, `ollama-deepseek-r1-14b`) — về lý thuyết đã được `think=False`
  tự động qua cùng `zero_shot.py`/`cot.py`, nhưng chưa có run thực tế kiểm chứng.
- [x] ~~Cân nhắc áp dụng `think=False` cho `cot_sc`, `self_refine` (Tier 1 còn lại)~~ —
  **đã áp dụng** (19/07/2026): `cot_sc.py` (1 chỗ), `self_refine.py` (3 chỗ). Chưa có
  run thực tế kiểm chứng trên Qwen3-32B (mới chỉ sửa code).
- [x] ~~Tier 2/3 nếu dùng Qwen3/DeepSeek-R1 qua Ollama~~ — Tier 3 đã xong ở
  `003-tier3-disable-thinking/`; Tier 2 (`voting_ensemble`) xong ở
  [`002-tier2-disable-thinking/`](../002-tier2-disable-thinking/README.md) (19/07/2026).

## 8. Artifacts & links

- Code thay đổi: [`infrastructure/llm_client.py`](../../infrastructure/llm_client.py),
  [`frameworks/tier1/zero_shot.py`](../../frameworks/tier1/zero_shot.py),
  [`frameworks/tier1/cot.py`](../../frameworks/tier1/cot.py),
  [`frameworks/tier1/cot_sc.py`](../../frameworks/tier1/cot_sc.py) (thêm 19/07/2026),
  [`frameworks/tier1/self_refine.py`](../../frameworks/tier1/self_refine.py) (thêm 19/07/2026)
- Kết quả mới: [`results/zero_shot__ollama-qwen3-32b__vm14k_clean__s42.jsonl`](../../results/zero_shot__ollama-qwen3-32b__vm14k_clean__s42.jsonl)
  (+ s43, s44)
- Backup kết quả cũ (có bug): [`results/_pre_nothink_backup/`](../../results/_pre_nothink_backup/)
- Related: `001-tier1-fix-parse-answer`
