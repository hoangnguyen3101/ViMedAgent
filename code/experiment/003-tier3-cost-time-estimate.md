# Ước tính thời gian & chi phí — Llama-3.3-70B cho Tier 2 + Tier 3

- **Ngày:** 17/07/2026
- **Phạm vi:** toàn bộ `data/vm14k_clean.jsonl` (11094 câu) × 3 seed (42/43/44) = 33,282 lượt/framework
- **Model:** `llama-3.3-70b` (Q4_K_M khi chạy local; bản cloud provider phục vụ độ chính xác cao hơn)

## Bảng tổng hợp

| Framework | Tier | Call/câu | Input token | Output token | Local (Quadro RTX 8000) | 2×A100 (tối ưu) | Groq | DeepInfra |
|---|---|---|---:|---:|---:|---:|---:|---:|
| `medagents` | 3 | 9 | 335.6M | 117.4M | 110.4 ngày | 101.9 giờ | $291 | $124 |
| `medagents_vi` | 3 | 8 | 321.5M | 100.4M | 96.1 ngày | 88.7 giờ | $269 | $114 |
| `mdagents` | 3 | 4 | 149.1M | 52.2M | 49.1 ngày | 45.3 giờ | $129 | $55 |
| `mdagents_vi` | 3 | 4 | 149.1M | 52.2M | 49.1 ngày | 45.3 giờ | $129 | $55 |
| `mad_debate` | 3 | 6 | 223.7M | 78.3M | 73.6 ngày | 67.9 giờ | $194 | $83 |
| `voting_ensemble` | 2 | 3 | 7.0M | 0.3M | 0.6 ngày | 0.5 giờ | $4 | $2 |
| **TỔNG** | — | — | **1,186.0M** | **400.7M** | **~378.8 ngày** | **~349.7 giờ (~14.6 ngày)** | **~$1,016** | **~$433** |

## Phương pháp luận & giả định

**Số token/call:**
- `medagents`: đo **trực tiếp** bằng tokenizer thật (qua `llama3.1-8b-ctx8k`, cùng họ tokenizer với Llama 3.3) trên câu hỏi dài nhất dataset + filler đúng bằng giới hạn `MAX_TOKENS_*` từng bước (worst-case thực tế).
- `medagents_vi`: suy ra từ `medagents` trừ bước "gather experts" (không có trong `medagents_vi`).
- `mdagents`, `mdagents_vi`: ước tính theo token/call trung bình của `medagents` (~1120 in / ~392 out mỗi call), giả định trung bình **4 call/câu** (giữa 3 nhánh độ khó Low=2 / Moderate=4 / High=5).
- `mad_debate`: cùng tỷ lệ token/call, giả định trung bình **6 call/câu** (giữa 3 vòng tối thiểu và 9 vòng tối đa khi không đồng thuận sớm).
- `voting_ensemble`: đo trực tiếp — prompt kiểu `zero_shot` (~70 token input, ~3 token output/call), 3 call/câu.

**Thời gian local (Quadro RTX 8000, 1 GPU hiện có):**
- Tốc độ sinh (decode): **14.5 token/giây**, suy từ băng thông bộ nhớ GPU (~624GB/s) ÷ 42GB weight Llama-3.3-70B Q4_K_M.
- Tốc độ xử lý prompt (prefill): **233 token/giây**, đo thực tế (`prompt_eval_duration` từ Ollama).
- Công thức: `input_tokens/233 + output_tokens/14.5`.

**Thời gian 2×A100 (kịch bản, chưa có phần cứng thật):**
- Hệ số tăng tốc ~26× so với Quadro (băng thông A100 80GB ~2039GB/s ≈ 3.27× × tăng `NUM_PARALLEL` tận dụng VRAM dư (~4×) × 2 GPU chạy song song).
- Đây là kịch bản **lạc quan**, cần cả đổi phần cứng lẫn tinh chỉnh lại config — xem thêm kịch bản bảo thủ hơn (~6.5×, ~58 ngày) đã bàn trong hội thoại.

**Chi phí cloud:**
- Groq (`llama-3.3-70b-versatile`): $0.59 / $0.79 per 1M token (input/output).
- DeepInfra: $0.23 / $0.40 per 1M token — rẻ nhất trong các provider đã khảo sát (Together AI, Fireworks đắt hơn ~1.7×).
- Dùng `MAX_TOKENS_*` (giới hạn cứng) làm ước tính output → đây là **cận trên**, chi phí thật có thể thấp hơn vì không phải bước nào cũng dùng hết giới hạn (vd bước "decision" giới hạn 16 token nhưng thường chỉ cần 1-2 token thật).

## Khuyến nghị

- **Rẻ + nhanh nhất trong các phương án đã xét:** DeepInfra (~$433, xong trong vài giờ tới 1-2 ngày tùy rate limit) — không cần đầu tư phần cứng.
- **Local (Quadro hiện có):** không khả thi cho toàn bộ phạm vi (~379 ngày).
- **Nếu cân nhắc mua/thuê 2×A100:** chỉ hợp lý nếu dùng lâu dài cho nhiều thí nghiệm khác, không riêng lần chạy này — chi phí thuê ngắn hạn nhiều khả năng vượt xa $433 của DeepInfra.
- **Muốn giảm thêm chi phí:** cắt bớt seed (3→1) hoặc lấy mẫu con thay vì full 11094 câu — giảm tuyến tính theo tỷ lệ.
