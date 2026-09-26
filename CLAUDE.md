# ViMedAgentv2 — luật bắt buộc

## 1. CẤM mọi thao tác git ghi. Chỉ được KIỂM TRA.

Không chạy, kể cả khi người dùng có vẻ đồng ý trong một câu nói thoáng qua:

```
git add · commit · push · init · clone · reset · clean · checkout · switch · restore
git rm · mv · merge · rebase · revert · stash · cherry-pick · apply · am
git fetch · pull · submodule · worktree · gc
git branch -d/-D/-m · git tag -d/-f · git remote add/rm/set-url · git config <ghi>
```

Được chạy tự do: `git status · diff · log · show · blame · ls-files · branch · tag · remote -v · config --get/--list`

Hook `.claude/hooks/block-git-write.sh` (`PreToolUse` trên `Bash`) chặn ở tầng harness, kể cả khi
lệnh nấp sau `&&`, `||`, `;`, `|`, `$(...)` hoặc có `-C <path>` xen giữa. `permissions.deny` trong
`.claude/settings.json` là lớp thứ hai.

**Why:** thư mục này chứa kết quả **không tái tạo lại được rẻ** — `code/results/` là 417 MB / 179 file
JSONL; riêng `mdagents_vi × sglang-qwen3-32b` tốn ~99.000 giây GPU. Một lần `git add . && push` nhầm là
đẩy toàn bộ lên remote; một lần `reset --hard` hay `clean -fd` nhầm là mất sạch, không có bản sao.

**How to apply:** muốn biết trạng thái thì dùng `ls`, `du -sh`, `find`, `diff -rq` với `../ViMedAgent`.
Người dùng chủ động yêu cầu push → **nói rõ luật này chặn và để họ tự chạy tay**, đừng tự gỡ hook.

## 2. Bí mật

`infrastructure/config.py` từng hardcode 6 khoá Groq (đã comment) — **xoá ngày 24/09/2026**, thay bằng
`GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")`. Sáu khoá đó phải coi là **đã lộ**.
`ViMedAgent` (v1) vẫn còn **đúng 6 khoá đó**, chưa dọn.

Không bao giờ ghi khoá vào file trong repo. Không ghi khoá ra file tạm, ra log, hay in ra terminal —
kể cả để "cho người dùng đối chiếu". `.gitignore` đã chặn sẵn `.env`, `*.key`, `*.pem`.

## 3. Quan hệ với ViMedAgent (v1)

v2 = bản sao của v1, **chỉ khác tầng hạ tầng**: `infrastructure/config.py`, `infrastructure/llm_client.py`,
`runner/run_experiments.py`, `scripts/`, cộng `.gitignore` + `.claude/` + `CLAUDE.md`.

`frameworks/**` phải **giống hệt v1**. Mọi khác biệt ở đó làm bảng v1 và v2 không so được với nhau.
Kiểm bất cứ lúc nào:

```bash
diff -rq ../ViMedAgent/code/frameworks code/frameworks
```

**Đã xảy ra một lần:** nhánh Moderate của MDAgents từng được viết lại cho trung thành code gốc
(`_run_mdt`, tranh luận nhiều vòng, 12–82 lời gọi/câu) rồi **hoàn tác 20/09/2026** theo quyết định của
hoangnv — các framework là **mô phỏng lại bài toán**, không cam kết replicate 100%. Đừng "sửa cho đúng
paper" lần nữa nếu không được yêu cầu rõ ràng.

## 4. Schema kết quả — đóng băng, sửa TRƯỚC khi chạy

```
id · seed · framework · model · gold · answer · correct · difficulty · topic
api_calls · tokens_in · tokens_out · n_llm_calls · n_usage_missing
individual_votes · round1_answers · audit_trail · timestamp
+ <file>.jsonl.meta.json  (manifest cấu hình lần chạy)
```

Thêm/bớt trường nào cũng làm dữ liệu cũ và mới không so được — **sửa trước khi chạy, không giữa chừng**.
Nơi tiêu thụ schema: `../Table_and_fig/code/table/build_matrix.py`.

Hai bất biến đáng tin:
- `api_calls == n_llm_calls` (framework tự đếm vs client đếm, độc lập nhau) — lệch nghĩa là có lời gọi lọt sổ.
- `n_usage_missing > 0` → `tokens_in/out` chỉ là **cận dưới**, không được báo cáo chi phí.

## 5. Ba điểm lệch phải khai trong Methods

| Điểm | Chi tiết |
|---|---|
| **Context 70B** | `ollama-llama3.3-70b` = ctx**4k**, còn 8B/32B = ctx8k. Lý do phần cứng ở `config.py`. |
| **MedGemma-4B** | bf16 không lượng tử hoá (các model khác 4-bit) · chuyên biệt y khoa · tinh chỉnh tiếng Anh nhưng VM14K là tiếng Việt. Xem README 3.3.1. |
| **Framework là mô phỏng** | MDAgents Moderate = 3 agent bỏ phiếu độc lập (không phải MDT tranh luận); High = 5 chuyên gia song song + 1 synthesizer (không phải ICT nhiều đội nối tiếp). |

## 6. Vận hành

Quy trình dựng server + chạy thí nghiệm + kiểm kết quả: gọi skill **`run-experiment`**
(`.claude/skills/run-experiment/`). Đừng tự chế lệnh serve — ngân sách bộ nhớ từng model đã đo sẵn ở đó.
