# Information Loss — cần chuẩn hoá lại sau khi có reasoning trace

Ghi chú bàn giao, không phải code. Mục đích: khi quay lại `metrics/failure.py`,
biết chính xác phải sửa gì và tại sao chưa sửa ngay bây giờ.

## Vì sao viết file này

`Hoang_Bảng I_Test` (Metric.xlsx) đã BỎ cột Information Loss khỏi Bảng III vì
với log cũ, nó bằng hệt Minority Suppression ở 18/18 ô — thiếu reasoning trace
của agent để tách hai khái niệm. `failure.py` hiện tại (dòng 51-53) code đúng
y hệt điều đó:

```python
if gold in agents:
    suppressed += 1
    info_loss += 1
```

`agents` ở đây là `pre_synthesis_answers(r)` — danh sách CHỮ CÁI đáp án cuối
của từng agent trước bước tổng hợp (`io.py::pre_synthesis_answers`). Vì cả hai
biến cộng dồn trong cùng một điều kiện, `info_loss_pct == minority_suppression_pct`
luôn luôn, đúng như spreadsheet đã ghi nhận. Đây không phải bug mới — code
hiện tại chỉ đang trung thực phản ánh giới hạn dữ liệu, và đã tự gắn nhãn
`INFO_LOSS_NOTE` cận dưới cho việc này.

## Đã thay đổi ở tier 3 (2026-08-26)

5 framework tier 3 giờ ghi thêm field `"reasoning"` vào các bước audit_trail
cấp-agent (trước đây chỉ có `"answer"`, reasoning bị `extract_reasoning()`
tính ra rồi vứt đi trước khi ghi JSON):

| Framework       | File                              | Bước có `reasoning` mới |
|-----------------|------------------------------------|--------------------------|
| medagents       | `frameworks/tier3/medagents.py`    | `analysis`, `consultation` |
| medagents_vi    | `frameworks/tier3/medagents_vi.py` | `analysis`, `consultation` |
| mdagents        | `frameworks/tier3/mdagents.py`     | `pcc`, `mdt`, `ict_expert` |
| mdagents_vi     | `frameworks/tier3/mdagents_vi.py`  | `pcc`, `mdt`, `ict_expert` |
| mad_debate      | `frameworks/tier3/mad_debate.py`   | mọi entry có `agent` (round 1 và round 2+) |

Field dùng chung tên `"reasoning"` ở mọi nơi để `io.py` xử lý đồng nhất.

**Chỉ áp dụng cho log chạy MỚI.** Mọi file `.jsonl` đang có trong
`results/tier3/` và `results_test/tier3/` (kể cả `pilot_100` dùng cho Bảng III)
đều được sinh TRƯỚC thay đổi này → không có `reasoning`. Phải chạy lại tier3
(ít nhất là `pilot_100`, 3 seed, 3 model) trước khi bất kỳ số info_loss nào
tính từ log mới có ý nghĩa.

## Việc cần làm trong `metrics/` (chưa làm, để sau)

1. **`io.py`**
   - Thêm hàm kiểu `pre_synthesis_reasoning(r) -> list[str]` song song với
     `pre_synthesis_answers`, đọc cùng field `_AGENT_STEPS` nhưng lấy
     `s.get("reasoning")` thay vì `s.get("answer")`. Giữ thứ tự khớp với
     `pre_synthesis_answers` (cùng index = cùng agent) để so sánh answer↔reasoning
     của từng agent.
   - `_AGENT_STEPS` hiện KHÔNG có `mad_debate` (nó đi qua nhánh `elif` riêng
     trong `pre_synthesis_answers`). Nếu thêm hàm reasoning song song, nhánh
     `mad_debate` cũng phải lấy đúng field `reasoning` ở entry có `round == 1`.
   - Cân nhắc thêm cờ kiểu `has_reasoning_trace(records) -> bool` (mẫu giống
     `validity.py::has_confidence`) để `failure_modes()` tự phát hiện log cũ
     (không có `reasoning`) và không âm thầm tính ra info_loss sai — nên trả
     `None`/cảnh báo thay vì một con số trông hợp lệ.

2. **`failure.py`** — định nghĩa lại `info_loss` cho đúng tinh thần paper
   ("đáp án đúng được NÊU trong reasoning trace"), tách khỏi Minority
   Suppression:
   - Minority Suppression (giữ nguyên): agent chọn đúng gold làm **đáp án
     cuối** của mình nhưng bị đa số đè.
   - Information Loss (định nghĩa lại): gold **xuất hiện trong text reasoning**
     của ít nhất một agent — kể cả agent đó KHÔNG chọn gold làm đáp án cuối
     (VD: agent cân nhắc rồi loại bỏ, hoặc nêu đúng luận điểm nhưng chọn nhầm
     đáp án) — và cuối cùng câu vẫn sai.
   - Cần một quy tắc match text rõ ràng trước khi code (không được đoán mò
     lúc implement): match theo chữ cái đáp án (`"B"`, `"(B)"`, `"answer is B"`
     …) dễ false-positive vì agent hay nhắc tới các lựa chọn để LOẠI chúng
     ("not B, so..."). Có thể cần heuristic phủ định đơn giản (loại các câu
     có "not", "rule out", "unlikely" ngay trước ký tự đáp án) tương tự cách
     `mdagents.py::parse_complexity` đã xử lý phủ định cho "not high".
   - Giữ nguyên mẫu số `n_wrong_multi_agent` — lý do trong docstring hiện tại
     vẫn đúng (MDAgents PCC 1-agent không định nghĩa được info loss).
   - Xoá `INFO_LOSS_NOTE` cận dưới sau khi định nghĩa mới không còn phụ thuộc
     riêng vào đáp án chữ cái nữa (hoặc đổi nội dung note nếu vẫn còn giới hạn
     khác, vd: reasoning bị `extract_reasoning()` cắt ở 500 ký tự / câu cuối
     cùng — có thể cắt mất đúng đoạn nêu gold nếu nó nằm giữa bài).

3. **Kiểm tra ngược** — sau khi có log mới, chạy lại và xác nhận
   `info_loss_pct != minority_suppression_pct` ở ít nhất một số ô (nếu vẫn
   trùng 100% thì định nghĩa match text chưa tách được hai khái niệm, quay
   lại bước 2).

## Việc KHÔNG nằm trong phạm vi ghi chú này

- Không đánh giá style/API của 6 file `accuracy.py`, `collaboration.py`,
  `failure.py`, `io.py`, `stats.py`, `validity.py` — theo yêu cầu, chưa sửa
  gì ở đây, file này chỉ ghi lại việc cần làm cho riêng Information Loss.
