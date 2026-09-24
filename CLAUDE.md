# ViMedAgentv2 — quy tắc bắt buộc

## 1. CẤM mọi thao tác git ghi. Chỉ được KIỂM TRA.

**Không được chạy**, dù người dùng có vẻ đồng ý trong một câu nói thoáng qua:

```
git add      git commit    git push      git init     git clone
git reset    git clean     git checkout  git switch   git restore
git rm       git mv        git merge     git rebase   git revert
git stash    git cherry-pick             git apply    git am
git fetch    git pull      git submodule git worktree git gc
git branch -d/-D/-m        git tag -d/-f
git remote add/rm/set-url  git config <ghi>
```

**Được chạy tự do** (đọc, không đổi gì):

```
git status   git diff   git log    git show      git blame
git ls-files git branch git tag    git remote -v git config --get/--list
```

Quy tắc này do hook `.claude/hooks/block-git-write.sh` chặn ở tầng harness
(`PreToolUse` trên `Bash`), cộng với `permissions.deny` trong `.claude/settings.json`.
Hook chặn cả khi lệnh nấp sau `&&`, `||`, `;`, `|`, `$(...)` hoặc có `-C <path>` xen giữa.

**Why:** thư mục này chứa kết quả thí nghiệm **không tái tạo lại được rẻ** —
`results/**/*.jsonl` khoảng 10 MB mỗi seed, mỗi bộ ba seed tốn hàng chục giờ GPU
(riêng `mdagents_vi × sglang-qwen3-32b` mất ~99.000 giây). Một lần
`git add . && git commit && git push` nhầm là đẩy toàn bộ dữ liệu lên remote;
một lần `git reset --hard` hay `git clean -fd` nhầm là mất sạch, không có bản sao.
Thư mục hiện **không phải** git repo, và phải giữ nguyên như vậy.

**How to apply:** khi cần biết trạng thái thư mục, dùng `ls`, `du -sh`, `find`,
`diff -rq` với ViMedAgent (v1) — không dùng git. Nếu người dùng chủ động yêu cầu
commit/push, **không tự chạy**: nói rõ rằng quy tắc này chặn, và để họ tự chạy tay.

## 2. Quan hệ với ViMedAgent (v1)

`ViMedAgentv2` = bản sao của `ViMedAgent` **chỉ khác tầng hạ tầng**:
`infrastructure/config.py`, `infrastructure/llm_client.py`, `runner/run_experiments.py`,
`scripts/`. Logic framework (`frameworks/**`) phải **giống hệt v1** — nếu buộc phải
lệch thì ghi rõ lý do vào comment ngay tại chỗ, vì mọi khác biệt ở đó làm bảng v1 và v2
không so sánh được với nhau.

## 3. Trước khi sửa schema kết quả

Mỗi bản ghi JSONL có: `id, seed, framework, model, gold, answer, correct, difficulty,
topic, api_calls, individual_votes, round1_answers, tokens_in, tokens_out, n_llm_calls,
n_usage_missing, audit_trail, timestamp`, kèm `<file>.meta.json` ghi cấu hình lần chạy.

Thêm/bớt trường nào cũng làm dữ liệu cũ và mới không so được. Sửa **trước** khi chạy,
không sửa giữa chừng. `Table_and_fig/code/table/build_matrix.py` là nơi tiêu thụ schema này.
