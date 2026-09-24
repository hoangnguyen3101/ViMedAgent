# Experiment <ID> — <short title>

<!-- Copy this folder with: scripts/new_experiment.sh <slug>
     Fill every section. Keep it honest: record what actually happened, not the plan. -->

- **Daspai* dd/mm/yyyy
- **Author:**
- **Status:** planned | running | done | abandoned
- **Base model:** e.g. Qwen2-VL-7B-Instruct
- **Method:** LoRA | QLoRA | full SFT | DPO | ...

## 1. Motivation / Hypothesis

Why run this experiment? Which limitation of a previous run (link it: exp `NNN`) prompted it,
and what do you expect to happen?

## 2. Setup

- **Data:** path(s), format, #samples (train / val / test)
- **Config:** [`./config.yaml`](./config.yaml) — key hyperparams: lr, epochs, batch, lora rank …
- **Hardware:** GPUs, expected wall-clock
- **Command:** `training/run_train.sh experiments/<ID>-<slug>`
- **Δ vs previous:** what changed compared to exp `NNN` (ideally exactly one variable)

## 3. What I did (log)

Chronological notes: steps taken, surprises, mid-run fixes, anything you'd forget in a week.

## 4. Results

- **Metrics** (vs baseline):

  | Metric | This exp | Baseline (exp NNN) | Δ |
  |--------|---------:|-------------------:|---:|
  | CER ↓ |          |                    |   |
  | Field-exact ↑ |   |                    |   |

- **Loss curve:** `./results/loss_curve.png`
- **Checkpoint:** `<external path>` (gitignored)
- **Qualitative:** a few good / bad sample outputs

## 5. Error analysis

Main failure modes (from `errors/` scripts): confusions, per-field breakdown, systematic mistakes.

## 6. Conclusion

Did the hypothesis hold? Keep or discard this direction? Why?

## 7. Next step

What experiment does this suggest? (link the follow-up once it exists)

## 8. Artifacts & links

- Config: [`./config.yaml`](./config.yaml)
- Logs: [`./logs/`](./logs/)
- Results: [`./results/`](./results/)
- Checkpoint: `<path>`
- Related: exp `NNN` · ticket · wandb run



Trên là định dạng cho mỗi lần thử nghiệm (mỗi thử nghiệm sẽ là 1 file). em suy nghĩ việc điền lại các thử nghiệm trước đó mà em đã từng thử nghiệm đối với cả spai và phần liên quan đến chất lượng ảnh nhé