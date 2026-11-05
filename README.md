# ViMedAgent: MedQA-USMLE and VM14K

This branch, `dev/feat-usmle`, extends ViMedAgent v2 to run medical multiple-choice experiments on **MedQA-USMLE (English)** alongside **VM14K (Vietnamese)**. It adds MedQA input normalization, separate USMLE result folders, four- and five-option answer handling, and stricter answer extraction.

The active USMLE dataset is the **four-option development split: 1,272 questions**. Results from this split are development results, not held-out test results.

## Quick start

The Python runner works on Windows or Linux and calls an existing model server over HTTP. Python 3.10 or newer is required; the client does not require PyTorch or a local GPU.

```bash
git clone --branch dev/feat-usmle https://github.com/hoangnguyen3101/ViMedAgent.git
cd ViMedAgent/code
python -m pip install -r requirements-client.txt
```

Check the dataset without contacting a model server or creating result files:

```bash
python -m runner.run_experiments --dataset data/dev.jsonl --validate-only
```

Expected: 1,272 unique questions, four options per question, and `results_USMLE` as the result directory.

### Run with Ollama

Start Ollama and create the model tag expected by the configuration:

```bash
ollama pull llama3.1:8b
ollama create llama3.1-8b-ctx8k -f infrastructure/ollama_modelfiles/Modelfile.llama3.1-8b
```

Run zero-shot on the development split:

```bash
python -m runner.run_experiments --dataset data/dev.jsonl --framework zero_shot --model ollama-llama3.1-8b --workers 1
```

This command runs all configured seeds: **42, 43, and 44**. The Ollama provider uses `http://localhost:11434`; model keys and served model names are defined in [config.py](code/infrastructure/config.py).

### Run only the 10-question smoke test

The bundled `pilot_usmle_10.jsonl` contains the first ten questions from the same development split, with their normalized IDs preserved. Run one seed:

```bash
python -c "from runner.run_experiments import run_one_seed; run_one_seed('zero_shot', 'ollama-llama3.1-8b', 'data/pilot_usmle_10.jsonl', seed=42, workers=1)"
```

Use the normal runner command with `--dataset data/pilot_usmle_10.jsonl` to run all three seeds instead.

## Datasets

| File in `code/data/` | Questions | Choices | Purpose |
|---|---:|---:|---|
| `dev.jsonl` | 1,272 | 4 | MedQA-US development split |
| `pilot_usmle_10.jsonl` | 10 | 4 | First ten MedQA-US development questions |
| `vm14k_clean.jsonl` | 11,094 | 4 | Existing Vietnamese benchmark |
| `pilot_100.jsonl` | 100 | 4 | Existing VM14K pilot |

`dev.jsonl` is copied from `questions/US/4_options/phrases_no_exclude_dev.jsonl` in the [original MedQA release](https://github.com/jind11/MedQA). This branch uses the US subset. The Mainland and Taiwan subsets are outside its experiment scope.

The loader accepts MedQA dictionaries with `answer_idx` and VM14K lists with letter-valued `answer`. It normalizes records in memory, preserves source files, and generates stable IDs when they are missing. Four- and five-option inputs are supported throughout prompts, parsing, voting, and metrics; the bundled USMLE files use four options.

MedQA-US does not provide VM14K's difficulty or medical-topic labels. Its result fields remain `null`, and metric summaries use `Unknown`. See the [framework notes](code/README.md#frameworks-and-missing-metadata) before running the `_vi` variants.

## Output

Run commands from `code/`. Result routing is shared by the runner, metric readers, and backend comparison script:

| Dataset | Result directory |
|---|---|
| MedQA-US / USMLE | `results_USMLE/` |
| `vm14k_clean` | `results/` |
| `pilot_100` and other non-USMLE pilots | `results_test/` |

For example:

```text
code/results_USMLE/tier1/zero_shot/
  zero_shot__ollama-llama3_1-8b__dev__s42.jsonl
  zero_shot__ollama-llama3_1-8b__dev__s42.jsonl.meta.json
  zero_shot__ollama-llama3_1-8b__dev__s43.jsonl
  zero_shot__ollama-llama3_1-8b__dev__s43.jsonl.meta.json
  zero_shot__ollama-llama3_1-8b__dev__s44.jsonl
  zero_shot__ollama-llama3_1-8b__dev__s44.jsonl.meta.json
```

Each JSONL line stores the prediction, gold label, correctness, model and seed, token usage, agent votes where applicable, and the raw response in `audit_trail`. The manifest stores the run configuration and answer-parser version. The terminal reports mean accuracy and sample standard deviation across completed seed runs.

The runner writes JSONL and manifests. CSV and Markdown reports from the local smoke test were exported separately; they are not automatic runner outputs. Generated results and logs are excluded from Git.

## Answer extraction

Zero-shot prompts ask for a single letter, but a model may still return explanations. The parser therefore:

1. Prefers an explicit `Final answer` declaration.
2. Accepts consistent answer declarations or standalone answer lines.
3. Ignores incidental option mentions in explanations.
4. Returns `null` for conflicting declarations without a final answer, or for invalid/out-of-range labels.

Unparsed or ambiguous answers count as incorrect in full-dataset accuracy. Raw responses are retained for auditing. The current parser version is `explicit-answer-v1`; older results must be identified and reparsed from their original responses before comparing parser versions. Re-parsing multi-agent logs cannot reproduce how different intermediate answers would have changed subsequent model calls.

## Frameworks and backends

Ten frameworks are available: `zero_shot`, `cot`, `cot_sc`, `self_refine`, `voting_ensemble`, `medagents`, `medagents_vi`, `mdagents`, `mdagents_vi`, and `mad_debate`.

Supported providers are Ollama, SGLang, and a LiteLLM proxy. For SGLang, use the repository's installation and launch scripts on the server, then point the client at its OpenAI-compatible endpoint:

```bash
# Linux server: Qwen3-32B-AWQ recipe; check available GPU memory first.
bash scripts/x86_cuda/install_sglang.sh
MODEL=Qwen/Qwen3-32B-AWQ MEM_FRACTION=0.65 MAX_RUNNING=32 bash scripts/x86_cuda/serve_sglang.sh
```

In a separate Linux client terminal:

```bash
export SGLANG_BASE_URL=http://127.0.0.1:30000/v1
python scripts/sglang_smoke_test.py --model sglang-qwen3-32b
python -m runner.run_experiments --dataset data/dev.jsonl --framework zero_shot --model sglang-qwen3-32b
```

On PowerShell, set the endpoint with `$env:SGLANG_BASE_URL = 'http://127.0.0.1:30000/v1'`. For GX10/GB10, use `scripts/install_gx10.sh` and `scripts/serve_gx10.sh` instead of the x86 installer. The [experiment workflow](.claude/skills/run-experiment/SKILL.md) records measured model-memory budgets and server checks.

## Validation

```bash
cd code
python -m unittest discover -s tests -v
python -m runner.run_experiments --dataset data/dev.jsonl --validate-only
python -m runner.run_experiments --dataset data/vm14k_clean.jsonl --validate-only
```

The 24 offline tests cover input normalization, both option counts, all frameworks and adaptive routes, shuffled voting, USMLE output routing, resume behavior, manifests, and answer-extraction regressions. They do not require a running model server.

For data schemas, metrics, configuration, and run details, see the [English runner guide](code/README.md).
