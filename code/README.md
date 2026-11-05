# Runner and evaluation guide

This guide describes the `dev/feat-usmle` branch. For installation and a quick start, see the [main README](../README.md). Run commands below from `code/`.

## Layout

```text
code/
  data/                  VM14K and MedQA-US inputs; normalization in loading.py
  frameworks/tier1/      Zero-shot, CoT, CoT-SC, Self-Refine
  frameworks/tier2/      Shuffled Voting Ensemble
  frameworks/tier3/      MedAgents, MDAgents, their _vi variants, MAD debate
  infrastructure/        Model configuration, HTTP clients, label helpers, parser
  runner/                Experiment dispatch, per-seed execution, manifests
  metrics/               Accuracy, collaboration metrics, validity checks
  scripts/               Model-server setup, smoke tests, backend comparisons
  tests/                 Offline regression tests
  results_USMLE/         Generated MedQA-US results, excluded from Git
  results/               Generated VM14K results, excluded from Git
  results_test/          Generated non-USMLE pilot results, excluded from Git
```

## Input schemas

### MedQA-US

The active `data/dev.jsonl` contains 1,272 four-option development questions from `questions/US/4_options/phrases_no_exclude_dev.jsonl` in the [MedQA release](https://github.com/jind11/MedQA). One source line has this shape:

```json
{
  "question": "A medical multiple-choice question",
  "options": {"A": "First option", "B": "Second option", "C": "Third option", "D": "Fourth option"},
  "answer": "Second option",
  "answer_idx": "B",
  "meta_info": "step1"
}
```

The loader sorts dictionary keys A-D (or A-E for five choices), converts options to a list, and takes the gold label from `answer_idx`. It checks that any answer text agrees with that label. Missing IDs become `medqa_us_<file_stem>_<source_line>`, such as `medqa_us_dev_000001`.

Existing IDs are preserved. `data/pilot_usmle_10.jsonl` holds the first ten normalized development questions with their original development IDs. Repeated seeds and resumed runs therefore use the same question IDs.

The source JSONL is never modified by loading. Duplicate IDs, missing/invalid gold labels, malformed option keys, and option counts other than four or five are rejected before inference. `meta_info` identifies exam sections; it is not an Easy/Medium/Hard difficulty label.

### VM14K

VM14K already uses the normalized schema:

```json
{
  "id": "question-id",
  "question": "A Vietnamese medical question",
  "options": ["First option", "Second option", "Third option", "Fourth option"],
  "answer": "B",
  "difficulty_level": "Medium",
  "medical_topic": ["Cardiology"]
}
```

The 11,094 existing records and their IDs are preserved. The current VM14K file contains two empty question strings; validation reports these without filtering them or changing the benchmark population.

## Run experiments

Validate inputs without model calls or output files:

```bash
python -m runner.run_experiments --dataset data/dev.jsonl --validate-only
```

Run a framework/model pair across all configured seeds:

```bash
python -m runner.run_experiments --dataset data/dev.jsonl --framework zero_shot --model ollama-llama3.1-8b --workers 1
```

Run only the ten-question pilot at seed 42:

```bash
python -c "from runner.run_experiments import run_one_seed; run_one_seed('zero_shot', 'ollama-llama3.1-8b', 'data/pilot_usmle_10.jsonl', seed=42, workers=1)"
```

The model server must be running and serving the exact model ID configured for the selected key. For `ollama-llama3.1-8b`, that ID is `llama3.1-8b-ctx8k`. Server installation and the Ollama tag-creation command are in the [main README](../README.md).

| CLI option | Behavior |
|---|---|
| `--dataset PATH` | JSONL input; defaults to `data/pilot_100.jsonl` |
| `--framework NAME` | Framework to execute; defaults to `zero_shot` |
| `--model KEY` | Key from `MODELS`; defaults to `sglang-qwen3-32b` |
| `--workers N` | Concurrent requests; defaults depend on the provider |
| `--validate-only` | Validate and summarize without inference |
| `--all` | Run the full configured framework/model matrix |
| `--no-resume` | Ignore completed IDs; currently still appends to the existing file |

Re-running the same normal command resumes by skipping completed question IDs. To get a fresh experiment, use a separate dataset/run name or archive the existing outputs first; `--no-resume` does not truncate an existing result file.

The configured seeds are 42, 43, and 44. They govern Python-side operations such as shuffling. They are not passed as sampling seeds to the model server, so generation is not guaranteed to be deterministic.

## Frameworks and missing metadata

| Tier | Framework | LLM calls per question |
|---|---|---:|
| 1 | `zero_shot` | 1 |
| 1 | `cot` | 1 |
| 1 | `cot_sc` | 5 |
| 1 | `self_refine` | 2-5 |
| 2 | `voting_ensemble` | 3 |
| 3 | `medagents` | 9 with three selected experts |
| 3 | `medagents_vi` | 8 with three experts |
| 3 | `mdagents` | 2 / 4 / 7 by routed complexity |
| 3 | `mdagents_vi` | 1 / 3 / 6 by difficulty label |
| 3 | `mad_debate` | Up to 9 |

The framework implementations follow the repository's existing protocols; they are re-implementations rather than exact reproductions of the original papers' code. Four-choice prompt text is preserved when expanding the choice count to five.

MedQA-US does not include `difficulty_level` or `medical_topic`:

- `mdagents` uses a model call to select Low/Moderate/High complexity.
- `medagents` uses a model call to recruit experts.
- `mdagents_vi` falls back to Medium when no difficulty label is supplied.
- `medagents_vi` falls back to its generic expert roles when no topic is supplied.

The runner prints warnings for the `_vi` fallbacks. These fallbacks are not inferred dataset labels. Result `difficulty` and `topic` remain `null`, and metric summaries use `Unknown`. Use the original `mdagents` and `medagents` variants for the main MedQA comparison unless an additional labeling protocol is defined.

## Answer extraction

`infrastructure.llm_client.parse_answer()` receives each question's option count. It strips thinking blocks and uses the `explicit-answer-v1` policy:

1. An explicit final-answer declaration takes priority; the last final declaration can revise an earlier provisional answer.
2. Otherwise, explicit answer declarations and standalone first/last answer lines must agree.
3. A concluding selection such as `Therefore, option A ...` is checked for conflicts with other answer declarations.
4. An unambiguous bold or parenthesized label is supported as a fallback.
5. Unresolved choices, invalid/out-of-range labels, or conflicting declarations return `None` (`null` in JSON).

The parser does not pick a letter just because it appears last in an option-by-option explanation. It never consults the gold answer to resolve a conflict. Zero-shot requests a single character, but its prompt is not a decoding constraint; raw output may include explanations.

For example, `A` on the first line followed by discussion of option D still selects A. `The correct answer is B ... Therefore, option A ...` is ambiguous unless a final answer resolves it. A null answer counts as incorrect in full-dataset accuracy.

Changing the parser changes scoring. Identify parser versions in manifests, retain raw responses, and record any offline reprocessing. For multi-agent methods, offline parsing changes cannot recreate the subsequent model messages that different intermediate choices would have produced.

## Result routing and files

`infrastructure.config.get_results_dir()` is shared by the runner, metric readers, and `scripts/compare_results.py`.

| Dataset stem | Root |
|---|---|
| `dev`, `train`, `test` | `results_USMLE/` |
| `phrases_no_exclude_dev`, `phrases_no_exclude_train`, `phrases_no_exclude_test` | `results_USMLE/` |
| Prefixes `usmle`, `medqa_us`, `pilot_usmle`, `pilot_medqa_us` | `results_USMLE/` |
| Other stems starting with `pilot` | `results_test/` |
| Other datasets, including `vm14k_clean` | `results/` |

Paths are relative to the process working directory; run from `code/`. The generic stems `dev`, `train`, and `test` are reserved for this branch's MedQA-US inputs.

```text
<root>/tier<N>/<framework>/<framework>__<safe_model>__<dataset_stem>__s<seed>.jsonl
<root>/tier<N>/<framework>/<framework>__<safe_model>__<dataset_stem>__s<seed>.jsonl.meta.json
```

Slashes and periods in model keys are replaced with underscores. JSONL records are appended as worker requests complete, so their order need not match input order; join by `id` and `seed`.

| Result fields | Meaning |
|---|---|
| `id`, `seed`, `framework`, `model` | Question and run identity |
| `gold`, `answer`, `correct` | Gold letter, parsed prediction, and correctness |
| `difficulty`, `topic` | Dataset labels, or null when absent |
| `api_calls`, `n_llm_calls` | Framework and client call counters |
| `tokens_in`, `tokens_out`, `n_usage_missing` | Token usage and missing-usage count |
| `individual_votes`, `round1_answers` | Sampling/agent votes where applicable |
| `audit_trail` | Step details and preserved raw model output |
| `timestamp` | Original result timestamp |

The manifest stores dataset path and size, model/provider/endpoint, seed, temperature, token budgets, code revision, parser version, and restart history. Recorded offline reparsing history is preserved when the manifest is updated.

After the configured seeds, the terminal prints mean accuracy and sample standard deviation. The runner does not automatically create CSV, Excel, or Markdown reports. The smoke-test CSV and summaries in a local working copy are separate exports.

## Metrics and checks

Check that each seed has the expected question count, unique IDs, `api_calls == n_llm_calls`, and `n_usage_missing == 0`. If usage is missing, the recorded token total is only a lower bound. Do not count repeated seeds for a question as independent questions in significance tests.

Accuracy, sampling Pass@k, collaboration metrics, and answer distributions accept A-E. Agent votes within one multi-agent run are not independent sampling attempts for Pass@k. For datasets outside the built-in VM14K sizes, metric readers use `n_samples` from the runner manifest to determine complete seeds.

Compare two model keys on the same USMLE dataset:

```bash
python scripts/compare_results.py --framework zero_shot --dataset dev --a ollama-llama3.1-8b --b sglang-llama3.1-8b
```

The legacy analysis CLI takes an explicit results directory:

```bash
python -m analysis.analyze_results --dataset dev --results-dir results_USMLE
```

## Configuration and validation

Model keys, token budgets, seed lists, and defaults live in `infrastructure/config.py`.

| Environment variable | Purpose |
|---|---|
| `SGLANG_BASE_URL` | SGLang endpoint; default `http://127.0.0.1:30000/v1` |
| `SGLANG_API_KEY` | API key if the SGLang server requires one |
| `LITELLM_BASE_URL` / `OPENAI_BASE_URL` | LiteLLM proxy endpoint |
| `LITELLM_API_KEY` / `OPENAI_API_KEY` | Proxy API key |
| `VIMED_REQUEST_TIMEOUT` | Per-request timeout in seconds; default 300 |
| `VIMED_NUM_WORKERS` | Override default concurrency |

Keep credentials in the environment. Generated results, logs, and environment files are excluded by `.gitignore`.

```bash
python -m unittest discover -s tests -v
python -m runner.run_experiments --dataset data/dev.jsonl --validate-only
python -m runner.run_experiments --dataset data/vm14k_clean.jsonl --validate-only
```

The 24 offline tests exercise parser precedence and ambiguity, both option counts, MedQA normalization, all framework routes, shuffled votes, output routing, result/manifest readers, and resume behavior. They mock model calls and require no inference server.
