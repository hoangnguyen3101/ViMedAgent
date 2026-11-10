"""
ViMedAgent — Main experiment runner

Usage:
    # Full matrix — one command runs everything
    python -m runner.run_experiments --dataset data/vm14k_clean.jsonl --all

    # Single framework + model (all seeds)
    python -m runner.run_experiments --dataset data/pilot_100.jsonl \\
        --framework zero_shot --model llama-3.1-8b

    # Force re-run (ignore existing results)
    python -m runner.run_experiments --dataset data/vm14k_clean.jsonl \\
        --all --no-resume

N_RUNS, SEEDS, EXPERIMENT_MATRIX — all set in infrastructure/config.py.
"""

import argparse
import json
import os
import random
import socket
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from infrastructure.config import (
    EXPERIMENT_MATRIX, ALL_FRAMEWORKS, ABLATION_FRAMEWORKS, get_results_dir,
    MODEL_NAMES, SEEDS, N_RUNS, NUM_WORKERS, FRAMEWORK_TIER, get_num_workers,
    MODELS, DEFAULT_TEMPERATURE, COT_SC_TEMPERATURE, FORCE_TEMPERATURE, REQUEST_TIMEOUT,
    get_base_url, get_model_base_url, model_url_env, get_api_key,
)
import infrastructure.config as cfg_mod
from infrastructure.llm_client import reset_usage, get_usage, ANSWER_PARSER_VERSION
from frameworks import (
    run_zero_shot, run_cot, run_cot_sc, run_self_refine,
    run_voting_ensemble,
    run_medagents, run_medagents_vi,
    run_mdagents,  run_mdagents_vi,
    run_mad_debate,
)
# Consultation ablations: imported directly so frameworks/__init__.py stays as in v1.
from frameworks.tier3.medagents_ablation import (
    run_medagents_hetero, run_medagents_noreport, HETERO_PANEL, NOREPORT_CONSULT_INPUT,
)
from metrics.compute import compute_pass_at_1, compute_mean_std
from data.loading import load_dataset


# ── Framework dispatch ────────────────────────────────────

def dispatch(framework: str, sample: dict, model: str, seed: int) -> dict:
    """
    Route to the correct framework runner.
    random.seed(seed) is set here so all Python-level randomness
    (e.g. voting_ensemble shuffles) is reproducible per seed.
    Each framework receives seed so it can set its own internal seed too.
    """
    # Set Python random seed for reproducibility.
    # voting_ensemble also calls random.seed(seed) internally — that is
    # intentional so the shuffle sequence is deterministic per seed.
    # Neither Ollama nor SGLang is given a seed parameter for the LLM call
    # itself; diversity across seeds comes from temperature=0.3 sampling.
    random.seed(seed)

    q     = sample["question"]
    opts  = sample["options"]
    diff  = sample.get("difficulty_level", "Medium")
    topic = sample.get("medical_topic")

    if framework == "zero_shot":
        return run_zero_shot(q, opts, model)
    elif framework == "cot":
        return run_cot(q, opts, model)
    elif framework == "cot_sc":
        return run_cot_sc(q, opts, model)
    elif framework == "self_refine":
        return run_self_refine(q, opts, model)
    elif framework == "voting_ensemble":
        # Pass seed so shuffle sequence is reproducible per seed
        return run_voting_ensemble(q, opts, model, seed=seed)
    elif framework == "medagents":
        return run_medagents(q, opts, model)
    elif framework == "medagents_vi":
        return run_medagents_vi(q, opts, model, medical_topic=topic)
    elif framework == "mdagents":
        return run_mdagents(q, opts, model)
    elif framework == "mdagents_vi":
        return run_mdagents_vi(q, opts, model, difficulty_level=diff)
    elif framework == "mad_debate":
        return run_mad_debate(q, opts, model)
    elif framework == "medagents_hetero":
        return run_medagents_hetero(q, opts, model)
    elif framework == "medagents_noreport":
        return run_medagents_noreport(q, opts, model)
    else:
        raise ValueError(f"Unknown framework: {framework!r}")


# ── File helpers ──────────────────────────────────────────

def result_dir(dataset_stem: str) -> str:
    """MedQA-US uses results_USMLE; VM14K uses its existing result folders."""
    return get_results_dir(dataset_stem)


def audit_trail_of(result: dict) -> list | None:
    """audit_trail của framework; nếu framework không tự ghi thì dựng từ `raw`.

    zero_shot / cot / cot_sc / voting_ensemble trả `raw` nhưng không có
    audit_trail, nên trước 26/09/2026 output đầy đủ của chúng bị bỏ đi — không
    parse lại được khi sửa parse_answer (xem exp 001). Dựng ở runner thay vì
    sửa frameworks/ để frameworks vẫn giống hệt v1; schema không thêm trường.

    voting_ensemble: `vote` là đáp án ĐÃ map về thứ tự gốc, còn `raw` là chữ
    cái theo thứ tự đã xáo — hai cái có thể khác chữ, đó là đúng.
    """
    if result.get("audit_trail") is not None:
        return result["audit_trail"]
    raw = result.get("raw")
    if raw is None:
        return None
    if isinstance(raw, list):
        votes = result.get("individual_votes") or [None] * len(raw)
        return [{"step": "sample", "index": i, "vote": v, "raw": r}
                for i, (r, v) in enumerate(zip(raw, votes))]
    return [{"step": "single", "answer": result.get("answer"), "raw": raw}]


def result_path(framework: str, model: str, dataset_stem: str, seed: int) -> str:
    safe_model = model.replace("/", "_").replace(".", "_")
    return os.path.join(
        result_dir(dataset_stem),
        FRAMEWORK_TIER[framework],
        framework,
        f"{framework}__{safe_model}__{dataset_stem}__s{seed}.jsonl",
    )


def load_done_ids(path: str) -> set:
    done = set()
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    done.add(json.loads(line)["id"])
                except Exception:
                    pass
    return done


def read_results(path: str) -> list:
    out = []
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    out.append(json.loads(line))
                except Exception:
                    pass
    return out


# ── Single-seed run ───────────────────────────────────────

def _write_manifest(out_path, framework, model, dataset_path, seed, n_samples) -> None:
    """Ghi <out>.meta.json: mọi thứ cần để tái lập run này mà tên file không nói được.

    Ghi đè mỗi lần chạy (kể cả resume) nên luôn phản ánh cấu hình lần chạy cuối;
    `runs` giữ lịch sử để biết một seed bị ngắt và chạy tiếp mấy lần.
    """
    meta_path = out_path + ".meta.json"
    prev = {}
    if os.path.exists(meta_path):
        try:
            with open(meta_path, encoding="utf-8") as f:
                prev = json.load(f)
        except Exception:
            prev = {}

    spec = MODELS.get(model, {})
    provider = spec.get("provider")
    meta = {
        "framework":     framework,
        "model":         model,
        "model_id":      spec.get("model_id"),
        "provider":      provider,
        "base_url":      get_model_base_url(model) if provider else None,
        "dataset":       os.path.abspath(dataset_path),
        "n_samples":     n_samples,
        "seed":          seed,
        "temperature":   DEFAULT_TEMPERATURE,
        "cot_sc_temperature": COT_SC_TEMPERATURE,
        "force_temperature": FORCE_TEMPERATURE,   # not None = every call used this temperature
        "request_timeout_s": REQUEST_TIMEOUT,
        "max_tokens":    {k: v for k, v in sorted(vars(cfg_mod).items())
                          if k.startswith("MAX_TOKENS_")},
        "code_version":  _git_describe(),
        "answer_parser_version": ANSWER_PARSER_VERSION,
        "runs": (prev.get("runs") or []) + [{
            "started": datetime.utcnow().isoformat(),
            "host":    socket.gethostname(),
            "python":  sys.version.split()[0],
        }],
    }
    if framework in ("medagents_hetero", "medagents_noreport"):
        # expert slot i -> model; gathering, report and decision use `model`
        panel = HETERO_PANEL if framework == "medagents_hetero" else (model,) * 3
        meta["panel"] = [{"slot": i + 1, "model": m, "model_id": MODELS[m]["model_id"],
                          "base_url": get_model_base_url(m)} for i, m in enumerate(panel)]
        meta["consult_input"] = "report" if framework == "medagents_hetero" else NOREPORT_CONSULT_INPUT
    if prev.get("answer_reprocessing"):
        meta["answer_reprocessing"] = prev["answer_reprocessing"]
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


# Paths whose state decides what a run computes; uncommitted changes here mark the run "-dirty".
_RUN_CODE_PATHS = ("frameworks", "infrastructure", "runner", "data/loading.py")


def _git_describe() -> str | None:
    """Short HEAD hash, plus "-dirty" when run-affecting code differs from HEAD.

    Without the suffix a run made with uncommitted framework/config changes would be
    recorded under the parent commit, i.e. under code it did not actually use.
    """
    try:
        import subprocess
        code_dir = str(Path(__file__).resolve().parent.parent)
        head = subprocess.check_output(
            ["git", "-C", code_dir, "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL, text=True, timeout=5).strip()
        status = subprocess.check_output(
            ["git", "-C", code_dir, "status", "--porcelain", "--", *_RUN_CODE_PATHS],
            stderr=subprocess.DEVNULL, text=True, timeout=10)
        dirty = any(line.strip() and "__pycache__" not in line for line in status.splitlines())
        return head + ("-dirty" if dirty else "")
    except Exception:
        return None


def check_endpoints(models: list[str]) -> None:
    """Fail fast unless every SGLang model has its own endpoint serving exactly that model.

    An SGLang server serves one model and largely ignores the request's model name, so a
    model whose URL is missing would silently be answered by whatever the default server
    runs. Used by the heterogeneous-panel ablation, where three models share one question.
    """
    import urllib.request
    problems, seen = [], {}
    for m in dict.fromkeys(models):
        spec = MODELS[m]
        if spec["provider"] != "sglang":
            continue
        env = model_url_env(m)
        url = os.environ.get(env)
        if not url:
            problems.append(f"{m}: set {env}=http://<host>:<port>/v1")
            continue
        if url in seen:
            problems.append(f"{m} and {seen[url]} share the endpoint {url}")
        seen[url] = m
        try:
            req = urllib.request.Request(url.rstrip("/") + "/models",
                                         headers={"Authorization": f"Bearer {get_api_key('sglang')}"})
            with urllib.request.urlopen(req, timeout=15) as resp:
                served = [d.get("id") for d in json.loads(resp.read()).get("data", [])]
        except Exception as exc:
            problems.append(f"{m}: cannot reach {url}/models ({exc})")
            continue
        if spec["model_id"] not in served:
            problems.append(f"{m}: {url} serves {served}, expected {spec['model_id']!r}")
    if problems:
        raise SystemExit("Endpoint check failed:\n  " + "\n  ".join(problems))
    print("Endpoint check OK: " + ", ".join(f"{m} -> {os.environ[model_url_env(m)]}"
                                            for m in dict.fromkeys(models) if MODELS[m]["provider"] == "sglang"))


def run_one_seed(
    framework: str,
    model: str,
    dataset_path: str,
    seed: int,
    resume: bool = True,
    workers: int | None = None,
) -> list:
    """
    Run one (framework, model, seed) combination over the full dataset.
    Appends to per-seed JSONL file. Supports resume (skips done IDs).
    Returns list of all result records for this seed.
    """
    dataset_stem = Path(dataset_path).stem
    out_path = result_path(framework, model, dataset_stem, seed)
    # Validate and adapt the entire input before writing files or calling a model.
    samples = load_dataset(dataset_path)
    if framework == "mdagents_vi":
        missing = sum(not s.get("difficulty_level") for s in samples)
        if missing:
            print(f"  WARNING: {missing} samples lack difficulty_level; "
                  "mdagents_vi uses the Medium fallback for them.")
    elif framework == "medagents_vi":
        missing = sum(not s.get("medical_topic") for s in samples)
        if missing:
            print(f"  WARNING: {missing} samples lack medical_topic; "
                  "medagents_vi uses generic specialist roles for them.")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    _write_manifest(out_path, framework, model, dataset_path, seed, len(samples))

    done_ids  = load_done_ids(out_path) if resume else set()
    remaining = [s for s in samples if s["id"] not in done_ids]

    # v2: concurrency depends on the backend — Ollama caps at OLLAMA_NUM_PARALLEL
    # (4), SGLang batches server-side so 32+ in flight is the sweet spot.
    n_workers = workers or get_num_workers(model)

    tag = f"{framework} x {model} | seed={seed} | workers={n_workers}"
    print(f"\n  [{tag}]")
    print(f"  Total: {len(samples)} | Done: {len(done_ids)} | Remaining: {len(remaining)}")

    if not remaining:
        print("  Already complete -- skipping.")
        return read_results(out_path)

    errors      = 0
    api_calls   = 0
    correct_run = sum(1 for r in read_results(out_path) if r.get("correct"))
    t0          = time.time()
    i           = 0
    write_lock  = threading.Lock()  # hoangnv: file writes now happen off worker threads

    def call_one(sample: dict) -> dict:
        # Bộ đếm token theo thread: xoá ở đầu câu, đọc ở cuối câu. Hợp lệ vì
        # mỗi câu chạy trọn vẹn trong đúng một worker thread và không framework
        # nào tự sinh thread con. Xem llm_client.reset_usage.
        reset_usage()
        result = dispatch(framework, sample, model, seed)
        result["_usage"] = get_usage()
        return result

    # hoangnv: dispatch calls concurrently instead of one-request-at-a-time.
    # Ollama's service already accepts OLLAMA_NUM_PARALLEL concurrent requests, so a
    # sequential loop left the GPU mostly idle. NUM_WORKERS matches that setting.
    with open(out_path, "a", encoding="utf-8") as fout, \
            ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(call_one, sample): sample for sample in remaining}
        try:
            for future in as_completed(futures):
                sample = futures[future]
                try:
                    result     = future.result()
                    is_correct = (result.get("answer") == sample.get("answer"))

                    record = {
                        "id":             sample["id"],
                        "seed":           seed,
                        "framework":      framework,
                        "model":          model,
                        "gold":           sample.get("answer"),
                        "answer":         result.get("answer"),
                        "correct":        is_correct,
                        "difficulty":     sample.get("difficulty_level"),
                        "topic":          sample.get("medical_topic"),
                        "api_calls":      result.get("api_calls", 1),
                        # individual_votes: used for Pass@k (cot_sc, voting_ensemble)
                        # For MAS frameworks this holds agent votes — NOT used for Pass@k
                        "individual_votes": (
                            result.get("individual_votes")
                            or result.get("round2_answers")
                            or result.get("final_votes")
                        ),
                        "round1_answers": result.get("round1_answers"),
                        # Token thật do server báo (SGLang: usage; Ollama native:
                        # prompt_eval_count/eval_count). api_calls đếm SỐ LỜI GỌI,
                        # token đo KHỐI LƯỢNG — hai trục chi phí khác nhau, phải có
                        # cả hai thì so sánh CoT-SC với MAS mới công bằng.
                        **(result.get("_usage") or {}),
                        "audit_trail":    audit_trail_of(result),
                        "timestamp":      datetime.utcnow().isoformat(),
                    }

                    with write_lock:
                        fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                        fout.flush()

                        api_calls   += record["api_calls"]
                        correct_run += is_correct
                        i           += 1
                        done_count   = len(done_ids) + i

                        if i % 10 == 0:
                            elapsed = time.time() - t0
                            acc     = correct_run / done_count * 100
                            speed   = i / elapsed if elapsed > 0 else 0
                            eta     = (len(remaining) - i) / speed if speed > 0 else 0
                            print(
                                f"  [{done_count}/{len(samples)}] "
                                f"acc={acc:.1f}% | calls={api_calls} | "
                                f"elapsed={elapsed:.0f}s | ETA={eta:.0f}s"
                            )

                except Exception as e:
                    with write_lock:
                        errors += 1
                        print(f"  ERROR [{sample.get('id','?')}]: {e}")
                        if errors > 10:
                            print("  10+ errors -- stopping this seed.")
                            for f in futures:
                                f.cancel()
                            break

        except KeyboardInterrupt:
            print(
                f"\n  Interrupted at {i}/{len(remaining)}. "
                f"Re-run with same args to resume."
            )
            for f in futures:
                f.cancel()

    return read_results(out_path)


# ── Multi-seed aggregation ────────────────────────────────

def run_all_seeds(
    framework: str,
    model: str,
    dataset_path: str,
    resume: bool = True,
    workers: int | None = None,
    seeds: list[int] | None = None,
) -> dict:
    """
    Run the given seeds (default: all N_RUNS seeds in config.SEEDS) for one
    (framework, model) pair. Returns aggregated mean +/- std metrics.
    """
    per_seed_metrics = []
    for seed in (seeds or SEEDS):
        results = run_one_seed(framework, model, dataset_path, seed, resume, workers)
        if results:
            per_seed_metrics.append(compute_pass_at_1(results))

    if not per_seed_metrics:
        return {}

    agg = compute_mean_std(per_seed_metrics)
    _print_agg(framework, model, agg)
    return agg


def _print_agg(framework: str, model: str, agg: dict):
    ov = agg.get("overall", {})
    print(f"\n  {'='*55}")
    print(f"  {framework} | {model} | {agg.get('n_seeds','?')} seeds")
    print(f"  {'='*55}")
    print(
        f"  Overall : {ov.get('mean',0):.2f}% +/- {ov.get('std',0):.2f}%"
        f"  (n={agg.get('n_total','?')})"
    )
    for diff, v in sorted(agg.get("by_difficulty", {}).items()):
        print(f"  {diff:12s}: {v.get('mean',0):.2f}% +/- {v.get('std',0):.2f}%")
    print(f"  {'='*55}")


# ── CLI ───────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="ViMedAgent experiment runner")
    parser.add_argument(
        "--dataset", default="data/pilot_100.jsonl",
        help="Path to VM14K or MedQA JSONL (4 or 5 options)",
    )
    parser.add_argument(
        "--framework", default="zero_shot",
        choices=ALL_FRAMEWORKS + ABLATION_FRAMEWORKS,
        help="Single framework to run (ignored when --all is set)",
    )
    parser.add_argument(
        "--model", default="sglang-qwen3-32b",
        choices=MODEL_NAMES,
        help="Single model to run (ignored when --all is set)",
    )
    parser.add_argument(
        "--all", action="store_true",
        help=(
            f"Run full EXPERIMENT_MATRIX "
            f"({len(EXPERIMENT_MATRIX)} pairs x {N_RUNS} seeds = "
            f"{len(EXPERIMENT_MATRIX)*N_RUNS} total runs)"
        ),
    )
    parser.add_argument(
        "--no-resume", action="store_true",
        help="Ignore existing results and start fresh",
    )
    parser.add_argument(
        "--workers", type=int, default=None,
        help="Concurrent LLM calls (default: per-provider value from config, "
             "or env VIMED_NUM_WORKERS)",
    )
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=None,
        help=f"Seeds to run for a single framework/model (default: config.SEEDS = {SEEDS})",
    )
    parser.add_argument(
        "--validate-only", action="store_true",
        help="Validate and summarize the dataset without LLM calls or result files",
    )
    args   = parser.parse_args()
    resume = not args.no_resume

    if args.validate_only:
        try:
            samples = load_dataset(args.dataset)
        except (OSError, ValueError) as exc:
            parser.error(str(exc))
        print(f"Dataset: {args.dataset}")
        print(f"Results directory: {get_results_dir(args.dataset)}")
        print(f"Samples: {len(samples)} | Unique IDs: {len({s['id'] for s in samples})}")
        print(f"Option counts: {dict(sorted(Counter(len(s['options']) for s in samples).items()))}")
        print(f"Gold answers: {dict(sorted(Counter(s['answer'] for s in samples).items()))}")
        print(f"Empty question strings: {sum(not s['question'].strip() for s in samples)}")
        print(f"Missing difficulty_level: {sum(not s.get('difficulty_level') for s in samples)}")
        print(f"Missing medical_topic: {sum(not s.get('medical_topic') for s in samples)}")
        print("Validation passed. No model calls or result files created.")
        return

    if args.all:
        total = len(EXPERIMENT_MATRIX) * N_RUNS
        print(f"\nViMedAgent -- Full matrix")
        print(f"  {len(EXPERIMENT_MATRIX)} framework x model pairs "
              f"x {N_RUNS} seeds = {total} runs")
        print(f"  Dataset : {args.dataset}")
        print(f"  Seeds   : {SEEDS}")

        all_agg = {}
        for fw, model in EXPERIMENT_MATRIX:
            print(f"\n{'='*60}")
            agg = run_all_seeds(fw, model, args.dataset, resume, args.workers)
            all_agg[(fw, model)] = agg

        # Final summary
        print(f"\n\n{'='*75}")
        print(f"  FINAL SUMMARY -- mean +/- std over {N_RUNS} seeds")
        print(f"  {'Framework':<25} {'Model':<22} {'Overall':>16}")
        print(f"{'='*75}")
        for (fw, model), agg in all_agg.items():
            ov = agg.get("overall", {})
            print(
                f"  {fw:<25} {model:<22} "
                f"{ov.get('mean',0):>6.2f}% +/- {ov.get('std',0):.2f}%"
            )
        print(f"{'='*75}")

    else:
        if args.framework == "medagents_hetero":
            check_endpoints([args.model, *HETERO_PANEL])
        run_all_seeds(args.framework, args.model, args.dataset, resume, args.workers, seeds=args.seeds)


if __name__ == "__main__":
    main()
