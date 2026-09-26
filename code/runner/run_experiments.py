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
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from infrastructure.config import (
    RESULTS_DIR, RESULTS_TEST_DIR, EXPERIMENT_MATRIX, ALL_FRAMEWORKS,
    MODEL_NAMES, SEEDS, N_RUNS, NUM_WORKERS, FRAMEWORK_TIER, get_num_workers,
    MODELS, DEFAULT_TEMPERATURE, REQUEST_TIMEOUT, get_base_url,
)
import infrastructure.config as cfg_mod
from infrastructure.llm_client import reset_usage, get_usage
from frameworks import (
    run_zero_shot, run_cot, run_cot_sc, run_self_refine,
    run_voting_ensemble,
    run_medagents, run_medagents_vi,
    run_mdagents,  run_mdagents_vi,
    run_mad_debate,
)
from metrics.compute import compute_pass_at_1, compute_mean_std


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
    else:
        raise ValueError(f"Unknown framework: {framework!r}")


# ── File helpers ──────────────────────────────────────────

def result_dir(dataset_stem: str) -> str:
    """Pilot/test datasets (stem starting with 'pilot') go to RESULTS_TEST_DIR
    instead of RESULTS_DIR, so quick trial runs never mix into the real results."""
    return RESULTS_TEST_DIR if dataset_stem.startswith("pilot") else RESULTS_DIR


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
        "base_url":      get_base_url(provider) if provider else None,
        "dataset":       os.path.abspath(dataset_path),
        "n_samples":     n_samples,
        "seed":          seed,
        "temperature":   DEFAULT_TEMPERATURE,
        "request_timeout_s": REQUEST_TIMEOUT,
        "max_tokens":    {k: v for k, v in sorted(vars(cfg_mod).items())
                          if k.startswith("MAX_TOKENS_")},
        "code_version":  _git_describe(),
        "runs": (prev.get("runs") or []) + [{
            "started": datetime.utcnow().isoformat(),
            "host":    socket.gethostname(),
            "python":  sys.version.split()[0],
        }],
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def _git_describe() -> str | None:
    """Không phải repo git nên thường trả None; giữ chỗ nếu sau này init git."""
    try:
        import subprocess
        return subprocess.check_output(
            ["git", "-C", str(Path(__file__).resolve().parent.parent), "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL, text=True, timeout=5).strip()
    except Exception:
        return None


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
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # Load dataset
    samples = []
    with open(dataset_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))

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
                        "audit_trail":    result.get("audit_trail"),
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
) -> dict:
    """
    Run all N_RUNS seeds for one (framework, model) pair.
    Returns aggregated mean +/- std metrics.
    """
    per_seed_metrics = []
    for seed in SEEDS:
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
        help="Path to dataset JSONL file",
    )
    parser.add_argument(
        "--framework", default="zero_shot",
        choices=ALL_FRAMEWORKS,
        help="Single framework to run (ignored when --all is set)",
    )
    parser.add_argument(
        "--model", default="llama-3.1-8b",
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
    args   = parser.parse_args()
    resume = not args.no_resume

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
        run_all_seeds(args.framework, args.model, args.dataset, resume, args.workers)


if __name__ == "__main__":
    main()
