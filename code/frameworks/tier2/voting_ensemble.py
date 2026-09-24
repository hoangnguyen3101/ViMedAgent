"""
ViMedAgent — Tier 2: Voting Ensemble
VM14K Table 3 "Ensemble" metric.

3 independent shuffled runs → majority vote.
Diversity from option-order permutation (eliminates position bias).
Tie-breaking: first run's answer (deterministic, not alphabetical).
Temperature=0.3 (consistent with all other frameworks).
API calls: n (default 3)
"""

import random

from infrastructure.llm_client import call_llm, parse_answer
from frameworks.utils import majority_vote, format_options


def build_prompt(question: str, shuffled_options: list[str]) -> list[dict]:
    return [{
        "role": "user",
        "content": (
            f"{question}\n"
            f"Choose the correct option from these answers:\n"
            f"{format_options(shuffled_options)}\n"
            f"Only response with 1 character. Example: A"
        ),
    }]


def run_voting_ensemble(
    question: str,
    options: list[str],
    model: str,
    n: int = 3,
    seed: int | None = None,
) -> dict:
    """
    VM14K Ensemble: n shuffled runs → majority vote.
    Ties resolved by first run's answer (set as fallback).

    seed: controls shuffle randomness for reproducibility across runs.
          Caller (run_experiments) passes the per-run seed so that
          seed 42 / 43 / 44 produce different shuffles.

    individual_votes: original-space answers for Pass@3 computation.

    API calls: n (default 3)
    """
    if seed is not None:
        random.seed(seed)

    votes: list[str | None] = []
    raws:  list[str]        = []
    first_answer: str | None = None

    for run_idx in range(n):
        indices       = list(range(len(options)))
        random.shuffle(indices)
        shuffled_opts = [options[i] for i in indices]

        # reverse_map: shuffled letter → original letter
        reverse_map = {
            chr(65 + new_pos): chr(65 + orig_idx)
            for new_pos, orig_idx in enumerate(indices)
        }

        msgs         = build_prompt(question, shuffled_opts)
        raw          = call_llm(model, msgs, think=False)
        ans          = parse_answer(raw)
        original_ans = reverse_map.get(ans) if ans else None

        if run_idx == 0:
            first_answer = original_ans

        votes.append(original_ans)
        raws.append(raw)

    final = majority_vote(votes, fallback=first_answer)

    return {
        "answer":           final,
        "individual_votes": votes,
        "raw":              raws,
        "api_calls":        n,
    }
