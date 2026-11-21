"""
MedAgents with Steps 1-2 replayed from a stored run, Steps 3-5 re-run (advisor's item 4).

Why: Steps 1-2 of a stored MedAgents row (expert gathering, the three analyses) cost 4 of 9 calls and
do not depend on the parser. The analysis LABELS do: Step 3 writes "[role] (Answer: X)" into the report
prompt, so an analysis whose label was missed at run time (None) reached the report as "Answer: None".
Here the stored analysis TEXT is kept and its label is extracted again with the current parse_answer
(vote_extraction, paper Section 3.6); Steps 3-5 are then run live, so the report, the three revisions and
the decision are generated from correct labels.

  Step 1  expert gathering   replayed (stored raw output and roles)
  Step 2  3 analyses         replayed (stored raw text); label re-extracted with parse_answer
  Step 3  report             LIVE   (1 call)
  Step 4  3 revisions        LIVE   (3 calls)
  Step 5  decision           LIVE   (1 call)

Prompts and flow are those of frameworks/tier3/medagents.py: the prompt builders and the role parser
are imported, not copied (tests/test_medagents_reextract.py checks the live calls are word for word
those of run_medagents).

api_calls = 5 = the number of live calls, so api_calls == n_llm_calls and tokens_in/out cover Steps 3-5
only; the 4 replayed calls are NOT included in this row's cost. The first four audit_trail entries are
copied from the source row with "reused": true; analysis entries keep the label stored at run time as
"answer_at_run_time" next to the re-extracted "answer".
"""

from infrastructure.llm_client import call_llm, parse_answer
from infrastructure.config import MAX_TOKENS_SHORT, MAX_TOKENS_MEDIUM
from frameworks.utils import extract_reasoning
from frameworks.tier3.medagents import (
    parse_roles, prompt_report, prompt_consult, prompt_decision,
)

N_EXPERTS = 3
REUSED_STEPS = ("expert_gathering", "analysis")
LIVE_CALLS = 5          # report + 3 revisions + decision
REUSED_CALLS = 4        # gathering + 3 analyses

_EXPECTED = ("expert_gathering", "analysis", "analysis", "analysis",
             "report", "consultation", "consultation", "consultation", "decision")


def check_source_trail(trail: list[dict]) -> None:
    """The source row must be a complete 9-call MedAgents row; refuse anything else."""
    steps = tuple(t.get("step") for t in trail or [])
    if steps != _EXPECTED:
        raise ValueError(f"source audit_trail is not a MedAgents row: steps={steps}")
    if any(not isinstance(t.get("raw"), str) for t in trail[:4]):
        raise ValueError("source row has a missing raw output in steps 1-2")


def run_medagents_reextract(question: str, options: list[str], model: str,
                            source_trail: list[dict]) -> dict:
    check_source_trail(source_trail)
    n = len(options)
    audit_trail = []

    # ── Step 1: replayed ──────────────────────────────────
    raw_roles = source_trail[0]["raw"]
    roles = parse_roles(raw_roles, fallback_n=N_EXPERTS)
    stored_roles = [t["role"] for t in source_trail[1:4]]
    if roles != stored_roles:
        raise ValueError(f"roles parsed from the stored gathering output {roles} "
                         f"differ from the stored analysis roles {stored_roles}")
    audit_trail.append({**source_trail[0], "reused": True})

    # ── Step 2: replayed text, label re-extracted ─────────
    analyses = []
    for role, step in zip(roles, source_trail[1:4]):
        raw = step["raw"]
        ans = parse_answer(raw, n_options=n)
        analyses.append({"role": role, "analysis": extract_reasoning(raw), "answer": ans, "raw": raw})
        audit_trail.append({"step": "analysis", "role": role, "answer": ans,
                            "answer_at_run_time": step.get("answer"),
                            "reasoning": analyses[-1]["analysis"], "raw": raw, "reused": True})

    api_calls = 0

    # ── Step 3: Report Summarization (live) ───────────────
    msgs = prompt_report(question, options, analyses)
    report = call_llm(model, msgs, max_tokens=MAX_TOKENS_MEDIUM, think=False)
    api_calls += 1
    audit_trail.append({"step": "report", "raw": report})

    # ── Step 4: Collaborative Consultation (live) ─────────
    final_opinions = []
    for a in analyses:
        msgs = prompt_consult(question, options, a["role"], report, a["analysis"])
        raw = call_llm(model, msgs, temperature=0.3, max_tokens=MAX_TOKENS_MEDIUM, think=False)
        ans = parse_answer(raw, n_options=n)
        api_calls += 1
        final_opinions.append({"role": a["role"], "answer": ans,
                               "reasoning": extract_reasoning(raw), "raw": raw})
        audit_trail.append({"step": "consultation", "role": a["role"], "answer": ans,
                            "reasoning": final_opinions[-1]["reasoning"], "raw": raw})

    # ── Step 5: Decision Making (live) ────────────────────
    msgs = prompt_decision(question, options, final_opinions)
    raw_dec = call_llm(model, msgs, max_tokens=MAX_TOKENS_SHORT, think=False)
    ans_dec = parse_answer(raw_dec, n_options=n)
    api_calls += 1
    audit_trail.append({"step": "decision", "answer": ans_dec, "raw": raw_dec})

    return {
        "answer":           ans_dec,
        "roles":            roles,
        "round1_answers":   [a["answer"] for a in analyses],
        "round2_answers":   [op["answer"] for op in final_opinions],
        "individual_votes": [op["answer"] for op in final_opinions],
        "audit_trail":      audit_trail,
        "api_calls":        api_calls,
    }
