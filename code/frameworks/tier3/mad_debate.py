"""
ViMedAgent — MAD: Multi-Agent Debate (Du et al. ICML 2024)
https://arxiv.org/abs/2305.14325

"Society of minds" approach:
  Round 1: N agents independently propose answers + reasoning
  Round 2+: Each agent reads all other agents' responses → rebuts/updates
  Early stop: if all agents converge to same answer
  Final: majority vote over last-round answers (per paper)

No specialist roles — debate is role-free (agents are anonymous).
N=3 agents, max_rounds=3 (paper used 2, but performance improves with more).

API calls: N * rounds (max 3*3=9, often less due to early stop)
"""

from infrastructure.llm_client import call_llm, parse_answer
from infrastructure.config import MAX_TOKENS_DEBATE
from frameworks.utils import majority_vote, format_options, extract_reasoning


# ── Prompts ───────────────────────────────────────────────

def prompt_initial(question: str, options: list[str],
                   agent_id: int) -> list[dict]:
    """Round 1: agent proposes initial answer + reasoning."""
    return [
        {
            "role": "system",
            "content": (
                f"You are Medical Agent #{agent_id + 1}. "
                f"Your goal is to reason carefully and argue for the correct answer."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Medical question:\n{question}\n\n"
                f"Options:\n{format_options(options)}\n\n"
                f"State your reasoning and answer (A/B/C/D)."
            ),
        },
    ]


def prompt_debate(question: str, options: list[str],
                  agent_id: int, round_num: int,
                  other_responses: list[str]) -> list[dict]:
    """Round 2+: agent reads others' arguments and updates position."""
    others_str = "\n\n".join(
        f"Agent #{i + 1}: {resp}"
        for i, resp in enumerate(other_responses)
    )
    return [
        {
            "role": "system",
            "content": (
                f"You are Medical Agent #{agent_id + 1}, Round {round_num}. "
                f"Review other agents' arguments and refine your answer."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Medical question:\n{question}\n\n"
                f"Options:\n{format_options(options)}\n\n"
                f"Other agents' arguments:\n{others_str}\n\n"
                f"Do you maintain your position or update it? "
                f"Argue concisely. End with your answer (A/B/C/D)."
            ),
        },
    ]


# ── Runner ────────────────────────────────────────────────

def run_mad_debate(question: str, options: list[str], model: str,
                   n_agents: int = 3, max_rounds: int = 3) -> dict:
    """
    MAD: Multi-Agent Debate (Du et al. ICML 2024).
    No specialist roles. Majority vote on final-round answers.
    API calls: n_agents * actual_rounds (early stop reduces this)
    """
    api_calls   = 0
    audit_trail = []

    # ── Round 1: initial positions ────────────────────────
    positions = []
    for i in range(n_agents):
        msgs = prompt_initial(question, options, i)
        raw  = call_llm(model, msgs, max_tokens=MAX_TOKENS_DEBATE, think=False)
        ans  = parse_answer(raw)
        positions.append({"agent": i, "raw": raw, "answer": ans})
        api_calls += 1
        audit_trail.append({"round": 1, "agent": i + 1, "answer": ans,
                            "reasoning": extract_reasoning(raw), "raw": raw})

    # ── Rounds 2+: debate ─────────────────────────────────
    for rnd in range(2, max_rounds + 1):
        valid = [p["answer"] for p in positions if p["answer"] in ("A","B","C","D")]

        # Early stop: consensus
        if valid and len(set(valid)) == 1:
            audit_trail.append({"round": rnd, "event": "consensus",
                                 "answer": valid[0]})
            break

        new_positions = []
        for i in range(n_agents):
            # Pass full raw responses so each agent has complete reasoning context
            # (Du et al. ICML 2024 pass full responses, not summaries)
            other_args = [
                positions[j]["raw"]
                for j in range(n_agents) if j != i
            ]
            msgs = prompt_debate(question, options, i, rnd, other_args)
            raw  = call_llm(model, msgs, max_tokens=MAX_TOKENS_DEBATE, think=False)
            ans  = parse_answer(raw)
            new_positions.append({"agent": i, "raw": raw, "answer": ans})
            api_calls += 1
            audit_trail.append({"round": rnd, "agent": i + 1, "answer": ans,
                                "reasoning": extract_reasoning(raw), "raw": raw})

        positions = new_positions

    # Final answers from last round
    final_votes = [p["answer"] for p in positions]

    return {
        "answer":           majority_vote(final_votes),
        "final_votes":      final_votes,
        "individual_votes": final_votes,
        "audit_trail":      audit_trail,
        "api_calls":        api_calls,
    }
