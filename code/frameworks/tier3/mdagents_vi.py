"""
ViMedAgent — MDAgents_VI (VM14K-adapted, contribution of this paper)

Same 4-step MDAgents framework (Kim et al. NeurIPS 2024), with one change:
  Step 1 (Complexity Check) uses VM14K's difficulty_level field directly
  instead of asking the LLM to classify. This follows MedAgentBoard
  benchmark evaluation protocol and eliminates 1 API call.

Mapping: Easy → Low, Medium → Moderate, Challenging/Hard → High

API calls:
  Low (Easy):          1 (PCC) = 1
  Moderate (Medium):   3 (MDT) = 3
  High (Hard/Chall.):  5 (ICT) + 1 (synth) = 6
"""

from infrastructure.llm_client import call_llm, parse_answer
from infrastructure.config import (
    MAX_TOKENS_SHORT, MAX_TOKENS_MEDIUM, MAX_TOKENS_AGENT,
)
from frameworks.utils import majority_vote, extract_reasoning
from frameworks.tier3.mdagents import (
    ICT_ROLES,
    prompt_pcc, prompt_mdt_agent,
    prompt_ict_expert, prompt_ict_synthesizer,
)


# ── Difficulty mapping ────────────────────────────────────

def map_difficulty(difficulty_level: str) -> str:
    """
    Map VM14K difficulty_level → MDAgents complexity tier.
    Easy → Low, Medium → Moderate, Challenging/Hard → High
    """
    d = difficulty_level.lower() if difficulty_level else "medium"
    if d == "easy":
        return "Low"
    if d in ("challenging", "hard"):
        return "High"
    return "Moderate"  # Medium or unknown


# ── Runner ────────────────────────────────────────────────

def run_mdagents_vi(question: str, options: list[str], model: str,
                    difficulty_level: str = "Medium") -> dict:
    """
    MDAgents_VI: VM14K-adapted variant (contribution of this paper).

    Replaces Step 1 (LLM-based complexity classification) with direct
    mapping from VM14K's difficulty_level field. Following MedAgentBoard
    benchmark evaluation protocol.

    API calls: 0 (check, no call) + team calls
      Easy: 1, Medium: 3, Hard: 6
    """
    api_calls   = 0
    audit_trail = []

    # ── Step 1: Dataset-based complexity mapping (no API call) ─
    complexity = map_difficulty(difficulty_level)
    audit_trail.append({"step": "complexity_check", "method": "dataset_label",
                        "difficulty_level": difficulty_level,
                        "complexity": complexity})

    # ── Step 2-4: Route by complexity ────────────────────

    if complexity == "Low":
        msgs = prompt_pcc(question, options)
        raw  = call_llm(model, msgs, max_tokens=MAX_TOKENS_MEDIUM, think=False)
        ans  = parse_answer(raw)
        api_calls += 1
        audit_trail.append({"step": "pcc", "answer": ans,
                            "reasoning": extract_reasoning(raw), "raw": raw})
        return {
            "answer":      ans,
            "complexity":  complexity,
            "mode":        "pcc",
            "audit_trail": audit_trail,
            "api_calls":   api_calls,
        }

    elif complexity == "Moderate":
        votes = []
        for i in range(3):
            msgs = prompt_mdt_agent(question, options, i)
            raw  = call_llm(model, msgs, think=False)
            ans  = parse_answer(raw)
            votes.append(ans)
            api_calls += 1
            audit_trail.append({"step": "mdt", "agent": i + 1, "answer": ans,
                                "reasoning": extract_reasoning(raw), "raw": raw})
        return {
            "answer":           majority_vote(votes),
            "complexity":       complexity,
            "mode":             "mdt",
            "individual_votes": votes,
            "audit_trail":      audit_trail,
            "api_calls":        api_calls,
        }

    else:  # High
        opinions = []
        for role in ICT_ROLES:
            msgs = prompt_ict_expert(question, options, role)
            raw  = call_llm(model, msgs, max_tokens=MAX_TOKENS_AGENT, think=False)
            ans  = parse_answer(raw)
            api_calls += 1
            opinions.append({
                "role":      role,
                "answer":    ans,
                "reasoning": extract_reasoning(raw),
            })
            audit_trail.append({"step": "ict_expert", "role": role, "answer": ans,
                                "reasoning": opinions[-1]["reasoning"], "raw": raw})

        msgs    = prompt_ict_synthesizer(question, options, opinions)
        raw_syn = call_llm(model, msgs, max_tokens=MAX_TOKENS_SHORT, think=False)
        ans_syn = parse_answer(raw_syn)
        api_calls += 1
        audit_trail.append({"step": "ict_synthesizer", "answer": ans_syn, "raw": raw_syn})

        return {
            "answer":           ans_syn,
            "complexity":       complexity,
            "mode":             "ict",
            "individual_votes": [op["answer"] for op in opinions],
            "audit_trail":      audit_trail,
            "api_calls":        api_calls,
        }
