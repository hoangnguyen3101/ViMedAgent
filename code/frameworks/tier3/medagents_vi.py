"""
ViMedAgent — MedAgents_VI (VM14K-adapted, contribution of this paper)

Same 5-step MedAgents framework (Tang et al. ACL 2024), with one change:
  Step 1 (Expert Gathering) uses VM14K's medical_topic field directly
  instead of asking the LLM to suggest roles. This eliminates 1 API call
  and grounds role selection in the dataset's verified specialty labels.

Topics outside the known mapping fall back to generic roles.
API calls per question: N (analysis) + 1 (report) + N (discuss) + 1 (decision)
  With N=3 experts: 3+1+3+1 = 8 calls
"""

from infrastructure.llm_client import call_llm, parse_answer
from infrastructure.config import (
    MAX_TOKENS_SHORT, MAX_TOKENS_MEDIUM, MAX_TOKENS_AGENT,
)
from frameworks.utils import extract_reasoning
from frameworks.tier3.medagents import (
    prompt_analysis, prompt_report,
    prompt_consult, prompt_decision,
)


# ── Topic → Specialist mapping ────────────────────────────
# Only VM14K's verified medical topics are mapped.
# Topics outside this dict (noise/non-medical) fall back to GENERIC_ROLES.
TOPIC_TO_ROLES = {
    "Obstetrics and Gynecology":           ["Obstetrician", "Gynecologist"],
    "Gastroenterology":                    ["Gastroenterologist", "General Practitioner"],
    "Pulmonology":                         ["Pulmonologist", "General Practitioner"],
    "Infectious Diseases":                 ["Infectious Disease Specialist", "General Practitioner"],
    "Endocrinology":                       ["Endocrinologist", "General Practitioner"],
    "Oncology":                            ["Oncologist", "Pathologist"],
    "Pathology":                           ["Pathologist", "General Practitioner"],
    "Pediatrics":                          ["Pediatrician", "General Practitioner"],
    "Radiology":                           ["Radiologist", "General Practitioner"],
    "Urology":                             ["Urologist", "Nephrologist"],
    "Pharmacology":                        ["Clinical Pharmacologist", "General Practitioner"],
    "General Medicine":                    ["General Practitioner", "Internal Medicine Physician"],
    "Nephrology":                          ["Nephrologist", "General Practitioner"],
    "Surgery":                             ["Surgeon", "Anesthesiologist"],
    "Internal Medicine":                   ["Internal Medicine Physician", "General Practitioner"],
    "Neurology":                           ["Neurologist", "General Practitioner"],
    "Toxicology":                          ["Toxicologist", "Emergency Physician"],
    "Hematology":                          ["Hematologist", "General Practitioner"],
    "Public Health":                       ["Public Health Physician", "General Practitioner"],
    "Preventive Healthcare":               ["Public Health Physician", "General Practitioner"],
    "Cardiology":                          ["Cardiologist", "General Practitioner"],
    "Otolaryngology":                      ["ENT Specialist", "General Practitioner"],
    "Eastern Medicine":                    ["Traditional Medicine Specialist", "General Practitioner"],
    "Emergency Medicine":                  ["Emergency Physician", "General Practitioner"],
    "Allergy and Immunology":              ["Allergist", "Immunologist"],
    "Orthopedics":                         ["Orthopedic Surgeon", "General Practitioner"],
    "Anatomy":                             ["General Practitioner", "Pathologist"],
    "Dermatology":                         ["Dermatologist", "General Practitioner"],
    "Psychiatry":                          ["Psychiatrist", "General Practitioner"],
    "Genetics":                            ["Geneticist", "General Practitioner"],
    "Geriatrics":                          ["Geriatrician", "General Practitioner"],
    "Physical Medicine and Rehabilitation":["Rehabilitation Specialist", "General Practitioner"],
    "Rheumatology":                        ["Rheumatologist", "General Practitioner"],
    "Ophthalmology":                       ["Ophthalmologist", "General Practitioner"],
    "Anesthesiology":                      ["Anesthesiologist", "General Practitioner"],
    "Biochemistry":                        ["General Practitioner", "Clinical Pharmacologist"],
    "Dentistry":                           ["Dentist", "General Practitioner"],
    "Physiology":                          ["General Practitioner", "Internal Medicine Physician"],
    "Periodontology":                      ["Periodontist", "Dentist"],
    "Environmental Health":                ["Public Health Physician", "General Practitioner"],
    "Cell Biology":                        ["Pathologist", "General Practitioner"],
    "Parasitology":                        ["Infectious Disease Specialist", "General Practitioner"],
    "Molecular Biology":                   ["Pathologist", "General Practitioner"],
    "Sports Medicine":                     ["Sports Medicine Physician", "Orthopedic Surgeon"],
    "Palliative Medicine":                 ["Palliative Care Specialist", "General Practitioner"],
    "Nuclear Medicine":                    ["Nuclear Medicine Specialist", "Radiologist"],
    "Embryology":                          ["General Practitioner", "Pathologist"],
    "Veterinary Medicine":                 ["Infectious Disease Specialist", "General Practitioner"],
    "Neurosurgery":                        ["Neurosurgeon", "Neurologist"],
    "Endodontics":                         ["Endodontist", "Dentist"],
    "Microbiology":                        ["Microbiologist", "Infectious Disease Specialist"],
    "Immunology":                          ["Immunologist", "General Practitioner"],
    "Laboratory Medicine":                 ["Pathologist", "General Practitioner"],
    "Pain Management":                     ["Anesthesiologist", "General Practitioner"],
    "Epidemiology":                        ["Public Health Physician", "General Practitioner"],
    "Nutrition":                           ["General Practitioner", "Internal Medicine Physician"],
    "Hepatology":                          ["Hepatologist", "Gastroenterologist"],
    "Medical Microbiology":                ["Microbiologist", "Infectious Disease Specialist"],
    "Pharmacognosy":                       ["Clinical Pharmacologist", "General Practitioner"],
    "Virology":                            ["Infectious Disease Specialist", "Microbiologist"],
    "Occupational Medicine":               ["Occupational Medicine Specialist", "General Practitioner"],
    "Nursing":                             ["General Practitioner", "Internal Medicine Physician"],
    "Neonatology":                         ["Neonatologist", "Pediatrician"],
    "Vascular Surgery":                    ["Vascular Surgeon", "General Practitioner"],
    "Addiction Medicine":                  ["Psychiatrist", "General Practitioner"],
    "Histology":                           ["Pathologist", "General Practitioner"],
    "Audiology":                           ["Audiologist", "ENT Specialist"],
}

GENERIC_ROLES = ["General Practitioner", "Internal Medicine Physician",
                 "Clinical Pharmacologist"]


def get_roles_from_topic(medical_topic, max_roles: int = 3) -> list[str]:
    """
    Map VM14K medical_topic → specialist roles.

    Rules:
    - medical_topic can be a string or list (VM14K uses list)
    - Only topics in TOPIC_TO_ROLES are mapped
    - Topics outside (noise from GPT-4o labeling) are ignored
    - If all topics are noise/unknown → fallback to GENERIC_ROLES
    - Max max_roles roles returned (deduplicated, insertion-order preserved)
    """
    topics = medical_topic if isinstance(medical_topic, list) else [medical_topic]
    topics = [t for t in topics if t]  # remove empty strings

    roles = []
    for t in topics:
        mapped = TOPIC_TO_ROLES.get(t)
        if mapped:
            roles.extend(mapped)

    # Deduplicate, preserve order
    seen, result = set(), []
    for r in roles:
        if r not in seen:
            seen.add(r)
            result.append(r)

    if not result:
        return GENERIC_ROLES[:max_roles]

    return result[:max_roles]


# ── Runner ────────────────────────────────────────────────

def run_medagents_vi(question: str, options: list[str], model: str,
                     medical_topic=None, n_experts: int = 3) -> dict:
    """
    MedAgents_VI: VM14K-adapted variant (contribution of this paper).

    Replaces Step 1 (LLM-based expert gathering) with direct lookup
    of VM14K's medical_topic field. All other steps identical to original.

    API calls: n_experts + 1 + n_experts + 1 = 2*n_experts + 2
      (default n=3 → 8 calls, saves 1 vs original)
    """
    api_calls   = 0
    audit_trail = []

    # ── Step 1: Topic-Aware Expert Selection (no API call) ─
    roles = get_roles_from_topic(medical_topic, max_roles=n_experts)
    # Pad to n_experts if fewer roles available
    while len(roles) < n_experts:
        roles.append("General Practitioner")
    roles = roles[:n_experts]
    audit_trail.append({"step": "expert_gathering", "method": "topic_aware",
                        "medical_topic": medical_topic, "roles": roles})

    # ── Step 2: Analysis Proposition ─────────────────────
    analyses = []
    for role in roles:
        msgs = prompt_analysis(question, options, role)
        raw  = call_llm(model, msgs, temperature=0.3, max_tokens=MAX_TOKENS_AGENT, think=False)
        ans  = parse_answer(raw)
        api_calls += 1
        analyses.append({
            "role":     role,
            "analysis": extract_reasoning(raw),
            "answer":   ans,
            "raw":      raw,
        })
        audit_trail.append({"step": "analysis", "role": role, "answer": ans,
                            "reasoning": analyses[-1]["analysis"], "raw": raw})

    # ── Step 3: Report Summarization ─────────────────────
    msgs   = prompt_report(question, options, analyses)
    report = call_llm(model, msgs, max_tokens=MAX_TOKENS_MEDIUM, think=False)
    api_calls += 1
    audit_trail.append({"step": "report", "raw": report})

    # ── Step 4: Collaborative Consultation ───────────────
    final_opinions = []
    for a in analyses:
        msgs = prompt_consult(question, options, a["role"], report, a["analysis"])
        raw  = call_llm(model, msgs, temperature=0.3, max_tokens=MAX_TOKENS_MEDIUM, think=False)
        ans  = parse_answer(raw)
        api_calls += 1
        final_opinions.append({
            "role":      a["role"],
            "answer":    ans,
            "reasoning": extract_reasoning(raw),
            "raw":       raw,
        })
        audit_trail.append({"step": "consultation", "role": a["role"], "answer": ans,
                            "reasoning": final_opinions[-1]["reasoning"], "raw": raw})

    # ── Step 5: Decision Making ───────────────────────────
    msgs    = prompt_decision(question, options, final_opinions)
    raw_dec = call_llm(model, msgs, max_tokens=MAX_TOKENS_SHORT, think=False)
    ans_dec = parse_answer(raw_dec)
    api_calls += 1
    audit_trail.append({"step": "decision", "answer": ans_dec, "raw": raw_dec})

    return {
        "answer":          ans_dec,
        "roles":           roles,
        "round1_answers":  [a["answer"] for a in analyses],
        "round2_answers":  [op["answer"] for op in final_opinions],
        "individual_votes":[op["answer"] for op in final_opinions],
        "audit_trail":     audit_trail,
        "api_calls":       api_calls,
    }
