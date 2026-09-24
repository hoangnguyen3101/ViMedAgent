"""Nạp kết quả run và chuẩn hoá. NGUỒN SỰ THẬT DUY NHẤT cho mọi chuẩn hoá.

Không module nào khác được tự đọc .jsonl hoặc tự định nghĩa lại
difficulty / topic / đáp án agent.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

from infrastructure.config import (
    SEEDS, FRAMEWORK_TIER, ALL_FRAMEWORKS, MODEL_NAMES,
    RESULTS_DIR, RESULTS_TEST_DIR,
)

OPTIONS = ("A", "B", "C", "D")

#: số câu kỳ vọng theo dataset — dùng để phát hiện run chạy dở
EXPECTED_N = {"vm14k_clean": 11094, "pilot_100": 100}

#: dataset có 4 mức; paper v6 Table I dùng 3 cột Easy/Moderate/Hard
DIFFICULTY_MAP = {"Easy": "Easy", "Medium": "Moderate", "Moderate": "Moderate",
                  "Challenging": "Hard", "Hard": "Hard"}
DIFFICULTY_ORDER = ("Easy", "Moderate", "Hard")

#: bước audit_trail mang đáp án của một agent, theo từng họ framework
_AGENT_STEPS = {
    "medagents": ("analysis",),
    "medagents_vi": ("analysis",),
    "mdagents": ("pcc", "mdt", "ict_expert"),
    "mdagents_vi": ("pcc", "mdt", "ict_expert"),
}


def norm_difficulty(d) -> str:
    return DIFFICULTY_MAP.get(d, "Unknown")


def norm_topics(t) -> list[str]:
    """medical_topic là multi-label; luôn trả list đã lọc rỗng."""
    if t is None:
        return []
    if isinstance(t, str):
        t = [t]
    return [x.strip() for x in t if x and str(x).strip()]


def pre_synthesis_answers(r: dict) -> list[str]:
    """Đáp án agent đưa ra TRƯỚC bước tổng hợp."""
    fw = r.get("framework", "")
    at = r.get("audit_trail") or []
    steps = _AGENT_STEPS.get(fw)
    if steps:
        out = [s.get("answer") for s in at if s.get("step") in steps]
        if not out and fw.startswith("medagents"):
            out = r.get("round1_answers") or []
    elif fw == "mad_debate":
        out = [s.get("answer") for s in at if s.get("round") == 1 and "agent" in s]
    else:
        out = r.get("individual_votes") or []
    return [a for a in out if a in OPTIONS]


def voting_answers(r: dict) -> list[str]:
    """Đáp án đi vào bước bỏ phiếu cuối; lùi về pre-synthesis nếu không có."""
    out = [a for a in (r.get("individual_votes")
                       or r.get("round2_answers")
                       or r.get("final_votes") or []) if a in OPTIONS]
    return out or pre_synthesis_answers(r)


def agent_roles(r: dict) -> list[str]:
    """Role đã gán cho từng agent; chỉ họ medagents lưu trường này."""
    for s in (r.get("audit_trail") or []):
        if s.get("step") == "expert_gathering":
            return list(s.get("roles") or [])
    return []


def normalize(rec: dict, framework: str = None, model: str = None) -> dict:
    rec.setdefault("framework", framework)
    rec.setdefault("model", model)
    rec["_difficulty"] = norm_difficulty(rec.get("difficulty"))
    rec["_topics"] = norm_topics(rec.get("topic"))
    rec["_correct"] = (rec.get("answer") == rec.get("gold"))
    rec["_key"] = (rec.get("id"), rec.get("seed"))
    return rec


# ── đọc file ──────────────────────────────────────────────

def results_root(dataset: str) -> str:
    return RESULTS_TEST_DIR if dataset.startswith("pilot") else RESULTS_DIR


def run_path(framework, model, dataset, seed, root=None) -> str:
    safe = model.replace("/", "_").replace(".", "_")
    root = root or results_root(dataset)
    return os.path.join(root, FRAMEWORK_TIER[framework], framework,
                        f"{framework}__{safe}__{dataset}__s{seed}.jsonl")


def load_run(framework, model, dataset, seed, root=None) -> list[dict]:
    path = run_path(framework, model, dataset, seed, root)
    if not os.path.exists(path):
        return []
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(normalize(json.loads(line), framework, model))
            except json.JSONDecodeError:
                continue
    return out


@dataclass
class Cell:
    """Một ô (framework, model) với mọi seed có sẵn."""
    framework: str
    model: str
    dataset: str
    seeds: dict[int, list[dict]] = field(default_factory=dict)

    @property
    def expected_n(self) -> int:
        return EXPECTED_N.get(self.dataset, 0)

    @property
    def complete_seeds(self) -> list[int]:
        exp = self.expected_n
        return sorted(s for s, rs in self.seeds.items() if exp and len(rs) >= exp)

    @property
    def n_seeds(self) -> int:
        return len(self.seeds)

    @property
    def status(self) -> str:
        if not self.seeds:
            return "missing"
        n_ok = len(self.complete_seeds)
        if n_ok == 0:
            return "partial"
        if n_ok < len(SEEDS):
            return f"incomplete ({n_ok}/{len(SEEDS)} seed)"
        return "ok"

    @property
    def reportable(self) -> bool:
        """Chỉ ô đủ cả 3 seed hoàn chỉnh mới được vào bảng paper."""
        return len(self.complete_seeds) >= len(SEEDS)

    def per_seed(self, only_complete=True) -> list[list[dict]]:
        keys = self.complete_seeds if only_complete else sorted(self.seeds)
        return [self.seeds[s] for s in keys]

    def flat(self, only_complete=True) -> list[dict]:
        return [r for rs in self.per_seed(only_complete) for r in rs]


def load_cell(framework, model, dataset, root=None) -> Cell:
    c = Cell(framework, model, dataset)
    for seed in SEEDS:
        rs = load_run(framework, model, dataset, seed, root)
        if rs:
            c.seeds[seed] = rs
    return c


def load_matrix(dataset, frameworks=None, models=None, root=None) -> dict:
    frameworks = frameworks or ALL_FRAMEWORKS
    models = models or MODEL_NAMES
    return {(fw, m): load_cell(fw, m, dataset, root)
            for fw in frameworks for m in models}


def discovered_models(dataset: str, root=None) -> list[str]:
    root = root or results_root(dataset)
    found = set()
    for fw in ALL_FRAMEWORKS:
        d = os.path.join(root, FRAMEWORK_TIER[fw], fw)
        if not os.path.isdir(d):
            continue
        for fn in os.listdir(d):
            parts = fn[:-6].split("__") if fn.endswith(".jsonl") else []
            if len(parts) == 4 and parts[2] == dataset:
                found.add(parts[1])
    return [m for m in MODEL_NAMES
            if m.replace("/", "_").replace(".", "_") in found]
