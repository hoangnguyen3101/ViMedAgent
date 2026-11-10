"""Offline tests for the consultation ablations A and B — no model server needed.

Run from code/:  python -m pytest tests/test_ablation.py -q
"""
import json
import os
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import infrastructure.config as cfg
import infrastructure.llm_client as llm
import frameworks.tier3.medagents as std
import frameworks.tier3.medagents_ablation as abl
from frameworks.utils import extract_reasoning
import runner.run_experiments as runner

Q = "Bệnh nhân nam 45 tuổi đau ngực trái. Chẩn đoán phù hợp nhất?"
OPTS = ["Viêm phổi", "Nhồi máu cơ tim", "Trào ngược dạ dày", "Viêm cơ tim"]
QWEN, L70, L8 = abl.HETERO_PANEL
LONG_REASONING = "Phân tích lâm sàng chi tiết về triệu chứng và cận lâm sàng. " * 30   # > 500 chars


class FakeLLM:
    """Deterministic stand-in for call_llm: records every call, answers by step and model."""

    def __init__(self, roles_text="Cardiologist, Pulmonologist, Gastroenterologist", by_model=None):
        self.calls = []
        self.roles_text = roles_text
        self.by_model = by_model or {}

    def __call__(self, model, messages, **kwargs):
        self.calls.append({"model": model, "messages": messages, "kwargs": kwargs})
        text = json.dumps(messages, ensure_ascii=False)
        if "coordinating a medical consultation" in text:
            return self.roles_text
        if "concise consultation report" in text:
            return "Report: experts mostly agree on B."
        if "Respond with only 1 character" in text:
            return "B"
        letter = self.by_model.get(model, "B")
        return f"{LONG_REASONING}Final answer: {letter}"


def _patch(monkeypatch, fake):
    monkeypatch.setattr(std, "call_llm", fake)
    monkeypatch.setattr(abl, "call_llm", fake)
    monkeypatch.setattr(abl, "get_last_call",
                        lambda: {"tokens_in": 11, "tokens_out": 7, "finish_reason": "stop",
                                 "temperature": 0.3, "max_tokens": 1024})


def _steps(r):
    return [(s["step"], s["model"]) for s in r["audit_trail"]]


# ── shared flow == MedAgents ──────────────────────────────

def test_same_prompts_and_flow_as_medagents_when_panel_is_one_model(monkeypatch):
    """panel (Qwen,)*3 + report == standard MedAgents: identical messages, kwargs, order, outputs."""
    f_std, f_abl = FakeLLM(), FakeLLM()
    _patch(monkeypatch, f_std)
    r_std = std.run_medagents(Q, OPTS, QWEN)
    _patch(monkeypatch, f_abl)
    r_abl = abl.run_medagents_ablation(Q, OPTS, QWEN, (QWEN,) * 3, "report")

    assert len(f_std.calls) == len(f_abl.calls) == 9
    for a, b in zip(f_std.calls, f_abl.calls):
        assert a == b
    for k in ("answer", "roles", "round1_answers", "round2_answers", "individual_votes", "api_calls"):
        assert r_std[k] == r_abl[k]
    strip = lambda at: [{k: v for k, v in s.items() if k not in ("model", "call")} for s in at]
    assert strip(r_std["audit_trail"]) == strip(r_abl["audit_trail"])


# ── A: heterogeneous panel ────────────────────────────────

def test_A_routes_each_step_to_the_right_model(monkeypatch):
    fake = FakeLLM(by_model={QWEN: "B", L70: "C", L8: "D"})
    _patch(monkeypatch, fake)
    r = abl.run_medagents_hetero(Q, OPTS, QWEN)

    assert [c["model"] for c in fake.calls] == [QWEN, QWEN, L70, L8, QWEN, QWEN, L70, L8, QWEN]
    assert r["api_calls"] == 9
    assert _steps(r) == [("expert_gathering", QWEN), ("analysis", QWEN), ("analysis", L70), ("analysis", L8),
                         ("report", QWEN), ("consultation", QWEN), ("consultation", L70),
                         ("consultation", L8), ("decision", QWEN)]
    assert r["round1_answers"] == ["B", "C", "D"]      # expert i answered by panel[i] ...
    assert r["round2_answers"] == ["B", "C", "D"]      # ... and keeps that model in consultation
    roles = [s["role"] for s in r["audit_trail"] if s["step"] == "analysis"]
    assert roles == ["Cardiologist", "Pulmonologist", "Gastroenterologist"]   # role i -> panel[i]
    assert all(s["call"]["finish_reason"] == "stop" for s in r["audit_trail"])


@pytest.mark.parametrize("roles_text, expected", [
    ("Cardiologist, Pulmonologist", ["Cardiologist", "Pulmonologist", "General Practitioner"]),
    ("A, B, C, D, E", ["A", "B", "C"]),
    ("", ["General Practitioner", "Internal Medicine Physician", "Clinical Pharmacologist"]),
])
def test_A_role_count_is_always_three(monkeypatch, roles_text, expected):
    fake = FakeLLM(roles_text=roles_text)
    _patch(monkeypatch, fake)
    r = abl.run_medagents_hetero(Q, OPTS, QWEN)
    assert r["roles"] == expected
    assert [c["model"] for c in fake.calls] == [QWEN, QWEN, L70, L8, QWEN, QWEN, L70, L8, QWEN]


# ── B: no report ──────────────────────────────────────────

def test_B_skips_report_and_uses_one_model(monkeypatch):
    fake = FakeLLM()
    _patch(monkeypatch, fake)
    r = abl.run_medagents_noreport(Q, OPTS, QWEN)
    assert r["api_calls"] == 8 and len(fake.calls) == 8
    assert _steps(r) == [("expert_gathering", QWEN), *[("analysis", QWEN)] * 3,
                         *[("consultation", QWEN)] * 3, ("decision", QWEN)]
    assert not any("concise consultation report" in json.dumps(c["messages"]) for c in fake.calls)


def test_B_consultation_prompt_differs_only_in_the_report_block(monkeypatch):
    fake = FakeLLM(by_model={QWEN: "C"})
    _patch(monkeypatch, fake)
    r = abl.run_medagents_noreport(Q, OPTS, QWEN)
    analyses = [s for s in r["audit_trail"] if s["step"] == "analysis"]
    consult_calls = [c for c in fake.calls if "Your previous analysis" in json.dumps(c["messages"], ensure_ascii=False)]
    assert len(consult_calls) == 3

    block = "\n\n".join(f"[{a['role']}] (Answer: {a['answer']}):\n{a['raw']}" for a in analyses)
    for call, a in zip(consult_calls, analyses):
        expected = std.prompt_consult(Q, OPTS, a["role"], "@@", extract_reasoning(a["raw"]))
        expected[-1]["content"] = expected[-1]["content"].replace(
            "Consultation report from all specialists:\n@@", "Analyses from all specialists:\n" + block)
        assert call["messages"] == expected
        assert call["kwargs"] == {"temperature": 0.3, "max_tokens": cfg.MAX_TOKENS_MEDIUM, "think": False}
    # B2 = verbatim: the full analysis text (well beyond the ~500-char cut) and its answer reach the consultant
    user = consult_calls[0]["messages"][-1]["content"]
    assert LONG_REASONING in user and "Final answer: C" in user and "(Answer: C)" in user


def test_B1_option_uses_the_truncated_analyses(monkeypatch):
    fake = FakeLLM()
    _patch(monkeypatch, fake)
    r = abl.run_medagents_ablation(Q, OPTS, QWEN, (QWEN,) * 3, "analyses_truncated")
    user = [c for c in fake.calls if "Your previous analysis" in json.dumps(c["messages"], ensure_ascii=False)][0]["messages"][-1]["content"]
    assert LONG_REASONING not in user
    assert extract_reasoning(r["audit_trail"][1]["raw"]) in user


def test_rejects_unknown_consult_input():
    with pytest.raises(ValueError):
        abl.run_medagents_ablation(Q, OPTS, QWEN, (QWEN,) * 3, "summary")


# ── client, config, runner ────────────────────────────────

def test_call_llm_records_last_call_and_forces_temperature(monkeypatch):
    seen = {}

    def create(**kw):
        seen.update(kw)
        return SimpleNamespace(usage=SimpleNamespace(prompt_tokens=40, completion_tokens=1024),
                               choices=[SimpleNamespace(finish_reason="length",
                                                        message=SimpleNamespace(content=" text "))])

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    monkeypatch.setattr(llm, "_get_client", lambda api_key, base_url: client)
    monkeypatch.setattr(llm, "FORCE_TEMPERATURE", 0.3)
    llm.reset_usage()
    out = llm.call_llm(L70, [{"role": "user", "content": "hi"}], temperature=0.5, max_tokens=1024, think=False)

    assert out == "text"
    assert seen["temperature"] == 0.3
    info = llm.get_last_call()
    assert info["model"] == L70 and info["model_id"] == cfg.MODELS[L70]["model_id"]
    assert (info["tokens_in"], info["tokens_out"], info["finish_reason"]) == (40, 1024, "length")
    assert llm.get_usage()["n_llm_calls"] == 1


def test_results_routing_and_per_model_urls(monkeypatch):
    assert cfg.get_results_dir("data/vm14k_sub2000.jsonl") == "results_ablation"
    assert cfg.get_results_dir("data/vm14k_clean.jsonl") == "results"
    assert cfg.model_url_env(L70) == "SGLANG_BASE_URL__SGLANG_LLAMA3_3_70B"
    monkeypatch.delenv(cfg.model_url_env(L70), raising=False)
    assert cfg.get_model_base_url(L70) == cfg.SGLANG_BASE_URL
    monkeypatch.setenv(cfg.model_url_env(L70), "http://gx10:30001/v1")
    assert cfg.get_model_base_url(L70) == "http://gx10:30001/v1"
    assert cfg.get_model_base_url(QWEN) == cfg.SGLANG_BASE_URL   # other models untouched


@pytest.mark.parametrize("framework, n_calls", [("medagents_hetero", 9), ("medagents_noreport", 8)])
def test_dispatch_routes_ablations(monkeypatch, framework, n_calls):
    fake = FakeLLM()
    _patch(monkeypatch, fake)
    sample = {"question": Q, "options": OPTS, "difficulty_level": "Easy", "medical_topic": ["Cardiology"]}
    r = runner.dispatch(framework, sample, QWEN, 42)
    assert r["api_calls"] == n_calls and len(fake.calls) == n_calls


@pytest.mark.parametrize("framework, panel, consult_input", [
    ("medagents_hetero", [QWEN, L70, L8], "report"),
    ("medagents_noreport", [QWEN, QWEN, QWEN], "analyses_full"),
])
def test_manifest_records_panel_and_consult_input(tmp_path, framework, panel, consult_input):
    out = str(tmp_path / f"{framework}__x__vm14k_sub2000__s42.jsonl")
    runner._write_manifest(out, framework, QWEN, "data/vm14k_sub2000.jsonl", 42, 2000)
    meta = json.load(open(out + ".meta.json", encoding="utf-8"))
    assert [p["model"] for p in meta["panel"]] == panel
    assert meta["consult_input"] == consult_input
    assert "force_temperature" in meta and "cot_sc_temperature" in meta


class _Resp:
    def __init__(self, payload): self.payload = payload
    def __enter__(self): return self
    def __exit__(self, *a): return False
    def read(self): return json.dumps(self.payload).encode()


def _serve(monkeypatch, mapping):
    """mapping: url -> served model id."""
    import urllib.request

    def urlopen(req, timeout=None):
        url = req.full_url.rsplit("/models", 1)[0]
        return _Resp({"data": [{"id": mapping[url]}]})

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)


def test_check_endpoints_accepts_correct_servers(monkeypatch):
    urls = {QWEN: "http://h:30000/v1", L70: "http://h:30001/v1", L8: "http://h:30002/v1"}
    for m, u in urls.items():
        monkeypatch.setenv(cfg.model_url_env(m), u)
    _serve(monkeypatch, {u: cfg.MODELS[m]["model_id"] for m, u in urls.items()})
    runner.check_endpoints([QWEN, *abl.HETERO_PANEL])   # no exception


def test_check_endpoints_rejects_wrong_model_and_shared_url(monkeypatch):
    urls = {QWEN: "http://h:30000/v1", L70: "http://h:30000/v1", L8: "http://h:30002/v1"}
    for m, u in urls.items():
        monkeypatch.setenv(cfg.model_url_env(m), u)
    _serve(monkeypatch, {"http://h:30000/v1": cfg.MODELS[QWEN]["model_id"],
                         "http://h:30002/v1": cfg.MODELS[QWEN]["model_id"]})
    with pytest.raises(SystemExit) as e:
        runner.check_endpoints([QWEN, *abl.HETERO_PANEL])
    msg = str(e.value)
    assert "share the endpoint" in msg and "expected" in msg
