"""Offline tests for medagents_reextract (steps 3-5 re-run on stored steps 1-2) — no model server.

Run from code/:  python -m pytest tests/test_medagents_reextract.py -q
"""
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import frameworks.tier3.medagents as std
import frameworks.tier3.medagents_reextract as rx
import runner.run_experiments as runner

Q = "A 45-year-old man has left-sided chest pain. What is the most likely diagnosis?"
OPTS = ["Pneumonia", "Myocardial infarction", "Reflux", "Myocarditis"]


class FakeLLM:
    def __init__(self, analysis_text="Analysis text. Final answer: B"):
        self.calls = []
        self.analysis_text = analysis_text

    def __call__(self, model, messages, **kwargs):
        self.calls.append({"model": model, "messages": messages, "kwargs": kwargs})
        text = json.dumps(messages, ensure_ascii=False)
        if "coordinating a medical consultation" in text:
            return "Cardiologist, Pulmonologist, Gastroenterologist"
        if "concise consultation report" in text:
            return "Report: experts agree on B."
        if "Respond with only 1 character" in text:
            return "B"
        if "Consultation report from all specialists" in text:
            return "I maintain my answer. Final answer: B"
        return self.analysis_text


def _stored_row(monkeypatch, analysis_text="Analysis text. Final answer: B"):
    fake = FakeLLM(analysis_text)
    monkeypatch.setattr(std, "call_llm", fake)
    return std.run_medagents(Q, OPTS, "m")


def test_live_calls_are_word_for_word_those_of_medagents(monkeypatch):
    stored = _stored_row(monkeypatch)
    f_std = FakeLLM()
    monkeypatch.setattr(std, "call_llm", f_std)
    std.run_medagents(Q, OPTS, "m")
    f_rx = FakeLLM()
    monkeypatch.setattr(rx, "call_llm", f_rx)
    out = rx.run_medagents_reextract(Q, OPTS, "m", stored["audit_trail"])

    assert len(f_std.calls) == 9 and len(f_rx.calls) == 5
    for a, b in zip(f_std.calls[4:], f_rx.calls):     # report, 3 revisions, decision
        assert a == b
    assert out["api_calls"] == 5
    assert out["answer"] == "B"


def test_analysis_label_is_re_extracted_and_reaches_the_report(monkeypatch):
    stored = _stored_row(monkeypatch, "Analysis. The correct answer is B.")
    trail = json.loads(json.dumps(stored["audit_trail"]))
    for t in trail[1:4]:                       # pretend the run-time parser missed every label
        t["answer"] = None
    f = FakeLLM()
    monkeypatch.setattr(rx, "call_llm", f)
    out = rx.run_medagents_reextract(Q, OPTS, "m", trail)

    assert out["round1_answers"] == ["B", "B", "B"]
    report_prompt = f.calls[0]["messages"][-1]["content"]
    assert "(Answer: B)" in report_prompt and "(Answer: None)" not in report_prompt
    assert [t["answer_at_run_time"] for t in out["audit_trail"][1:4]] == [None, None, None]


def test_first_four_steps_are_copied_and_flagged_reused(monkeypatch):
    stored = _stored_row(monkeypatch)
    monkeypatch.setattr(rx, "call_llm", FakeLLM())
    out = rx.run_medagents_reextract(Q, OPTS, "m", stored["audit_trail"])
    trail = out["audit_trail"]
    assert [t["step"] for t in trail] == [t["step"] for t in stored["audit_trail"]]
    assert [bool(t.get("reused")) for t in trail] == [True] * 4 + [False] * 5
    for new, old in zip(trail[:4], stored["audit_trail"][:4]):
        assert new["raw"] == old["raw"]
    assert out["roles"] == stored["roles"]


def test_refuses_an_incomplete_or_foreign_source_row(monkeypatch):
    stored = _stored_row(monkeypatch)
    monkeypatch.setattr(rx, "call_llm", FakeLLM())
    with pytest.raises(ValueError):
        rx.run_medagents_reextract(Q, OPTS, "m", stored["audit_trail"][:8])
    bad = json.loads(json.dumps(stored["audit_trail"]))
    bad[2]["role"] = "Dentist"
    with pytest.raises(ValueError):
        rx.run_medagents_reextract(Q, OPTS, "m", bad)
    bad = json.loads(json.dumps(stored["audit_trail"]))
    bad[1]["raw"] = None
    with pytest.raises(ValueError):
        rx.run_medagents_reextract(Q, OPTS, "m", bad)


def test_runner_loader_rejects_duplicates_and_partial_rows(tmp_path, monkeypatch):
    stored = _stored_row(monkeypatch)
    good = {"id": 1, "audit_trail": stored["audit_trail"]}
    p = tmp_path / "src.jsonl"
    p.write_text(json.dumps(good) + "\n" + json.dumps({**good, "id": 2}) + "\n")
    assert set(runner.load_source_trails(str(p))) == {1, 2}
    p.write_text(json.dumps(good) + "\n" + json.dumps(good) + "\n")
    with pytest.raises(ValueError):
        runner.load_source_trails(str(p))
    p.write_text(json.dumps({"id": 3, "audit_trail": stored["audit_trail"][:5]}) + "\n")
    with pytest.raises(ValueError):
        runner.load_source_trails(str(p))


def test_dispatch_uses_the_source_row_of_the_same_model_and_seed(monkeypatch):
    stored = _stored_row(monkeypatch)
    monkeypatch.setattr(rx, "call_llm", FakeLLM())
    runner._SOURCE_TRAILS[("m", 42)] = {7: stored["audit_trail"]}
    sample = {"id": 7, "question": Q, "options": OPTS}
    assert runner.dispatch("medagents_reextract", sample, "m", 42)["api_calls"] == 5
    with pytest.raises(KeyError):
        runner.dispatch("medagents_reextract", sample, "m", 43)
    with pytest.raises(KeyError):
        runner.dispatch("medagents_reextract", {**sample, "id": 8}, "m", 42)
