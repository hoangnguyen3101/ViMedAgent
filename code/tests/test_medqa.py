"""Offline regressions for VM14K and five-option MedQA; no model requests."""

import contextlib
import importlib
import io
import json
from pathlib import Path
import re
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.loading import load_dataset, normalize_sample
from frameworks.utils import majority_vote
from infrastructure.llm_client import parse_answer, ANSWER_PARSER_VERSION
from infrastructure import config
from metrics.accuracy import pass_at_k
from metrics.compute import compute_pass_at_1, compute_pass_at_k, compute_nmar
from metrics.io import load_cell, normalize, pre_synthesis_answers, voting_answers, run_path
from metrics.validity import answer_distribution
from runner import run_experiments as runner
from scripts.compare_results import load as load_comparison


def medqa_record():
    return {"question": "A clinical question?",
            "options": {"E": "target", "C": "third", "A": "first",
                        "D": "fourth", "B": "second"},
            "answer": "target", "answer_idx": "E", "meta_info": "step1"}


class DatasetTests(unittest.TestCase):
    def test_medqa_uses_answer_idx_and_orders_options(self):
        record = medqa_record()
        sample = normalize_sample(record, "dev", 7)
        self.assertEqual(sample["options"], ["first", "second", "third", "fourth", "target"])
        self.assertEqual(sample["answer"], "E")
        self.assertEqual(sample["id"], "medqa_us_dev_000007")
        self.assertEqual(record, medqa_record())
        self.assertNotIn("difficulty_level", sample)
        self.assertNotIn("medical_topic", sample)

    def test_vm14k_is_preserved(self):
        record = {"id": "vm1", "question": "Question?", "options": ["a", "b", "c", "d"],
                  "answer": "D", "difficulty_level": "Easy", "medical_topic": ["Cardiology"]}
        self.assertEqual(normalize_sample(record, "vm14k", 1), record)

    def test_invalid_records_fail_before_inference(self):
        invalid = [dict(medqa_record(), answer="wrong text"),
                   dict(medqa_record(), answer_idx="F"),
                   dict(medqa_record(), question=None),
                   dict(medqa_record(), id=[]),
                   dict(medqa_record(), options={"A": "a", "B": "b", "C": "c"}),
                   dict(medqa_record(), options={"A": "a", "B": "b", "C": "c", "D": "d", "F": "f"}),
                   dict(medqa_record(), options=["a", "b", "c", "d", ""])]
        for record in invalid:
            with self.subTest(record=record), self.assertRaises(ValueError):
                normalize_sample(record, "dev", 1)

    def test_loader_rejects_duplicates_and_reports_line_number(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "dev.jsonl"
            record = dict(medqa_record(), id="duplicate")
            path.write_text((json.dumps(record) + "\n") * 2, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, r"dev.jsonl:2: Duplicate id"):
                load_dataset(str(path))


class AnswerTests(unittest.TestCase):
    def test_all_parser_formats_accept_e_only_for_five_options(self):
        outputs = ["E", "Final answer: E", "Final answer: **E**", "**E**", "(E)",
                   "The answer: e", "Conclusion E", "<think>A is possible.</think>E"]
        for text in outputs:
            with self.subTest(text=text):
                self.assertEqual(parse_answer(text, n_options=5), "E")
                self.assertIsNone(parse_answer(text))
        self.assertIsNone(parse_answer("F", n_options=5))
        self.assertIsNone(parse_answer("<think>E</think>", n_options=5))

    def test_four_option_parser_priority_is_preserved(self):
        self.assertEqual(parse_answer("Option A is possible. Final answer: D"), "D")
        self.assertEqual(parse_answer("The patient is a 62-year-old man."), None)
        self.assertEqual(parse_answer("<think>A</think>Final answer: C"), "C")

    def test_majority_and_tie_breaking_include_e(self):
        self.assertEqual(majority_vote(["E", "A", "E"], n_options=5), "E")
        self.assertEqual(majority_vote(["E", "A"], fallback="E", n_options=5), "E")
        self.assertEqual(majority_vote(["E", "A"], n_options=5), "A")
        self.assertIsNone(majority_vote(["E"]))


class FrameworkTests(unittest.TestCase):
    def test_all_frameworks_and_adaptive_routes_with_four_and_five_options(self):
        for n in (4, 5):
            label = "D" if n == 4 else "E"
            options = [f"choice{i}" for i in range(n - 1)] + ["TARGET_CHOICE"]
            raw = f"Reasoning. Final answer: {label}"
            roles = "1. Cardiologist\n2. Pharmacologist\n3. General Practitioner"
            cases = [
                ("tier1.zero_shot", "run_zero_shot", {}, [raw]),
                ("tier1.cot", "run_cot", {}, [raw]),
                ("tier1.cot_sc", "run_cot_sc", {}, [raw, "B", raw, "B", raw]),
                ("tier1.self_refine", "run_self_refine", {},
                 [raw, "INCORRECT. Review the evidence.", raw, "CORRECT. No changes needed."]),
                ("tier3.medagents", "run_medagents", {}, [roles] + [raw] * 8),
                ("tier3.medagents_vi", "run_medagents_vi", {}, [raw] * 8),
                ("tier3.mad_debate", "run_mad_debate", {}, [raw, "B", raw] + [raw] * 3),
            ]
            for complexity, calls, difficulty in [("Low", 1, "Easy"), ("Moderate", 3, "Medium"), ("High", 6, "Hard")]:
                cases.append(("tier3.mdagents", "run_mdagents", {}, [complexity] + [raw] * calls))
                cases.append(("tier3.mdagents_vi", "run_mdagents_vi", {"difficulty_level": difficulty}, [raw] * calls))
            for module_name, function_name, kwargs, outputs in cases:
                module = importlib.import_module("frameworks." + module_name)
                with self.subTest(n=n, framework=function_name, kwargs=kwargs), \
                        patch.object(module, "call_llm", side_effect=outputs) as mocked:
                    result = getattr(module, function_name)("Question?", options, "sglang-qwen3-32b", **kwargs)
                    self.assertEqual(result["answer"], label)
                    self.assertEqual(result["api_calls"], mocked.call_count)
                    for call in mocked.call_args_list:
                        prompt = "\n".join(m["content"] for m in call.args[1])
                        self.assertIn(f"{label}. TARGET_CHOICE", prompt)
                        if n == 5:
                            self.assertNotIn("or D.", prompt)
                            self.assertNotIn("(A/B/C/D)", prompt)

    def test_shuffled_voting_maps_fifth_option_back_to_original(self):
        module = importlib.import_module("frameworks.tier2.voting_ensemble")
        for n in (4, 5):
            options = ["other"] * (n - 1) + ["TARGET_CHOICE"]
            seen_labels = []

            def answer_target(model, messages, **kwargs):
                letter = re.search(r"^([A-E])\. TARGET_CHOICE$", messages[0]["content"], re.M).group(1)
                seen_labels.append(letter)
                return letter

            with patch.object(module, "call_llm", side_effect=answer_target):
                result = module.run_voting_ensemble("Question?", options, "sglang-qwen3-32b", n=12, seed=42)
            label = "D" if n == 4 else "E"
            self.assertEqual(result["individual_votes"], [label] * 12)
            self.assertEqual(result["answer"], label)
            self.assertGreater(len(set(seen_labels)), 1)


class MetricsAndRunnerTests(unittest.TestCase):
    def test_results_routing_for_usmle_vm14k_and_pilots(self):
        cases = {
            "dev": "results_USMLE", "train": "results_USMLE", "test": "results_USMLE",
            "phrases_no_exclude_dev": "results_USMLE",
            "phrases_no_exclude_train": "results_USMLE",
            "phrases_no_exclude_test": "results_USMLE",
            "medqa_us_dev": "results_USMLE", "usmle_dev": "results_USMLE",
            "pilot_usmle_100": "results_USMLE",
            "vm14k_clean": "results", "pilot_100": "results_test",
        }
        for dataset, expected in cases.items():
            with self.subTest(dataset=dataset):
                path = runner.result_path("zero_shot", "sglang-qwen3-32b", dataset, 42)
                self.assertEqual(Path(path).parts[0], expected)
                self.assertEqual(path, run_path("zero_shot", "sglang-qwen3-32b", dataset, 42))

    def test_e_is_retained_by_accuracy_and_collaboration_metrics(self):
        record = normalize({"id": "x", "seed": 42, "framework": "cot_sc", "gold": "E", "answer": "E",
                            "individual_votes": ["E", "A", "E"]})
        self.assertEqual(compute_pass_at_1([record])["overall"], 100)
        self.assertEqual(compute_pass_at_k([record], 1)["overall"], 100)
        self.assertEqual(pass_at_k([record], 1)["pass_at_k"], 100)
        self.assertEqual(pre_synthesis_answers(record), ["E", "A", "E"])
        self.assertEqual(voting_answers(record), ["E", "A", "E"])
        self.assertIn("E", answer_distribution([record])["pred_pct"])
        old_record = normalize(dict(record, gold="D", answer="D"))
        self.assertEqual(set(answer_distribution([old_record])["pred_pct"]), set("ABCD"))
        # Correct minority E must remain eligible for NMAR.
        wrong = dict(record, answer="A", individual_votes=["A", "A", "E"])
        self.assertEqual(compute_nmar([wrong]), (100.0, 1, 1))

    def test_runner_scores_medqa_and_resumes_with_same_generated_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            dataset = Path(tmp) / "dev.jsonl"
            dataset.write_text(json.dumps(medqa_record()) + "\n", encoding="utf-8")
            results_dir = str(Path(tmp) / "results_USMLE")
            with patch.object(config, "RESULTS_USMLE_DIR", results_dir), \
                    patch("frameworks.tier1.zero_shot.call_llm", return_value="E") as mocked, \
                    contextlib.redirect_stdout(io.StringIO()):
                records = runner.run_one_seed("zero_shot", "sglang-qwen3-32b", str(dataset), 42, workers=1)
                again = runner.run_one_seed("zero_shot", "sglang-qwen3-32b", str(dataset), 42, workers=1)
            self.assertEqual(mocked.call_count, 1)
            self.assertEqual(len(again), 1)
            self.assertTrue(records[0]["correct"])
            self.assertEqual(records[0]["gold"], "E")
            self.assertEqual(records[0]["id"], "medqa_us_dev_000001")
            self.assertIsNone(records[0]["difficulty"])
            self.assertIsNone(records[0]["topic"])
            manifest_path = next(Path(results_dir).rglob("*.meta.json"))
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["answer_parser_version"], ANSWER_PARSER_VERSION)
            cell = load_cell("zero_shot", "sglang-qwen3-32b", "dev", root=results_dir)
            self.assertEqual(cell.expected_n, 1)
            self.assertEqual(cell.complete_seeds, [42])

    def test_bad_input_creates_no_results_and_makes_no_model_calls(self):
        with tempfile.TemporaryDirectory() as tmp:
            dataset = Path(tmp) / "dev.jsonl"
            dataset.write_text(json.dumps(dict(medqa_record(), answer="wrong")) + "\n", encoding="utf-8")
            results_dir = Path(tmp) / "results_USMLE"
            with patch.object(config, "RESULTS_USMLE_DIR", str(results_dir)), \
                    patch("frameworks.tier1.zero_shot.call_llm") as mocked:
                with self.assertRaisesRegex(ValueError, "does not match answer_idx"):
                    runner.run_one_seed("zero_shot", "sglang-qwen3-32b", str(dataset), 42, workers=1)
            mocked.assert_not_called()
            self.assertFalse(results_dir.exists())

    def test_four_option_usmle_seeds_manifests_resume_and_readers_use_same_folder(self):
        with tempfile.TemporaryDirectory() as tmp:
            dataset = Path(tmp) / "dev.jsonl"
            record = medqa_record()
            record["options"].pop("E")
            record.update(answer_idx="D", answer="fourth")
            dataset.write_text(json.dumps(record) + "\n", encoding="utf-8")
            usmle_dir = Path(tmp) / "results_USMLE"
            vm14k_dir = Path(tmp) / "results"
            with patch.object(config, "RESULTS_USMLE_DIR", str(usmle_dir)), \
                    patch.object(config, "RESULTS_DIR", str(vm14k_dir)), \
                    patch("frameworks.tier1.zero_shot.call_llm", return_value="D") as mocked, \
                    contextlib.redirect_stdout(io.StringIO()):
                for seed in config.SEEDS:
                    runner.run_one_seed("zero_shot", "sglang-qwen3-32b", str(dataset), seed, workers=1)
                    runner.run_one_seed("zero_shot", "sglang-qwen3-32b", str(dataset), seed, workers=1)
                cell = load_cell("zero_shot", "sglang-qwen3-32b", "dev")
                comparison = load_comparison("zero_shot", "sglang-qwen3-32b", "dev")
            self.assertEqual(mocked.call_count, len(config.SEEDS))
            self.assertTrue(cell.reportable)
            self.assertEqual(cell.complete_seeds, sorted(config.SEEDS))
            self.assertEqual(set(comparison), set(config.SEEDS))
            self.assertTrue(all(r["correct"] for r in cell.flat()))
            self.assertEqual(len(list(usmle_dir.rglob("*.jsonl"))), 3)
            self.assertEqual(len(list(usmle_dir.rglob("*.meta.json"))), 3)
            self.assertFalse(vm14k_dir.exists())


if __name__ == "__main__":
    unittest.main()
