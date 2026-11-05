"""Parser regressions: declared answers must beat incidental option mentions."""

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from infrastructure.llm_client import parse_answer


class AnswerParsingTests(unittest.TestCase):
    def test_leading_answer_is_not_replaced_by_discussion_of_option_d(self):
        raw = ("A\n\nOption A (Atenolol) is appropriate.\n"
               "Option B is a diuretic.\nOption C is another diuretic.\n"
               "Option D (Nitroglycerin) treats acute attacks.\n"
               "The correct answer is Atenolol.")
        self.assertEqual(parse_answer(raw), "A")

    def test_explicit_answer_is_not_replaced_by_other_options(self):
        for marker in ("The correct answer is", "The answer is", "Answer:", "Đáp án đúng là"):
            with self.subTest(marker=marker):
                self.assertEqual(parse_answer(f"{marker} B.\nOption C is wrong. Option D is wrong."), "B")

    def test_final_answer_overrides_provisional_answer_and_discussion(self):
        self.assertEqual(parse_answer("A\nThe answer is B.\nFinal answer: **(C)**\nOption D is incorrect."), "C")
        self.assertEqual(parse_answer("Final answer: A\nCorrection. Final answer is D."), "D")
        self.assertEqual(parse_answer("Final answer would be C."), "C")

    def test_contradictory_declarations_are_ambiguous(self):
        examples = ["The answer is A.\nThe correct answer is B.",
                    "A\nThe correct answer is B.",
                    "The correct answer is B.\nTherefore, option A (Hypoperfusion) should be treated first."]
        for raw in examples:
            with self.subTest(raw=raw):
                self.assertIsNone(parse_answer(raw))

    def test_repeated_consistent_declarations_are_accepted(self):
        self.assertEqual(parse_answer("B\nThe correct answer is B.\nTherefore, option B is appropriate."), "B")

    def test_option_discussion_alone_is_not_an_answer(self):
        examples = ["Option A is plausible. Option D is not appropriate.",
                    "A. first choice\nB. second choice\nC. third choice\nD. fourth choice",
                    "Vitamin D is discussed here.",
                    "The correct answer is a beta-blocker.",
                    "The patient is a 62-year-old man.",
                    "Candidates are (A) and (B)."]
        for raw in examples:
            with self.subTest(raw=raw):
                self.assertIsNone(parse_answer(raw))

    def test_unresolved_multiple_answers_are_not_scored_as_one(self):
        for raw in ("Final answer: A or B", "Final answer: A/B", "Answer: A and B"):
            with self.subTest(raw=raw):
                self.assertIsNone(parse_answer(raw))

    def test_out_of_range_final_answer_does_not_fall_back_to_discussion(self):
        self.assertIsNone(parse_answer("A\nFinal answer: E"))
        self.assertEqual(parse_answer("A\nFinal answer: E", n_options=5), "E")
        self.assertIsNone(parse_answer("The answer is F.\nOption D is wrong.", n_options=5))

    def test_thinking_blocks_are_not_used_as_answers(self):
        self.assertEqual(parse_answer("<think>The answer is A.</think>\nB"), "B")
        self.assertIsNone(parse_answer("<think>Final answer: A"))
        self.assertEqual(parse_answer("<THINK>Final answer: A</THINK>\nC"), "C")

    def test_negated_concluding_option_does_not_override_real_answer(self):
        self.assertEqual(parse_answer("The answer is B. Therefore, option A is incorrect."), "B")


if __name__ == "__main__":
    unittest.main()
