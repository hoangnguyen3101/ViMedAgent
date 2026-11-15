"""Section 3.6 rule: the last explicit, non-negated statement of an answer decides the vote."""

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from infrastructure.vote_extraction import extract_vote

OPTS = ["Expectant management", "Histopathologic evaluation", "Elective surgery", "Immediate surgery"]


class LastExplicitStatementTests(unittest.TestCase):
    def test_declaration_forms(self):
        for raw, want in [("My final answer is: B\n\nBecause...", "B"),
                          ("The correct answer is C.", "C"),
                          ("Answer: **(D)**", "D"),
                          ("Đáp án đúng là A", "A"),
                          ("I maintain my answer: B", "B"),
                          ("Therefore, my final answer remains **D. Atonement**.", "D")]:
            with self.subTest(raw=raw):
                self.assertEqual(extract_vote(raw), want)

    def test_letter_followed_by_option_text(self):
        self.assertEqual(extract_vote("Based on this, the most likely pathogen is **B. Coxsackie A virus**."), "B")
        self.assertEqual(extract_vote("**Answer:**\nD. Traveler's diarrhea due to ETEC.\n\nThis is most likely."), "D")
        self.assertEqual(extract_vote("I agree that Option A is the correct answer."), "A")
        self.assertEqual(extract_vote("...making option D the most appropriate choice."), "D")

    def test_last_statement_wins(self):
        self.assertEqual(extract_vote("The answer is A.\nThe correct answer is B."), "B")
        self.assertEqual(extract_vote("A\nThe correct answer is B."), "B")
        self.assertEqual(extract_vote("Final answer: A\nCorrection. Final answer is D."), "D")

    def test_negated_statement_is_skipped(self):
        self.assertEqual(extract_vote("The answer is B. Therefore, option A is incorrect."), "B")
        self.assertEqual(extract_vote("The correct answer is B.\nOption C is not the correct answer."), "B")

    def test_option_list_is_not_an_answer(self):
        listing = "Evaluation of each option:\nA. Foo: no.\nB. Bar: yes.\nC. Baz: no.\nD. Qux: no."
        self.assertIsNone(extract_vote(listing))
        self.assertEqual(extract_vote(listing + "\n\n**Answer:**\nB. Bar\n\nThis option fits."), "B")

    def test_article_abbreviations_and_pronoun_are_not_labels(self):
        for raw in ("The correct answer is a beta-blocker.",
                    "The patient is a 62-year-old man.",
                    "Vitamin D is discussed here."):
            with self.subTest(raw=raw):
                self.assertIsNone(extract_vote(raw))
        self.assertEqual(extract_vote("Answer: B. Thyroid peroxidase oxidises I- to I2."), "B")
        self.assertEqual(extract_vote("The answer is B because C. difficile is excluded."), "B")
        self.assertEqual(extract_vote("C. difficile colitis is discussed.\n\nAnswer: A"), "A")

    def test_unresolved_or_out_of_range_is_missing(self):
        for raw in ("Final answer: A or B", "Final answer: A/B", "A\nFinal answer: E",
                    "The answer is F.\nOption D is wrong."):
            with self.subTest(raw=raw):
                self.assertIsNone(extract_vote(raw))
        self.assertEqual(extract_vote("Final answer: E", n_options=5), "E")

    def test_mention_of_an_option_inside_reasoning_is_not_a_statement(self):
        self.assertEqual(extract_vote("Assessing depression (option D) should be considered later.\n"
                                      "My final answer is: C"), "C")
        self.assertIsNone(extract_vote("Assessing depression (option D) should be considered."))

    def test_thinking_blocks_are_ignored(self):
        self.assertEqual(extract_vote("<think>The answer is A.</think>\nFinal answer: B"), "B")
        self.assertIsNone(extract_vote("<think>Final answer: A"))

    def test_option_text_fallback_only_when_options_given(self):
        raw = "The evidence points away from surgery. Expectant management is the most appropriate step."
        self.assertIsNone(extract_vote(raw))
        self.assertEqual(extract_vote(raw, 4, OPTS), "A")
        # two options named in the closing sentences: no vote
        self.assertIsNone(extract_vote("Expectant management or elective surgery.", 4, OPTS))
        # a letter statement always beats the text fallback
        self.assertEqual(extract_vote("Answer: C\nExpectant management is discussed.", 4, OPTS), "C")

class VietnameseTests(unittest.TestCase):
    """VM14K outputs: the rule must hold for Vietnamese cue words too."""

    def test_vietnamese_declarations(self):
        for raw, want in [("Câu trả lời đúng là B. Đau giảm khi nâng tinh hoàn.", "B"),
                          ("**Lựa chọn đúng đắn:**\nC. Xét nghiệm hCG rồi tư vấn", "C"),
                          ("Sau khi đánh giá từng lựa chọn, tôi chọn **A. Yếu hơn người lớn**.", "A"),
                          ("**Kết luận:**\nLựa chọn B là đúng. Giảm nồng độ hormon.", "B"),
                          ("**3. Answer:**\n\n✅ **A. Là các dải mờ**", "A"),
                          ("Do đó, **C là đáp án chính xác nhất** theo đánh giá.", "C"),
                          ("Do đó, **D. Tất cả đúng** là câu trả lời chính xác.", "D")]:
            with self.subTest(raw=raw):
                self.assertEqual(extract_vote(raw), want)

    def test_hedge_after_the_answer_is_not_a_second_answer(self):
        raw = "Tôi chọn đáp án D. Tuy nhiên, đáp án B và C cũng có thể đúng, tùy tình trạng bệnh nhân."
        self.assertEqual(extract_vote(raw), "D")

    def test_verdict_does_not_reach_back_across_sentences(self):
        raw = "Cách ly (D) chủ yếu kiểm soát dịch bệnh, không phải điều trị. Do đó, **A là lựa chọn đúng** nhất."
        self.assertEqual(extract_vote(raw), "A")

    def test_refusal_to_choose_is_missing(self):
        self.assertIsNone(extract_vote("Vì không có option nào phù hợp hoàn toàn, tôi sẽ không chọn một option cụ thể."))


if __name__ == "__main__":
    unittest.main()
