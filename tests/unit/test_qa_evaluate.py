import pytest
from qa.qa_evaluate import accuracy, evaluate


class TestEvaluate:
    def test_answer_contained_in_prediction(self):
        assert evaluate("Jane Austen", "The author is Jane Austen.") is True

    def test_answer_not_in_prediction(self):
        assert evaluate("Jane Austen", "The author is Charles Dickens.") is False

    def test_exact_match(self):
        assert evaluate("Paris", "Paris") is True

    def test_case_sensitive_mismatch(self):
        assert evaluate("paris", "Paris") is False

    def test_empty_answer(self):
        assert evaluate("", "Any prediction contains empty string") is True

    def test_empty_prediction(self):
        assert evaluate("Jane Austen", "") is False

    def test_partial_word_match(self):
        # "Aus" is in "Jane Austen" — substring check, not word boundary
        assert evaluate("Aus", "Jane Austen") is True


class TestAccuracy:
    def test_all_correct(self):
        assert accuracy([True, True, True]) == 1.0

    def test_all_wrong(self):
        assert accuracy([False, False, False]) == 0.0

    def test_half_correct(self):
        assert accuracy([True, False, True, False]) == 0.5

    def test_single_correct(self):
        assert accuracy([True]) == 1.0

    def test_single_wrong(self):
        assert accuracy([False]) == 0.0

    def test_one_out_of_three(self):
        assert pytest.approx(accuracy([True, False, False])) == 1 / 3
