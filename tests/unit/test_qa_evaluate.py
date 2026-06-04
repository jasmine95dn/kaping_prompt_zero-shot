import pytest
from qa.qa_evaluate import (
    accuracy,
    bleu_1,
    corpus_metrics,
    evaluate,
    exact_match,
    normalize_answer,
    rouge_l,
    token_f1,
)


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

    def test_numeric_scores(self):
        # also works for floats in [0, 1]
        assert pytest.approx(accuracy([1.0, 0.5, 0.0])) == 0.5

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            accuracy([])


class TestNormalizeAnswer:
    def test_lowercases(self):
        assert normalize_answer("PARIS") == "paris"

    def test_strips_articles(self):
        assert normalize_answer("the Eiffel Tower") == "eiffel tower"

    def test_strips_punctuation(self):
        assert normalize_answer("Paris, France!") == "paris france"

    def test_collapses_whitespace(self):
        assert normalize_answer("  Paris   France  ") == "paris france"

    def test_strips_a_an_the_only_as_words(self):
        # "Tha" should not lose its leading characters — only whole-word articles
        assert normalize_answer("Thames") == "thames"


class TestExactMatch:
    def test_case_insensitive(self):
        assert exact_match("Paris", "paris") is True

    def test_article_insensitive(self):
        assert exact_match("Eiffel Tower", "the Eiffel Tower") is True

    def test_punctuation_insensitive(self):
        assert exact_match("Paris", "Paris.") is True

    def test_substring_is_not_match(self):
        assert exact_match("Jane Austen", "The author is Jane Austen.") is False

    def test_mismatch(self):
        assert exact_match("Paris", "London") is False

    def test_both_empty(self):
        assert exact_match("", "") is True


class TestTokenF1:
    def test_perfect_match(self):
        assert token_f1("Jane Austen", "Jane Austen") == 1.0

    def test_no_overlap(self):
        assert token_f1("Jane Austen", "Charles Dickens") == 0.0

    def test_partial_overlap(self):
        # gold: {jane, austen} (2 tokens), pred: {jane, smith} (2 tokens), overlap=1
        # precision = 1/2, recall = 1/2, f1 = 0.5
        assert token_f1("Jane Austen", "Jane Smith") == pytest.approx(0.5)

    def test_pred_is_longer_sentence(self):
        # gold: {jane, austen}, pred normalized: {author, is, jane, austen} -> overlap=2
        # precision = 2/4 = 0.5, recall = 2/2 = 1.0, f1 = 2/3
        assert token_f1("Jane Austen", "The author is Jane Austen") == pytest.approx(
            2 / 3
        )

    def test_normalization_applied(self):
        assert token_f1("the Paris", "Paris.") == 1.0

    def test_empty_gold_and_pred(self):
        assert token_f1("", "") == 1.0

    def test_empty_pred_only(self):
        assert token_f1("Paris", "") == 0.0


class TestBleu1:
    def test_perfect_match(self):
        assert bleu_1("Jane Austen", "Jane Austen") == pytest.approx(1.0)

    def test_no_overlap(self):
        assert bleu_1("Jane Austen", "Charles Dickens") == 0.0

    def test_brevity_penalty_when_pred_shorter(self):
        # gold len 2, pred len 1, precision 1.0, BP = exp(1 - 2/1) = exp(-1)
        import math

        assert bleu_1("Jane Austen", "Jane") == pytest.approx(math.exp(-1))

    def test_no_bp_when_pred_at_least_as_long(self):
        # precision = 2/3, no BP since pred len (3) >= gold len (2)
        assert bleu_1("Jane Austen", "Jane Austen now") == pytest.approx(2 / 3)

    def test_clipped_precision(self):
        # gold has "jane" once; pred has "jane" 3 times — clip to 1, precision = 1/3
        # pred len 3 >= gold len 1, so BP = 1
        assert bleu_1("Jane", "Jane Jane Jane") == pytest.approx(1 / 3)

    def test_empty_inputs(self):
        assert bleu_1("", "anything") == 0.0
        assert bleu_1("anything", "") == 0.0


class TestRougeL:
    def test_perfect_match(self):
        assert rouge_l("Jane Austen", "Jane Austen") == pytest.approx(1.0)

    def test_no_overlap(self):
        assert rouge_l("Jane Austen", "Charles Dickens") == 0.0

    def test_lcs_preserves_order(self):
        # gold tokens: [x, y, z], pred tokens: [z, y, x]  -> LCS length = 1
        # precision = 1/3, recall = 1/3, f1 = 1/3
        assert rouge_l("x y z", "z y x") == pytest.approx(1 / 3)

    def test_subsequence_with_gap(self):
        # gold: [jane, austen], pred: [jane, marie, austen] -> LCS=2
        # precision = 2/3, recall = 1.0, f1 = 0.8
        assert rouge_l("Jane Austen", "Jane Marie Austen") == pytest.approx(0.8)

    def test_empty_inputs(self):
        assert rouge_l("", "anything") == 0.0
        assert rouge_l("anything", "") == 0.0


class TestCorpusMetrics:
    def test_all_correct(self):
        m = corpus_metrics(["Paris", "London"], ["Paris", "London"])
        assert m["exact_match"] == 1.0
        assert m["token_f1"] == 1.0
        assert m["bleu_1"] == pytest.approx(1.0)
        assert m["rouge_l"] == pytest.approx(1.0)
        assert m["containment"] == 1.0

    def test_mixed(self):
        m = corpus_metrics(
            ["Jane Austen", "Paris"],
            ["The author is Jane Austen.", "London"],
        )
        # containment: True (first contains "Jane Austen"), False -> 0.5
        assert m["containment"] == 0.5
        # exact_match: first is False (whole-string compare), second False -> 0.0
        assert m["exact_match"] == 0.0
        # token_f1 average of (2/3) and 0.0
        assert m["token_f1"] == pytest.approx((2 / 3) / 2)

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            corpus_metrics(["a"], ["a", "b"])

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            corpus_metrics([], [])
