"""
Evaluation metrics for KAPING QA outputs.

All per-example metrics take (gold_answer, predicted_answer) and return a score
in [0, 1] (float) or a boolean. `corpus_metrics` aggregates a parallel list of
gold/predicted answers into mean scores.

Normalization (`normalize_answer`) follows the SQuAD convention: lowercase,
strip articles, strip punctuation, collapse whitespace. It is applied by every
metric below *except* `evaluate`, which is preserved as the original loose
substring-containment check (case-sensitive, no normalization).
"""

import math
import re
import string
from collections import Counter


def normalize_answer(s: str) -> str:
    """SQuAD-style normalization: lowercase, drop articles + punctuation, collapse whitespace."""
    s = s.lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = "".join(ch for ch in s if ch not in string.punctuation)
    return " ".join(s.split())


def evaluate(answer: str, predicted: str) -> bool:
    """Loose substring containment (legacy metric — case-sensitive, no normalization)."""
    return answer in predicted


def exact_match(answer: str, predicted: str) -> bool:
    """Normalized exact match."""
    return normalize_answer(answer) == normalize_answer(predicted)


def token_f1(answer: str, predicted: str) -> float:
    """Token-level F1 over normalized text (SQuAD convention)."""
    gold_tokens = normalize_answer(answer).split()
    pred_tokens = normalize_answer(predicted).split()

    if not gold_tokens or not pred_tokens:
        # Both empty -> perfect match; otherwise no overlap is possible.
        return float(gold_tokens == pred_tokens)

    overlap = sum((Counter(gold_tokens) & Counter(pred_tokens)).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def bleu_1(answer: str, predicted: str) -> float:
    """Sentence-level BLEU-1 (clipped unigram precision with brevity penalty)."""
    gold_tokens = normalize_answer(answer).split()
    pred_tokens = normalize_answer(predicted).split()
    if not gold_tokens or not pred_tokens:
        return 0.0

    gold_counts = Counter(gold_tokens)
    pred_counts = Counter(pred_tokens)
    clipped = sum(min(c, gold_counts[t]) for t, c in pred_counts.items())
    precision = clipped / len(pred_tokens)

    if len(pred_tokens) >= len(gold_tokens):
        bp = 1.0
    else:
        bp = math.exp(1 - len(gold_tokens) / len(pred_tokens))

    return bp * precision


def rouge_l(answer: str, predicted: str) -> float:
    """ROUGE-L F-measure based on longest common subsequence of tokens."""
    gold_tokens = normalize_answer(answer).split()
    pred_tokens = normalize_answer(predicted).split()
    if not gold_tokens or not pred_tokens:
        return 0.0

    m, n = len(gold_tokens), len(pred_tokens)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m):
        for j in range(n):
            if gold_tokens[i] == pred_tokens[j]:
                dp[i + 1][j + 1] = dp[i][j] + 1
            else:
                dp[i + 1][j + 1] = max(dp[i][j + 1], dp[i + 1][j])
    lcs = dp[m][n]
    if lcs == 0:
        return 0.0
    precision = lcs / n
    recall = lcs / m
    return 2 * precision * recall / (precision + recall)


def accuracy(evaluated) -> float:
    """Mean of a list of booleans (or numeric scores in [0, 1])."""
    if not evaluated:
        raise ValueError("Cannot compute accuracy over an empty list")
    return sum(bool(x) if isinstance(x, bool) else x for x in evaluated) / len(
        evaluated
    )


def corpus_metrics(answers: list, predictions: list) -> dict:
    """Compute mean of every per-example metric across parallel answer/prediction lists."""
    if len(answers) != len(predictions):
        raise ValueError("answers and predictions must have the same length")
    if not answers:
        raise ValueError("Cannot compute metrics over an empty dataset")

    pairs = list(zip(answers, predictions))
    n = len(pairs)
    return {
        "containment": sum(evaluate(a, p) for a, p in pairs) / n,
        "exact_match": sum(exact_match(a, p) for a, p in pairs) / n,
        "token_f1": sum(token_f1(a, p) for a, p in pairs) / n,
        "bleu_1": sum(bleu_1(a, p) for a, p in pairs) / n,
        "rouge_l": sum(rouge_l(a, p) for a, p in pairs) / n,
    }
