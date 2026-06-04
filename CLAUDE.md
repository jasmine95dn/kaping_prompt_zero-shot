# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Workflow

Never edit directly on `main`. Always create a feature branch first (e.g. `git checkout -b feature/<short-name>`) before making changes.

## Commands

Install (Poetry, Python 3.11/3.12 only):

```sh
poetry install --with dev
```

Lint (ruff via pre-commit — same step CI runs):

```sh
pre-commit run --all-files
```

Tests:

```sh
# Default: unit + e2e, no real ML downloads (this is what CI runs)
poetry run pytest -m "not system and not slow" -v

# Run a single test
poetry run pytest tests/unit/test_qa_evaluate.py::TestEvaluate::test_exact_match -v

# Full suite (requires network + model downloads, very slow)
poetry run pytest -v
```

Run the pipeline:

```sh
poetry run python run.py --input <mintaka_dataset.json>
poetry run python run.py -h   # all flags
```

Docker entrypoint is `run.py`, so `docker run --rm kaping --input ...` works after `docker build -t kaping .`.

## Architecture

KAPING is a 3-stage retrieval-augmented prompt builder, then an HF inference call. Per-question flow lives in `kaping/model.py::pipeline()`:

1. **Entity extraction** (`kaping/entity_extractor.py`) — ReFInED links question entities to Wikidata.
2. **Entity verbalization** (`kaping/entity_verbalization.py`) — REBEL extracts relation triples from each entity's Wikipedia page.
3. **Entity injection** (`kaping/entity_injection.py`) — MPNet (`all-mpnet-base-v2`) embeds both question and triples, cosine-similarity picks top-k, and the triples are spliced into a leading prompt template. `--random` swaps top-k for random-k (baseline); `--no_knowledge` skips retrieval entirely (prompt-only baseline).

The resulting prompt goes to `qa/qa_inference.py`, which dispatches to a HuggingFace pipeline. **The QA model generates the answer from the full Question+Context prompt — it does not extract a span.** `--inference_task` (`text2text-generation` vs `text-generation`) must match the model architecture. `gpt2` requires `text-generation`; this is checked in `run.py` and exits early on mismatch.

Evaluation (`qa/qa_evaluate.py`) computes five metrics, all aggregated by `corpus_metrics(answers, predictions)` and printed by `run.py`:

| Metric | What it measures |
|--------|------------------|
| `containment` | Original loose substring check: `gold in predicted`. Case-sensitive, no normalization. |
| `exact_match` | SQuAD-style normalized equality (lowercase, strip `a`/`an`/`the`, strip punctuation, collapse whitespace). |
| `token_f1` | Token-level F1 over normalized text. Standard SQuAD-style QA F1. |
| `bleu_1` | Sentence-level BLEU-1 (clipped unigram precision × brevity penalty) over normalized text. |
| `rouge_l` | LCS-based ROUGE-L F-measure over normalized text. |

All metrics except `containment` apply `normalize_answer`, so a standalone "a"/"an"/"the" token gets stripped — write tests with non-article tokens.

### Non-obvious gotchas

- `pipeline()` instantiates fresh `RefinedEntityExtractor`, `RebelEntityVerbalizer`, and `MPNetEntityInjector` **per question** (`kaping/model.py:20-22`). For multi-question runs this reloads three heavy models each iteration. Don't refactor casually — but be aware before benchmarking.
- `arguments.py` defaults `--model_name` to `bert-large-uncased` with `--inference_task=text2text-generation`. BERT is not a seq2seq model; this combination is what the original reimplementation used, not a bug to "fix."
- Unit tests stub heavy ML deps at import time via `tests/unit/conftest.py` — it inserts `MagicMock()` into `sys.modules` for `sentence_transformers`, `transformers`, `refined`, and `sklearn.metrics.pairwise` *before* test modules import them. New unit tests that touch these libraries must either (a) accept the mock or (b) move to `tests/e2e/` or `tests/system/`.
- Pytest markers `slow` and `system` are deselected by default in CI. Mark any test needing real model downloads or full environment setup accordingly, or it will silently run (and likely fail) in CI.

## Dataset format

`qa/qa_preprocessing.py::load_dataset` expects Mintaka JSON: each record needs `question`, `questionEntity[].mention`, `answer`, `complexityType`. Loader returns a list of `Pair` objects; `Pair.pr_answer` is filled in by `run.py` after inference and written to the output CSV.
