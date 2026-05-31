"""
End-to-end tests for the KAPING pipeline.
ML components are mocked at the class level in kaping.model so that
the orchestration logic and data flow are tested without loading real models.
"""

import pytest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch


QUESTION = "Who is the author of Lady Susan?"
TRIPLES = [
    "(Lady Susan, author, Jane Austen)",
    "(Lady Susan, genre, epistolary novel)",
]
EXPECTED_PROMPT = (
    "Below are facts in the form of the triple meaningful to answer the questions "
    f"{', '.join(TRIPLES)} Question: {QUESTION} Answer: "
)
NO_KNOWLEDGE_PROMPT = f"Please answer this question Question: {QUESTION} Answer: "


def make_config(k=3, random=False, no_knowledge=False):
    return SimpleNamespace(k=k, random=random, no_knowledge=no_knowledge)


@pytest.fixture
def mock_extractor():
    instance = MagicMock()
    instance.return_value = [("Lady Susan", "Lady Susan")]
    return instance


@pytest.fixture
def mock_verbalizer():
    instance = MagicMock()
    instance.return_value = TRIPLES
    return instance


@pytest.fixture
def mock_injector():
    instance = MagicMock()
    instance.return_value = EXPECTED_PROMPT
    return instance


@pytest.fixture
def mock_injector_no_knowledge():
    instance = MagicMock()
    instance.return_value = NO_KNOWLEDGE_PROMPT
    return instance


class TestPipelineOrchestration:
    def test_pipeline_returns_string_prompt(
        self, mock_extractor, mock_verbalizer, mock_injector
    ):
        with (
            patch("kaping.model.RefinedEntityExtractor", return_value=mock_extractor),
            patch("kaping.model.RebelEntityVerbalizer", return_value=mock_verbalizer),
            patch("kaping.model.MPNetEntityInjector", return_value=mock_injector),
        ):
            from kaping.model import pipeline

            result = pipeline(make_config(), QUESTION)

        assert isinstance(result, str)

    def test_pipeline_no_knowledge_returns_base_prompt(
        self, mock_extractor, mock_verbalizer, mock_injector_no_knowledge
    ):
        with (
            patch("kaping.model.RefinedEntityExtractor", return_value=mock_extractor),
            patch("kaping.model.RebelEntityVerbalizer", return_value=mock_verbalizer),
            patch(
                "kaping.model.MPNetEntityInjector",
                return_value=mock_injector_no_knowledge,
            ),
        ):
            from kaping.model import pipeline

            result = pipeline(make_config(no_knowledge=True), QUESTION)

        assert "Please answer this question" in result
        assert QUESTION in result
        assert result.endswith("Answer: ")

    def test_pipeline_prompt_ends_with_answer_cue(
        self, mock_extractor, mock_verbalizer, mock_injector
    ):
        with (
            patch("kaping.model.RefinedEntityExtractor", return_value=mock_extractor),
            patch("kaping.model.RebelEntityVerbalizer", return_value=mock_verbalizer),
            patch("kaping.model.MPNetEntityInjector", return_value=mock_injector),
        ):
            from kaping.model import pipeline

            result = pipeline(make_config(), QUESTION)

        assert result.endswith("Answer: ")

    def test_pipeline_calls_extractor_with_question(
        self, mock_extractor, mock_verbalizer, mock_injector
    ):
        with (
            patch("kaping.model.RefinedEntityExtractor", return_value=mock_extractor),
            patch("kaping.model.RebelEntityVerbalizer", return_value=mock_verbalizer),
            patch("kaping.model.MPNetEntityInjector", return_value=mock_injector),
        ):
            from kaping.model import pipeline

            pipeline(make_config(), QUESTION)

        mock_extractor.assert_called_once_with(QUESTION)

    def test_pipeline_verbalizes_each_extracted_entity(
        self, mock_extractor, mock_verbalizer, mock_injector
    ):
        mock_extractor.return_value = [
            ("Lady Susan", "Lady Susan"),
            ("Jane Austen", "Jane Austen"),
        ]
        with (
            patch("kaping.model.RefinedEntityExtractor", return_value=mock_extractor),
            patch("kaping.model.RebelEntityVerbalizer", return_value=mock_verbalizer),
            patch("kaping.model.MPNetEntityInjector", return_value=mock_injector),
        ):
            from kaping.model import pipeline

            pipeline(make_config(), QUESTION)

        assert mock_verbalizer.call_count == 2
        mock_verbalizer.assert_any_call("Lady Susan", "Lady Susan")
        mock_verbalizer.assert_any_call("Jane Austen", "Jane Austen")

    def test_pipeline_passes_config_to_injector(
        self, mock_extractor, mock_verbalizer, mock_injector
    ):
        config = make_config(k=5, random=True)
        with (
            patch("kaping.model.RefinedEntityExtractor", return_value=mock_extractor),
            patch("kaping.model.RebelEntityVerbalizer", return_value=mock_verbalizer),
            patch("kaping.model.MPNetEntityInjector", return_value=mock_injector),
        ):
            from kaping.model import pipeline

            pipeline(config, QUESTION)

        mock_injector.assert_called_once_with(
            [QUESTION], TRIPLES, k=5, random=True, no_knowledge=False
        )

    def test_pipeline_no_entity_found_calls_injector_with_empty_triples(
        self, mock_verbalizer, mock_injector
    ):
        mock_extractor = MagicMock()
        mock_extractor.return_value = []  # no entities found
        with (
            patch("kaping.model.RefinedEntityExtractor", return_value=mock_extractor),
            patch("kaping.model.RebelEntityVerbalizer", return_value=mock_verbalizer),
            patch("kaping.model.MPNetEntityInjector", return_value=mock_injector),
        ):
            from kaping.model import pipeline

            pipeline(make_config(), QUESTION)

        mock_injector.assert_called_once_with(
            [QUESTION], [], k=3, random=False, no_knowledge=False
        )
        mock_verbalizer.assert_not_called()


class TestEndToEndDataFlow:
    def _mock_args(self, sample_mintaka_file, output_file, no_knowledge=True):
        return SimpleNamespace(
            input=sample_mintaka_file,
            output=output_file,
            k=3,
            random=False,
            no_knowledge=no_knowledge,
            inference_task="text2text-generation",
            model_name="t5-small",
            device=-1,
        )

    def test_full_run_produces_output_file(self, sample_mintaka_file, tmp_path):
        output_file = str(tmp_path / "results.csv")
        mock_args = self._mock_args(sample_mintaka_file, output_file)

        with (
            patch("run.k_parser", return_value=mock_args),
            patch(
                "kaping.model.RefinedEntityExtractor",
                return_value=MagicMock(return_value=[]),
            ),
            patch("kaping.model.RebelEntityVerbalizer", return_value=MagicMock()),
            patch(
                "kaping.model.MPNetEntityInjector",
                return_value=MagicMock(return_value=NO_KNOWLEDGE_PROMPT),
            ),
            patch(
                "qa.qa_inference.pipeline",
                return_value=MagicMock(
                    return_value=[{"generated_text": "Jane Austen"}]
                ),
            ),
        ):
            from run import main

            main()

        assert (tmp_path / "results.csv").exists()

    def test_output_file_contains_questions_and_answers(
        self, sample_mintaka_file, tmp_path
    ):
        output_file = str(tmp_path / "results.csv")
        mock_args = self._mock_args(sample_mintaka_file, output_file)

        with (
            patch("run.k_parser", return_value=mock_args),
            patch(
                "kaping.model.RefinedEntityExtractor",
                return_value=MagicMock(return_value=[]),
            ),
            patch("kaping.model.RebelEntityVerbalizer", return_value=MagicMock()),
            patch(
                "kaping.model.MPNetEntityInjector",
                return_value=MagicMock(return_value=NO_KNOWLEDGE_PROMPT),
            ),
            patch(
                "qa.qa_inference.pipeline",
                return_value=MagicMock(
                    return_value=[{"generated_text": "Jane Austen"}]
                ),
            ),
        ):
            from run import main

            main()

        content = (tmp_path / "results.csv").read_text()
        assert "Who is the author of Lady Susan?" in content
        assert "Jane Austen" in content
        assert "Predicted answer:" in content
