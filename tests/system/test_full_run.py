"""
System tests — require real ML model downloads and a live internet connection.
Skipped by default. Run with: pytest -m system
"""

import json
import pytest
from types import SimpleNamespace


pytestmark = pytest.mark.system


@pytest.fixture
def system_mintaka_file(tmp_path):
    data = [
        {
            "question": "Who is the author of Lady Susan?",
            "questionEntity": [{"mention": "Lady Susan"}],
            "answer": "Jane Austen",
            "complexityType": "simple",
        }
    ]
    path = tmp_path / "system_sample.json"
    path.write_text(json.dumps(data))
    return str(path)


@pytest.mark.slow
def test_entity_extractor_finds_entity():
    """RefinedEntityExtractor should detect 'Lady Susan' in the question."""
    from kaping.entity_extractor import RefinedEntityExtractor

    extractor = RefinedEntityExtractor(device=-1)
    result = extractor("Who is the author of Lady Susan?")

    assert isinstance(result, list)
    assert len(result) > 0
    texts = [text for text, _ in result]
    assert any("Lady Susan" in t for t in texts)


@pytest.mark.slow
def test_entity_extractor_question_ending_with_no_space():
    """Ensure extractor handles '?' attached to last token without a space."""
    from kaping.entity_extractor import RefinedEntityExtractor

    extractor = RefinedEntityExtractor(device=-1)
    result = extractor("Who wrote Hamlet?")
    assert isinstance(result, list)


@pytest.mark.slow
def test_entity_verbalizer_returns_triples():
    """RebelEntityVerbalizer should return a list of triple strings."""
    from kaping.entity_verbalization import RebelEntityVerbalizer

    verbalizer = RebelEntityVerbalizer(device=-1)
    result = verbalizer("Lady Susan", "Lady Susan")

    assert isinstance(result, list)
    # Each triple is formatted as "(head, relation, tail)"
    for triple in result:
        assert triple.startswith("(") and triple.endswith(")")


@pytest.mark.slow
def test_injector_produces_prompt_with_real_embeddings():
    """MPNetEntityInjector should embed triples and build a ranked prompt."""
    from kaping.entity_injection import MPNetEntityInjector

    injector = MPNetEntityInjector(device=-1)
    triples = [
        "(Lady Susan, author, Jane Austen)",
        "(Lady Susan, genre, epistolary novel)",
        "(Jane Austen, nationality, British)",
    ]
    result = injector(["Who is the author of Lady Susan?"], triples, k=2)

    assert isinstance(result, str)
    assert "Answer: " in result


@pytest.mark.slow
def test_full_pipeline_with_knowledge(system_mintaka_file, tmp_path):
    """Full run.py pipeline with real models, no_knowledge=False."""
    from unittest.mock import patch

    output_file = str(tmp_path / "system_output.csv")
    mock_args = SimpleNamespace(
        input=system_mintaka_file,
        output=output_file,
        k=3,
        random=False,
        no_knowledge=False,
        inference_task="text2text-generation",
        model_name="t5-small",
        device=-1,
    )

    with patch("arguments.k_parser", return_value=mock_args):
        from run import main

        main()

    content = (tmp_path / "system_output.csv").read_text()
    assert "Who is the author of Lady Susan?" in content
    assert "Predicted answer:" in content


@pytest.mark.slow
def test_full_pipeline_no_knowledge(system_mintaka_file, tmp_path):
    """Full run.py pipeline using no_knowledge baseline."""
    from unittest.mock import patch

    output_file = str(tmp_path / "system_output_nk.csv")
    mock_args = SimpleNamespace(
        input=system_mintaka_file,
        output=output_file,
        k=3,
        random=False,
        no_knowledge=True,
        inference_task="text2text-generation",
        model_name="t5-small",
        device=-1,
    )

    with patch("arguments.k_parser", return_value=mock_args):
        from run import main

        main()

    content = (tmp_path / "system_output_nk.csv").read_text()
    assert "Who is the author of Lady Susan?" in content


@pytest.mark.slow
def test_full_pipeline_random_knowledge(system_mintaka_file, tmp_path):
    """Full run.py pipeline using random knowledge baseline."""
    from unittest.mock import patch

    output_file = str(tmp_path / "system_output_random.csv")
    mock_args = SimpleNamespace(
        input=system_mintaka_file,
        output=output_file,
        k=3,
        random=True,
        no_knowledge=False,
        inference_task="text2text-generation",
        model_name="t5-small",
        device=-1,
    )

    with patch("arguments.k_parser", return_value=mock_args):
        from run import main

        main()

    content = (tmp_path / "system_output_random.csv").read_text()
    assert "Who is the author of Lady Susan?" in content
