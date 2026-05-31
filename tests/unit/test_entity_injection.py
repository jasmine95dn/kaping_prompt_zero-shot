import numpy as np
import pytest
from unittest.mock import MagicMock, patch
from kaping.entity_injection import MPNetEntityInjector


@pytest.fixture
def injector():
    with patch("kaping.entity_injection.SentenceTransformer"):
        return MPNetEntityInjector()


class TestInjection:
    def test_no_knowledge_contains_base_prompt(self, injector):
        result = injector.injection(
            "Who is the author of Lady Susan?", no_knowledge=True
        )
        assert MPNetEntityInjector.no_knowledge_prompt in result

    def test_no_knowledge_contains_question(self, injector):
        question = "Who is the author of Lady Susan?"
        result = injector.injection(question, no_knowledge=True)
        assert question in result

    def test_no_knowledge_ends_with_answer_cue(self, injector):
        result = injector.injection("Q?", no_knowledge=True)
        assert result.endswith("Answer: ")

    def test_with_triples_contains_leading_prompt(self, injector, sample_triples):
        result = injector.injection("Q?", triples=sample_triples)
        assert MPNetEntityInjector.leading_prompt in result

    def test_with_triples_contains_all_triples(self, injector, sample_triples):
        result = injector.injection("Q?", triples=sample_triples)
        for triple in sample_triples:
            assert triple in result

    def test_with_triples_contains_question(self, injector, sample_triples):
        question = "Who is the author of Lady Susan?"
        result = injector.injection(question, triples=sample_triples)
        assert question in result

    def test_with_triples_ends_with_answer_cue(self, injector, sample_triples):
        result = injector.injection("Q?", triples=sample_triples)
        assert result.endswith("Answer: ")


class TestTopKTripleExtractor:
    def test_returns_k_triples(self, injector):
        question = np.random.rand(1, 16)
        triples_emb = np.random.rand(10, 16)
        triples = [f"triple_{i}" for i in range(10)]
        fake_sim = np.random.rand(1, 10)
        with patch("kaping.entity_injection.cosine_similarity", return_value=fake_sim):
            result = injector.top_k_triple_extractor(
                question, triples_emb, triples, k=3
            )
        assert len(result) == 3

    def test_clamps_k_when_fewer_triples_available(self, injector):
        question = np.random.rand(1, 16)
        triples_emb = np.random.rand(2, 16)
        triples = ["t1", "t2"]
        fake_sim = np.random.rand(1, 2)
        with patch("kaping.entity_injection.cosine_similarity", return_value=fake_sim):
            result = injector.top_k_triple_extractor(
                question, triples_emb, triples, k=10
            )
        assert len(result) == 2

    def test_returns_string_triples_not_embeddings(self, injector):
        question = np.random.rand(1, 16)
        triples_emb = np.random.rand(5, 16)
        triples = [f"triple_{i}" for i in range(5)]
        fake_sim = np.random.rand(1, 5)
        with patch("kaping.entity_injection.cosine_similarity", return_value=fake_sim):
            result = injector.top_k_triple_extractor(
                question, triples_emb, triples, k=3
            )
        assert all(isinstance(t, str) for t in result)
        assert all(t in triples for t in result)

    def test_random_mode_returns_k_items(self, injector):
        question = np.random.rand(1, 16)
        triples_emb = np.random.rand(5, 16)
        triples = ["t1", "t2", "t3", "t4", "t5"]
        result = injector.top_k_triple_extractor(
            question, triples_emb, triples, k=3, random=True
        )
        assert len(result) == 3

    def test_random_mode_returns_from_input(self, injector):
        question = np.random.rand(1, 16)
        triples_emb = np.random.rand(5, 16)
        triples = ["t1", "t2", "t3", "t4", "t5"]
        result = injector.top_k_triple_extractor(
            question, triples_emb, triples, k=3, random=True
        )
        assert all(t in triples for t in result)

    def test_most_similar_triple_is_selected(self, injector):
        question = np.array([[1.0, 0.0, 0.0]])
        triples_emb = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        triples = ["most_similar", "second", "third"]
        fake_sim = np.array([[0.9, 0.3, 0.1]])
        with patch("kaping.entity_injection.cosine_similarity", return_value=fake_sim):
            result = injector.top_k_triple_extractor(
                question, triples_emb, triples, k=1
            )
        assert result[0] == "most_similar"


class TestCall:
    def test_no_knowledge_bypasses_embedding(self, injector):
        injector.sentence_embedding = MagicMock()
        result = injector(["Who is the author?"], [], no_knowledge=True)
        injector.sentence_embedding.assert_not_called()
        assert MPNetEntityInjector.no_knowledge_prompt in result

    def test_question_must_be_list(self, injector):
        with pytest.raises(AssertionError):
            injector("not a list", [], no_knowledge=True)

    def test_triples_must_be_list(self, injector):
        with pytest.raises(AssertionError):
            injector(["Q?"], "not a list")

    def test_calls_sentence_embedding_for_question_and_triples(
        self, injector, sample_triples
    ):
        emb = np.random.rand(len(sample_triples), 16)
        injector.sentence_embedding = MagicMock(return_value=emb)
        injector.top_k_triple_extractor = MagicMock(return_value=sample_triples[:2])
        injector(["Who is the author?"], sample_triples)
        assert injector.sentence_embedding.call_count == 2
