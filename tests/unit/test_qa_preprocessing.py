import pytest
from qa.qa_preprocessing import Pair, load_dataset


class TestPair:
    def test_initialization(self):
        pair = Pair("Who wrote Hamlet?", ["Hamlet"], "Shakespeare", "simple")
        assert pair.question == "Who wrote Hamlet?"
        assert pair.entities == ["Hamlet"]
        assert pair.answer == "Shakespeare"
        assert pair.q_type == "simple"

    def test_pr_answer_defaults_to_none(self):
        pair = Pair("Q?", [], "A", "simple")
        assert pair.pr_answer is None

    def test_pr_answer_can_be_set(self):
        pair = Pair("Q?", [], "A", "simple")
        pair.pr_answer = "predicted"
        assert pair.pr_answer == "predicted"

    def test_multiple_entities(self):
        pair = Pair("Q?", ["entity1", "entity2", "entity3"], "A", "comparative")
        assert len(pair.entities) == 3


class TestLoadDataset:
    def test_loads_single_entry(self, sample_mintaka_file):
        pairs = load_dataset(sample_mintaka_file)
        assert len(pairs) == 2

    def test_correct_question(self, sample_mintaka_file):
        pairs = load_dataset(sample_mintaka_file)
        assert pairs[0].question == "Who is the author of Lady Susan?"

    def test_correct_answer(self, sample_mintaka_file):
        pairs = load_dataset(sample_mintaka_file)
        assert pairs[0].answer == "Jane Austen"

    def test_correct_entities(self, sample_mintaka_file):
        pairs = load_dataset(sample_mintaka_file)
        assert pairs[0].entities == ["Lady Susan"]

    def test_multiple_entities(self, sample_mintaka_file):
        pairs = load_dataset(sample_mintaka_file)
        assert pairs[1].entities == ["Black Eyed Peas", "The Beatles"]

    def test_correct_complexity_type(self, sample_mintaka_file):
        pairs = load_dataset(sample_mintaka_file)
        assert pairs[0].q_type == "simple"
        assert pairs[1].q_type == "comparative"

    def test_pr_answer_is_none_after_load(self, sample_mintaka_file):
        pairs = load_dataset(sample_mintaka_file)
        assert all(p.pr_answer is None for p in pairs)

    def test_returns_pairs_as_list(self, sample_mintaka_file):
        result = load_dataset(sample_mintaka_file)
        assert isinstance(result, list)
        assert all(isinstance(p, Pair) for p in result)

    def test_empty_dataset(self, tmp_path):
        path = tmp_path / "empty.json"
        path.write_text("[]")
        assert load_dataset(str(path)) == []

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            load_dataset("/nonexistent/path/data.json")
