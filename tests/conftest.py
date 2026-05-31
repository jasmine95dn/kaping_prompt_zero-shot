import json
import pytest


SAMPLE_MINTAKA = [
    {
        "question": "Who is the author of Lady Susan?",
        "questionEntity": [{"mention": "Lady Susan"}],
        "answer": "Jane Austen",
        "complexityType": "simple",
    },
    {
        "question": "Which band has more members, Black Eyed Peas or The Beatles?",
        "questionEntity": [{"mention": "Black Eyed Peas"}, {"mention": "The Beatles"}],
        "answer": "Black Eyed Peas",
        "complexityType": "comparative",
    },
]

SAMPLE_TRIPLES = [
    "(Lady Susan, author, Jane Austen)",
    "(Lady Susan, genre, epistolary novel)",
    "(Jane Austen, nationality, British)",
    "(Jane Austen, born in, Steventon)",
    "(Lady Susan, published, 1871)",
]


@pytest.fixture
def sample_mintaka_file(tmp_path):
    path = tmp_path / "mintaka_sample.json"
    path.write_text(json.dumps(SAMPLE_MINTAKA))
    return str(path)


@pytest.fixture
def sample_triples():
    return SAMPLE_TRIPLES


@pytest.fixture
def sample_question():
    return "Who is the author of Lady Susan?"
