import pandas as pd

from src.data import question_type, sample_questions
from src.generator import NOT_FOUND, build_messages
from src.metrics import abstained, citation_stats, cited_numbers, sentences, strip_citations, yes_no


def test_question_type():
    assert question_type("Yes, papilin is secreted") == "yes"
    assert question_type("No. It is not.") == "no"
    assert question_type("Notch signalling is involved") == "open"  # "No" must be a whole word


def test_yes_no_parsing():
    assert yes_no("Yes, it is [1].") == "yes"
    assert yes_no("**No** - the sources show...") == "no"
    assert yes_no("It depends") is None


def test_citations():
    text = "A is true [1]. B is also true [2, 3]. C [7]."
    assert cited_numbers(text) == {1, 2, 3, 7}
    assert citation_stats(text, n_sources=3) == {"has_citation": True, "invalid_citation": True}
    assert strip_citations(text) == "A is true. B is also true. C."


def test_sentences_skip_fragments():
    assert sentences("Yes [1]. The protein is secreted into the matrix [2].") == [
        "The protein is secreted into the matrix."]


def test_abstention():
    assert abstained(NOT_FOUND)
    assert abstained("I'm sorry. The sources do not contain the answer")
    assert not abstained("Yes, it does [1].")


def test_prompt_numbers_sources_and_truncates():
    long = {"text": "word " * 1000}
    msgs = build_messages("Q?", [{"text": "first"}, long])
    user = msgs[1]["content"]
    assert "[1] first" in user and "[2] word" in user
    assert user.count("word") <= 300
    assert len(build_messages("Q?", None)) == 2  # closed-book prompt has no sources


def test_sample_is_balanced():
    q = pd.DataFrame({"type": ["yes"] * 300 + ["no"] * 100 + ["open"] * 400})
    s = sample_questions(q, n=200)
    assert s["type"].value_counts().to_dict() == {"open": 100, "yes": 50, "no": 50}
