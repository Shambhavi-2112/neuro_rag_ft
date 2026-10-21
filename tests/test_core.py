import pytest
from rag_pipeline import chunk_text, build_prompt
from evaluation import recall_at_k, reciprocal_rank, load_jsonl_questions


def test_chunking_returns_content():
    chunks = chunk_text("alpha beta gamma " * 100, size=80, overlap=10)
    assert chunks
    assert all(c.strip() for c in chunks)


def test_invalid_overlap_raises():
    with pytest.raises(ValueError):
        chunk_text("hello", size=10, overlap=10)


def test_recall_at_k():
    assert recall_at_k({"a", "b"}, ["x", "a", "b"], 3) == 1.0


def test_reciprocal_rank():
    assert reciprocal_rank({"a"}, ["x", "a"]) == 0.5
    assert reciprocal_rank({"a"}, ["x"]) == 0.0


def test_jsonl_validation():
    rows = load_jsonl_questions('{"question":"What?","relevant_chunk_ids":["abc"]}')
    assert rows[0]["question"] == "What?"
    with pytest.raises(ValueError):
        load_jsonl_questions('{"question":"missing ids"}')


def test_prompt_citations():
    prompt = build_prompt("What?", [{
        "source": "paper.pdf", "page": 2, "text": "Evidence text", "id": "abc"
    }])
    assert "[S1]" in prompt
    assert "paper.pdf" in prompt
