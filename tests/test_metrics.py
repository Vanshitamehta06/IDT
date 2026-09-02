from eval.metrics import correctness, hallucination_rate, relevance, retrieval_quality


def test_correctness_keywords() -> None:
    item = {"gold_keywords": ["rag", "retrieval"], "must_cite": False}
    assert correctness("RAG uses retrieval", item) == 1.0


def test_relevance_jaccard() -> None:
    assert relevance("dense retrieval BM25", "dense retrieval BM25") == 1.0


def test_retrieval_files() -> None:
    item = {"expected_files": ["services/orchestrator.py"]}
    retrieved = [{"filename": "services/orchestrator.py", "text": "run()"}]
    assert retrieval_quality(item, retrieved) == 1.0


def test_hallucination_refuse() -> None:
    assert hallucination_rate("The evidence is insufficient to answer.", [{"text": "unrelated"}]) == 0.0
