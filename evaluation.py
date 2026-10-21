import json


def load_jsonl_questions(text: str) -> list[dict]:
    rows = []
    for line_no, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON on line {line_no}: {exc}") from exc
        if "question" not in row or "relevant_chunk_ids" not in row:
            raise ValueError(f"Line {line_no} must contain question and relevant_chunk_ids")
        rows.append(row)
    return rows


def recall_at_k(relevant_ids, retrieved_ids, k: int) -> float:
    relevant = set(relevant_ids)
    if not relevant:
        return 0.0
    return len(relevant.intersection(retrieved_ids[:k])) / len(relevant)


def reciprocal_rank(relevant_ids, retrieved_ids) -> float:
    relevant = set(relevant_ids)
    for rank, ident in enumerate(retrieved_ids, 1):
        if ident in relevant:
            return 1.0 / rank
    return 0.0


def evaluate_retrieval(questions: list[dict], retriever, k: int = 4) -> dict:
    if not questions:
        raise ValueError("Evaluation set is empty")
    recalls, reciprocal_ranks = [], []
    for item in questions:
        hits = retriever(item["question"], k=k)
        ids = [h["id"] for h in hits]
        recalls.append(recall_at_k(item["relevant_chunk_ids"], ids, k))
        reciprocal_ranks.append(reciprocal_rank(item["relevant_chunk_ids"], ids))
    return {
        "queries": len(questions),
        f"recall@{k}": sum(recalls) / len(recalls),
        "mrr": sum(reciprocal_ranks) / len(reciprocal_ranks),
        "warning": "Metrics depend on manually annotated relevant chunk IDs; this evaluates retrieval, not answer factuality."
    }
