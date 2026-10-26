"""Retrieval metrics that need no LLM: did the right file show up, and how high?"""


def first_relevant_rank(retrieved: list[str], relevant: set[str]) -> int | None:
    """1-based rank of the first retrieved source that is relevant, or None."""
    for rank, source in enumerate(retrieved, start=1):
        if source in relevant:
            return rank
    return None


def hit_rate(ranks: list[int | None], k: int) -> float:
    """Share of questions with a relevant source in the top k."""
    return sum(1 for r in ranks if r is not None and r <= k) / len(ranks)


def mrr(ranks: list[int | None]) -> float:
    """Mean reciprocal rank: 1.0 if the right source is always first, 0.5 if always second."""
    return sum(1 / r for r in ranks if r is not None) / len(ranks)
