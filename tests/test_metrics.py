from rag.metrics import first_relevant_rank, hit_rate, mrr


def test_first_relevant_rank():
    assert first_relevant_rank(["a.md", "b.md", "c.md"], {"c.md"}) == 3
    assert first_relevant_rank(["a.md"], {"z.md"}) is None


def test_hit_rate_and_mrr():
    ranks = [1, 2, None, 5]
    assert hit_rate(ranks, 1) == 0.25
    assert hit_rate(ranks, 5) == 0.75
    assert mrr(ranks) == (1 + 0.5 + 0.2) / 4
