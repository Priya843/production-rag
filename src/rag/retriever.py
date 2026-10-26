"""Retrieval: hybrid search for recall, then a cross-encoder rerank for precision."""

from dataclasses import dataclass, field

from qdrant_client import models

from rag import encoders, store
from rag.config import settings


@dataclass
class Hit:
    text: str
    source: str
    section: str
    score: float
    metadata: dict = field(default_factory=dict)


class Retriever:
    def __init__(self) -> None:
        self.client = store.get_client()

    def search(self, query: str, mode: str = "hybrid", limit: int | None = None) -> list[Hit]:
        dense = next(encoders.dense_encoder().query_embed(query)).tolist()
        sparse = next(encoders.sparse_encoder().query_embed(query))
        sparse = models.SparseVector(indices=sparse.indices.tolist(), values=sparse.values.tolist())
        points = store.search(self.client, dense, sparse, mode, limit or settings.candidates)
        return [
            Hit(p.payload["text"], p.payload["source"], p.payload["section"], p.score,
                {k: v for k, v in p.payload.items() if k not in ("text", "source", "section")})
            for p in points
        ]

    def rerank(self, query: str, hits: list[Hit]) -> list[Hit]:
        # A cross-encoder reads the query and the chunk together, so it judges relevance
        # far better than comparing two separately computed vectors, but it is too slow
        # to run over the whole corpus. That is why it only sees the top candidates.
        scores = encoders.reranker().rerank(query, [h.text for h in hits], batch_size=settings.encoder_batch)
        for hit, score in zip(hits, scores, strict=True):
            hit.score = float(score)
        return sorted(hits, key=lambda h: h.score, reverse=True)

    def retrieve(self, query: str, top_k: int | None = None, rerank: bool = True) -> list[Hit]:
        hits = self.search(query, "hybrid", settings.candidates)
        if rerank and hits:
            hits = self.rerank(query, hits)
        return hits[: top_k or settings.top_k]

    def close(self) -> None:
        # Release embedded Qdrant's folder lock now; doing it at interpreter exit errors on Windows.
        self.client.close()
