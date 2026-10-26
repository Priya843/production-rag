"""Qdrant collection with two named vectors per chunk: "dense" (meaning) and "sparse" (BM25 keywords)."""

from qdrant_client import QdrantClient, models

from rag.chunking import Chunk
from rag.config import settings

DENSE, SPARSE = "dense", "sparse"


def get_client() -> QdrantClient:
    if settings.qdrant_url:
        return QdrantClient(url=settings.qdrant_url)
    return QdrantClient(path=str(settings.qdrant_path))


def recreate_collection(client: QdrantClient, dim: int) -> None:
    if client.collection_exists(settings.collection):
        client.delete_collection(settings.collection)
    client.create_collection(
        settings.collection,
        vectors_config={DENSE: models.VectorParams(size=dim, distance=models.Distance.COSINE)},
        # BM25 needs inverse document frequency, which Qdrant computes over the collection.
        sparse_vectors_config={SPARSE: models.SparseVectorParams(modifier=models.Modifier.IDF)},
    )


def upsert(client: QdrantClient, chunks: list[Chunk], dense: list, sparse: list) -> None:
    points = [
        models.PointStruct(
            id=chunk.id,
            vector={
                DENSE: d.tolist(),
                SPARSE: models.SparseVector(indices=s.indices.tolist(), values=s.values.tolist()),
            },
            payload={"text": chunk.text, "source": chunk.source, "section": chunk.section, **chunk.metadata},
        )
        for chunk, d, s in zip(chunks, dense, sparse, strict=True)
    ]
    client.upsert(settings.collection, points=points)


def search(
    client: QdrantClient, dense_query: list[float], sparse_query: models.SparseVector, mode: str, limit: int
) -> list[models.ScoredPoint]:
    """mode: "dense", "sparse" or "hybrid" (both, merged with Reciprocal Rank Fusion)."""
    if mode == "dense":
        kwargs = {"query": dense_query, "using": DENSE}
    elif mode == "sparse":
        kwargs = {"query": sparse_query, "using": SPARSE}
    elif mode == "hybrid":
        # RRF scores each point by 1 / (k + rank) in each list, so the two searches'
        # raw scores (cosine vs BM25) never need to be put on the same scale.
        kwargs = {
            "prefetch": [
                models.Prefetch(query=dense_query, using=DENSE, limit=limit),
                models.Prefetch(query=sparse_query, using=SPARSE, limit=limit),
            ],
            "query": models.FusionQuery(fusion=models.Fusion.RRF),
        }
    else:
        raise ValueError(f"unknown search mode: {mode}")
    return client.query_points(settings.collection, limit=limit, with_payload=True, **kwargs).points
