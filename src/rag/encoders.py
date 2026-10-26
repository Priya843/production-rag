"""The three local models: dense embeddings, sparse BM25 vectors, and a cross-encoder reranker.

All run on CPU through ONNX Runtime (fastembed); they download once into .cache/.
"""

from functools import lru_cache

from fastembed import SparseTextEmbedding, TextEmbedding
from fastembed.rerank.cross_encoder import TextCrossEncoder

from rag.config import settings


@lru_cache
def dense_encoder() -> TextEmbedding:
    return TextEmbedding(settings.dense_model, cache_dir=str(settings.model_cache))


@lru_cache
def sparse_encoder() -> SparseTextEmbedding:
    return SparseTextEmbedding(settings.sparse_model, cache_dir=str(settings.model_cache))


@lru_cache
def reranker() -> TextCrossEncoder:
    return TextCrossEncoder(settings.rerank_model, cache_dir=str(settings.model_cache))
