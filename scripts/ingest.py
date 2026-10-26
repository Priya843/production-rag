"""Load -> chunk -> embed (dense + sparse) -> write to Qdrant. Rebuilds the collection each run."""

import time

from rag import encoders, store
from rag.chunking import chunk_documents
from rag.config import settings
from rag.loaders import load_documents

BATCH = 256  # chunks per Qdrant upsert


def main() -> None:
    start = time.perf_counter()
    docs = load_documents(settings.data_dir)
    if not docs:
        raise SystemExit(f"No .md/.txt/.pdf files in {settings.data_dir}. Run scripts/fetch_docs.py first.")
    chunks = chunk_documents(docs, settings.chunk_size, settings.chunk_overlap)
    sources = len({c.source for c in chunks})
    print(f"{len(docs)} documents from {sources} files -> {len(chunks)} chunks")

    client = store.get_client()
    dense_model, sparse_model = encoders.dense_encoder(), encoders.sparse_encoder()
    for i in range(0, len(chunks), BATCH):
        batch = chunks[i : i + BATCH]
        texts = [c.text for c in batch]
        dense = list(dense_model.embed(texts, batch_size=settings.encoder_batch))
        sparse = list(sparse_model.embed(texts, batch_size=settings.encoder_batch))
        if i == 0:
            store.recreate_collection(client, dim=len(dense[0]))
        store.upsert(client, batch, dense, sparse)
        print(f"  indexed {min(i + BATCH, len(chunks))}/{len(chunks)}", flush=True)

    client.close()
    print(f"Done in {time.perf_counter() - start:.0f}s -> collection '{settings.collection}'")


if __name__ == "__main__":
    main()
