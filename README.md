# production-rag

A RAG system built the way production teams build them: hybrid search, a cross-encoder reranker, a real vector database (Qdrant), and measured quality, both for retrieval and with Ragas for answers.

The sample corpus is the Ultralytics YOLO docs (export, TensorRT, Jetson, tracking, ...), so the questions are ones a computer-vision engineer actually asks. Point it at any folder of `.md` / `.txt` / `.pdf` files to use your own documents.

```
                ┌─────────── ingest (scripts/ingest.py) ───────────┐
 data/raw/*.md ─► clean + split on headings ─► dense vectors (bge-small)  ─┐
                  (heading path kept in chunk)  sparse vectors (BM25)     ─┴─► Qdrant
                └──────────────────────────────────────────────────┘

                ┌──────────── query (scripts/ask.py) ──────────────┐
 question ─► dense search ─┐
            BM25 search ───┴─► RRF fusion (top 30) ─► cross-encoder rerank (top 5) ─► LLM answer with [n] citations
                └──────────────────────────────────────────────────┘
```

## Quick start (Windows)

```bash
py -3.12 -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev]"

python scripts/fetch_docs.py        # sample corpus -> data/raw/ultralytics/
python scripts/ingest.py            # chunk + embed + index (local CPU, no API key, ~5 min)
python scripts/ask.py "How do I export a YOLO model to TensorRT with INT8?" --no-llm
python scripts/eval_retrieval.py    # dense vs BM25 vs hybrid vs hybrid+rerank
```

Answers and the Ragas eval need an LLM. Set `OPENAI_API_KEY` or `ANTHROPIC_API_KEY` (or put `LLM_MODEL=...` in `.env`, see `.env.example`):

```bash
python scripts/ask.py "How do I export a YOLO model to TensorRT with INT8?"
python scripts/eval_ragas.py --limit 3   # cheap smoke test first
python scripts/eval_ragas.py
```

Qdrant runs embedded (a folder, `qdrant_data/`) by default. To use the server and its dashboard: `docker compose up -d`, set `QDRANT_URL=http://localhost:6333`, re-run ingest, and open http://localhost:6333/dashboard.

## What each file teaches

| File | Concept |
|---|---|
| `src/rag/chunking.py` | Structure-aware chunking: split on headings, never inside code blocks, keep the heading path in every chunk, overlap between chunks |
| `src/rag/store.py` | One Qdrant collection with two named vectors (dense + sparse BM25 with IDF), and hybrid search fused with Reciprocal Rank Fusion |
| `src/rag/encoders.py` | Local ONNX models via fastembed: dense embeddings, BM25 sparse vectors, cross-encoder reranker |
| `src/rag/retriever.py` | Two-stage retrieval: cheap, high-recall search first, then an expensive, precise cross-encoder on only the top candidates |
| `src/rag/generator.py` | Grounded prompting: numbered context, required citations, an explicit "I don't know" |
| `scripts/eval_retrieval.py` | Retrieval metrics without an LLM: hit@k and MRR against a golden set |
| `scripts/eval_ragas.py` | LLM-as-judge metrics: faithfulness, context precision/recall, factual correctness |
| `eval/golden.jsonl` | The golden set: question, reference answer, and which file holds the answer |

## Results on the sample corpus

Measured on a laptop CPU (8 threads): 126 docs, 3,214 chunks, 22 golden questions. Ingest took about 5 minutes.

| Mode | hit@1 | hit@3 | hit@5 | MRR@10 | ms/query |
|---|---|---|---|---|---|
| dense (bge-small) | 0.68 | 0.86 | 0.91 | 0.787 | 43 |
| sparse (BM25) | 0.77 | 0.95 | 1.00 | 0.860 | 133 |
| hybrid (RRF) | 0.77 | 0.95 | 1.00 | 0.860 | 120 |
| hybrid + rerank, jina-reranker-v1-tiny (default) | 0.77 | 0.91 | 1.00 | 0.861 | 1,741 |
| hybrid + rerank, ms-marco-MiniLM-L-6 | 0.68 | 0.95 | 0.95 | 0.818 | 2,071 |

What the numbers say:

- **BM25 beats dense search here.** Questions about technical docs share exact terms with the answer (`quantize`, `workspace`, `persist`), which is where keyword search shines.
- **Hybrid ties BM25 on this set.** It is insurance for paraphrased questions, and this golden set has few. Exercise 4 tests that.
- **The rerankers did not improve file-level hit rate, and they cost 1.5–2 s per query on CPU.** File-level hit rate can't see whether the best chunk *inside* the right file came first, which is where rerankers usually help. Ragas context precision measures that. Decide from your own numbers, not from blog posts.
- **Batch size 1 was fastest** for both embedding and reranking on this CPU. Batches are padded to their longest text, and the padding cost more than batching saved (see `ENCODER_BATCH` in `config.py`).

Ragas scores need an LLM with API credits. Run `python scripts/eval_ragas.py` and add a row here.

## Exercises for weeks 1–2

Change one thing at a time, re-run `eval_retrieval.py`, and write down the numbers. That log is what you talk about in interviews.

1. **Chunk size.** Try `CHUNK_SIZE=600` and `2000` (re-ingest each time). Smaller chunks tend to rank better but give the LLM less context.
2. **Remove the heading prefix** in `chunk_document` and measure how much it was worth.
3. **Swap models.** `DENSE_MODEL=BAAI/bge-base-en-v1.5` or `RERANK_MODEL=BAAI/bge-reranker-base`: better or worse quality, and at what latency? Also try `CANDIDATES=10` to see how reranker latency scales.
4. **Grow the golden set** to 50 questions, including paraphrased ones that share no keywords with the docs. Note which retrieval mode wins on those.
5. **Add questions the docs can't answer** and check that the model says "I don't know".
6. **Metadata filters.** Store the folder (`guides`, `integrations`, ...) in the payload and filter on it in `store.search`.
7. **Query rewriting.** Ask the LLM for 3 rephrasings of the question, retrieve for each, and fuse the results. This is the bridge to week 3: in LangGraph, that becomes a node.
8. **Serve it.** Wrap `retrieve` + `answer` in a FastAPI `POST /ask` endpoint with Docker.

## Interview talking points this project gives you

- Why hybrid search: dense search misses exact identifiers (`int8`, `workspace`, `yolo11n.engine`), and BM25 misses paraphrases. RRF merges the two by rank, so the scores never need calibrating.
- Why rerank only the top 30: a cross-encoder reads query and chunk together, which is accurate but too slow to run on every chunk.
- How you measured it: hit@k and MRR on a golden set, then Ragas faithfulness to catch hallucinations, not just "it looked right".
- What surprised you: the reranker didn't help on this corpus, and batch size 1 beat batching on CPU. Explaining *why* (exact-term questions, padding cost) is a stronger answer than "I added a reranker and it got better".
- Why the golden set includes questions where the docs changed recently (the default tracker is TrackTrack, INT8 uses `quantize=8`): an LLM answering from memory gets them wrong, so they show whether answers really come from the retrieved context.

The sample corpus is the [Ultralytics docs](https://github.com/ultralytics/ultralytics/tree/main/docs) (AGPL-3.0). It is downloaded on demand and not committed.
