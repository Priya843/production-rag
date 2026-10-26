"""Ask a question from the command line.

    python scripts/ask.py "How do I export YOLO to TensorRT with INT8?"
    python scripts/ask.py "..." --no-llm       # retrieval only, no API key needed
    python scripts/ask.py "..." --no-rerank    # see what the reranker changes
"""

import argparse
import time

from rag.config import settings
from rag.generator import answer
from rag.retriever import Retriever


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("question")
    parser.add_argument("--no-llm", action="store_true", help="only show the retrieved chunks")
    parser.add_argument("--no-rerank", action="store_true")
    parser.add_argument("-k", type=int, default=settings.top_k)
    args = parser.parse_args()

    retriever = Retriever()
    start = time.perf_counter()
    hits = retriever.retrieve(args.question, top_k=args.k, rerank=not args.no_rerank)
    retrieval_ms = (time.perf_counter() - start) * 1000
    retriever.close()

    for i, hit in enumerate(hits, start=1):
        preview = hit.text.replace("\n", " ")[:160]
        print(f"[{i}] {hit.score:7.3f}  {hit.source}\n      {preview}...")
    print(f"\nretrieval: {retrieval_ms:.0f} ms (includes loading the models; see eval_retrieval.py for warm timings)")

    if not args.no_llm:
        start = time.perf_counter()
        result = answer(args.question, hits)
        print(f"generation ({settings.llm_model}): {(time.perf_counter() - start):.1f} s\n")
        print(result.text)


if __name__ == "__main__":
    main()
