"""Compare retrieval strategies on the golden set. Uses no LLM, so it is free and fast to rerun.

    python scripts/eval_retrieval.py
    python scripts/eval_retrieval.py --show-misses

Run it after every change to chunking, models or search settings: if the right
document is not retrieved, no prompt can fix the answer.
"""

import argparse
import time
from pathlib import Path

from rag.config import ROOT, settings
from rag.loaders import load_golden
from rag.metrics import first_relevant_rank, hit_rate, mrr
from rag.retriever import Retriever

DEPTH = 10  # ranks are measured within the top 10


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--golden", type=Path, default=ROOT / "eval" / "golden.jsonl")
    parser.add_argument("--show-misses", action="store_true", help="list questions the best mode missed")
    args = parser.parse_args()

    golden = load_golden(args.golden)
    retriever = Retriever()
    retriever.retrieve("warm up the models")  # keep model loading out of the timings

    modes = {
        "dense": lambda q: retriever.search(q, "dense", DEPTH),
        "sparse (BM25)": lambda q: retriever.search(q, "sparse", DEPTH),
        "hybrid (RRF)": lambda q: retriever.search(q, "hybrid", DEPTH),
        "hybrid + rerank": lambda q: retriever.retrieve(q, top_k=DEPTH),
    }

    print(f"{len(golden)} questions, candidates for rerank = {settings.candidates}\n")
    print(f"{'mode':<17} {'hit@1':>6} {'hit@3':>6} {'hit@5':>6} {'MRR@10':>7} {'ms/query':>9}")
    misses = []
    for name, run in modes.items():
        ranks, start = [], time.perf_counter()
        for item in golden:
            hits = run(item["question"])
            rank = first_relevant_rank([h.source for h in hits], set(item["sources"]))
            ranks.append(rank)
            if name == "hybrid + rerank" and (rank is None or rank > 3):
                misses.append((item, rank, [h.source for h in hits[:3]]))
        ms = (time.perf_counter() - start) * 1000 / len(golden)
        print(f"{name:<17} {hit_rate(ranks, 1):>6.2f} {hit_rate(ranks, 3):>6.2f} "
              f"{hit_rate(ranks, 5):>6.2f} {mrr(ranks):>7.3f} {ms:>9.0f}")
    retriever.close()

    if args.show_misses:
        print("\nhybrid + rerank, relevant source not in top 3:")
        for item, rank, top in misses:
            print(f"- {item['question']}\n    expected {item['sources']}, rank {rank}, got {top}")


if __name__ == "__main__":
    main()
