"""End-to-end eval with Ragas: retrieve -> answer -> an LLM judge scores each answer.

    python scripts/eval_ragas.py --limit 3     # quick, cheap smoke test
    python scripts/eval_ragas.py               # full golden set

Costs API calls (answer model + judge model). Each run is saved to eval/results/.

Metrics (all 0-1, higher is better):
  faithfulness       are the answer's claims supported by the retrieved chunks? (hallucination check)
  context_precision  are the relevant chunks ranked above the irrelevant ones?
  context_recall     do the retrieved chunks contain everything the reference answer needs?
  factual_correctness  does the answer agree with the reference answer?
"""

import argparse
import warnings
from datetime import datetime

from langchain.chat_models import init_chat_model
from ragas import EvaluationDataset, evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import FactualCorrectness, Faithfulness, LLMContextPrecisionWithReference, LLMContextRecall

from rag.config import ROOT, settings
from rag.generator import answer
from rag.loaders import load_golden
from rag.retriever import Retriever

# Ragas 0.4 marks LangchainLLMWrapper deprecated, but it is still the provider-neutral way
# to plug in any LangChain chat model as the judge.
warnings.filterwarnings("ignore", category=DeprecationWarning, module="ragas")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, help="only the first N questions")
    args = parser.parse_args()

    golden = load_golden(ROOT / "eval" / "golden.jsonl")[: args.limit]
    retriever = Retriever()

    rows = []
    for i, item in enumerate(golden, start=1):
        print(f"[{i}/{len(golden)}] {item['question']}")
        hits = retriever.retrieve(item["question"])
        result = answer(item["question"], hits)
        rows.append({
            "user_input": item["question"],
            "response": result.text,
            "retrieved_contexts": [h.text for h in hits],
            "reference": item["reference"],
        })
    retriever.close()

    # bypass_temperature: Ragas otherwise sets a temperature, which current
    # Claude and OpenAI reasoning models reject.
    judge = LangchainLLMWrapper(init_chat_model(settings.judge_model, max_tokens=16000), bypass_temperature=True)
    print(f"\nScoring with judge {settings.judge_model} ...")
    scores = evaluate(
        EvaluationDataset.from_list(rows),
        metrics=[Faithfulness(), LLMContextPrecisionWithReference(), LLMContextRecall(), FactualCorrectness()],
        llm=judge,
    )

    df = scores.to_pandas()
    out = ROOT / "eval" / "results" / f"ragas_{datetime.now():%Y%m%d_%H%M%S}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    print(f"\nanswer model: {settings.llm_model}   judge: {settings.judge_model}   questions: {len(rows)}")
    for column in df.columns:
        if df[column].dtype.kind == "f":
            print(f"  {column:<22} {df[column].mean():.3f}")
    print(f"\nPer-question scores: {out}")


if __name__ == "__main__":
    main()
