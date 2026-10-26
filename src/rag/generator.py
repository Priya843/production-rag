"""Answer a question from retrieved chunks, with numbered citations."""

from dataclasses import dataclass
from functools import lru_cache

from langchain.chat_models import init_chat_model

from rag.config import settings
from rag.retriever import Hit

SYSTEM_PROMPT = """You answer questions about technical documentation using only the numbered context passages you are given.

- Cite the passages that support each statement, like [1] or [2][4].
- If the passages do not contain the answer, reply exactly: "I don't know based on the provided documents." Do not fill gaps from your own knowledge; a wrong answer with a citation is worse than no answer.
- Be concise and concrete. Quote exact argument names, commands and values from the passages."""


@dataclass
class Answer:
    question: str
    text: str
    hits: list[Hit]


@lru_cache
def chat_model(model: str):
    # Same code for any provider: "anthropic:claude-opus-5-5", "openai:gpt-5-mini", ...
    return init_chat_model(model, max_tokens=16000)


def format_context(hits: list[Hit]) -> str:
    return "\n\n".join(f"[{i}] source: {h.source}\n{h.text}" for i, h in enumerate(hits, start=1))


def answer(question: str, hits: list[Hit], model: str | None = None) -> Answer:
    if not hits:
        return Answer(question, "I don't know based on the provided documents.", hits)
    response = chat_model(model or settings.llm_model).invoke(
        [
            ("system", SYSTEM_PROMPT),
            ("human", f"Context passages:\n\n{format_context(hits)}\n\nQuestion: {question}"),
        ]
    )
    return Answer(question, response.text, hits)
