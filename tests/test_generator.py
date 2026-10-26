from langchain_core.language_models.fake_chat_models import FakeListChatModel

from rag import generator
from rag.retriever import Hit

HITS = [
    Hit("Export > TensorRT\n\nUse quantize=8.", "integrations/tensorrt.md", "Export > TensorRT", 0.9),
    Hit("Modes > Track\n\nThe default tracker is TrackTrack.", "modes/track.md", "Modes > Track", 0.5),
]


def test_context_is_numbered_with_sources():
    context = generator.format_context(HITS)
    assert context.startswith("[1] source: integrations/tensorrt.md\nExport > TensorRT")
    assert "\n\n[2] source: modes/track.md\n" in context


def test_answer_returns_model_text(monkeypatch):
    fake = FakeListChatModel(responses=["Pass quantize=8 [1]."])
    monkeypatch.setattr(generator, "chat_model", lambda model: fake)
    result = generator.answer("How do I export INT8?", HITS, model="fake")
    assert result.text == "Pass quantize=8 [1]."
    assert result.hits == HITS


def test_no_hits_means_no_llm_call(monkeypatch):
    monkeypatch.setattr(generator, "chat_model", lambda model: (_ for _ in ()).throw(AssertionError("called")))
    assert generator.answer("anything", []).text == "I don't know based on the provided documents."
