"""Load source files from disk (Markdown, plain text, PDF) and the golden eval set."""

import json
from dataclasses import dataclass, field
from pathlib import Path

from pypdf import PdfReader

SUFFIXES = {".md", ".markdown", ".txt", ".pdf"}


@dataclass
class Document:
    text: str
    source: str  # path relative to the data dir; used for citations and retrieval evals
    metadata: dict = field(default_factory=dict)


def load_documents(data_dir: Path) -> list[Document]:
    docs = []
    for path in sorted(data_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SUFFIXES:
            continue
        source = path.relative_to(data_dir).as_posix()
        if path.suffix.lower() == ".pdf":
            # One Document per page, so an answer can cite the page it came from.
            for number, page in enumerate(PdfReader(path).pages, start=1):
                text = page.extract_text() or ""
                if text.strip():
                    docs.append(Document(text, source, {"page": number}))
        else:
            docs.append(Document(path.read_text(encoding="utf-8", errors="ignore"), source))
    return docs


def load_golden(path: Path) -> list[dict]:
    """One JSON object per line: {"question", "reference", "sources"}."""
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
