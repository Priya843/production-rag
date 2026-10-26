"""Split documents into chunks that remember which section they came from.

Two choices matter more than the exact chunk size:
1. Split on Markdown headings first, so one chunk rarely mixes two topics.
2. Prefix every chunk with its heading path ("TensorRT Export > Arguments"), so a
   chunk that only says "set half=True" can still be found by a question about TensorRT.
"""

import re
import uuid
from dataclasses import dataclass, field

from rag.loaders import Document

HEADING = re.compile(r"^(#{1,6})\s+(.+?)(?:\s+#+)?\s*$")
FENCE = re.compile(r"^\s*(```|~~~)")


@dataclass
class Chunk:
    text: str  # heading path + body: what gets embedded and what the LLM reads
    source: str
    section: str
    metadata: dict = field(default_factory=dict)

    @property
    def id(self) -> str:
        # Deterministic, so re-ingesting the same corpus gives the same point IDs.
        key = f"{self.source}|{self.metadata.get('page', '')}|{self.metadata['index']}"
        return str(uuid.uuid5(uuid.NAMESPACE_URL, key))


def clean_markdown(text: str) -> str:
    """Drop YAML front matter, HTML comments and tags, and MkDocs template tags."""
    text = re.sub(r"\A---\n.*?\n---\n", "", text, flags=re.S)
    text = re.sub(r"<!--.*?-->", "", text, flags=re.S)
    text = re.sub(r"\{%.*?%\}", "", text, flags=re.S)
    text = re.sub(r"</?[a-zA-Z][^<>\n]*>", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_sections(text: str) -> list[tuple[list[str], str]]:
    """Return (heading_path, body) pairs. A '#' inside a code block is not a heading."""
    sections, path, body, in_code = [], [], [], False
    for line in text.splitlines():
        if FENCE.match(line):
            in_code = not in_code
        match = None if in_code else HEADING.match(line)
        if match:
            if "".join(body).strip():
                sections.append((path, "\n".join(body).strip()))
            level = len(match.group(1))
            path = path[: level - 1] + [match.group(2).strip()]
            body = []
        else:
            body.append(line)
    if "".join(body).strip():
        sections.append((path, "\n".join(body).strip()))
    return sections


def _hard_split(text: str, size: int) -> list[str]:
    """Last resort for a block longer than `size`: cut on lines, then on characters."""
    lines = []
    for line in text.splitlines():
        lines.extend(line[i : i + size] for i in range(0, max(len(line), 1), size))
    pieces, current = [], ""
    for line in lines:
        if current and len(current) + 1 + len(line) > size:
            pieces.append(current)
            current = line
        else:
            current = f"{current}\n{line}" if current else line
    if current:
        pieces.append(current)
    return pieces


def _blocks(body: str, size: int) -> list[str]:
    """Paragraphs, with each fenced code block kept whole when it fits."""
    blocks, current, in_code = [], [], False
    for line in body.splitlines():
        if FENCE.match(line):
            in_code = not in_code
        if not line.strip() and not in_code:
            if current:
                blocks.append("\n".join(current))
                current = []
        else:
            current.append(line)
    if current:
        blocks.append("\n".join(current))
    return [piece for block in blocks for piece in (_hard_split(block, size) if len(block) > size else [block])]


def _pack(blocks: list[str], size: int, overlap: int) -> list[str]:
    """Greedily fill chunks up to `size`; start each new chunk with the previous one's tail."""
    chunks, current = [], []
    for block in blocks:
        if current and len("\n\n".join([*current, block])) > size:
            chunks.append("\n\n".join(current))
            tail = []
            for previous in reversed(current):
                if len("\n\n".join([previous, *tail])) > overlap:
                    break
                tail.insert(0, previous)
            current = tail
        current.append(block)
    if current:
        chunks.append("\n\n".join(current))
    return chunks


def chunk_document(doc: Document, size: int, overlap: int) -> list[Chunk]:
    text = doc.text.replace("\r\n", "\n")
    is_markdown = doc.source.lower().endswith((".md", ".markdown"))
    sections = split_sections(clean_markdown(text)) if is_markdown else [([], text.strip())]

    chunks = []
    for path, body in sections:
        section = " > ".join(path)
        for piece in _pack(_blocks(body, size), size, overlap):
            chunk_text = f"{section}\n\n{piece}" if section else piece
            metadata = {**doc.metadata, "index": len(chunks)}
            chunks.append(Chunk(chunk_text, doc.source, section, metadata))
    return chunks


def chunk_documents(docs: list[Document], size: int, overlap: int) -> list[Chunk]:
    return [chunk for doc in docs for chunk in chunk_document(doc, size, overlap)]
