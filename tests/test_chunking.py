from rag.chunking import chunk_document, clean_markdown, split_sections
from rag.loaders import Document

DOC = """---
comments: true
description: Export guide
---

# Export

Intro paragraph.

## TensorRT

Use `half=True` for FP16.

```python
# Load a model  <- a comment, not a heading
model.export(format="engine")
```

### Arguments

workspace sets the GPU memory limit.
"""


def test_front_matter_and_html_are_removed():
    text = clean_markdown('---\nkey: value\n---\n# Title\n<p align="center">Hi</p><!-- note -->\n{% include "x.md" %}')
    assert text == "# Title\nHi"


def test_comparison_operators_survive_cleaning():
    assert clean_markdown("if a < 5 and b > 3:") == "if a < 5 and b > 3:"


def test_headings_inside_code_blocks_are_ignored():
    sections = split_sections(clean_markdown(DOC))
    paths = [path for path, _ in sections]
    assert paths == [["Export"], ["Export", "TensorRT"], ["Export", "TensorRT", "Arguments"]]
    assert "# Load a model" in sections[1][1]


def test_chunks_carry_their_heading_path():
    chunks = chunk_document(Document(DOC, "export.md"), size=1200, overlap=100)
    assert chunks[-1].text.startswith("Export > TensorRT > Arguments\n\nworkspace")
    assert all(c.source == "export.md" for c in chunks)


def test_long_sections_are_split_with_overlap_and_size_limit():
    body = "\n\n".join(f"Paragraph {i} " + "word " * 30 for i in range(20))
    chunks = chunk_document(Document(f"# Big\n\n{body}", "big.md"), size=500, overlap=200)
    assert len(chunks) > 1
    assert all(len(c.text) <= 500 + len("Big\n\n") for c in chunks)
    # the last paragraph of one chunk opens the next one
    first_last_para = chunks[0].text.split("\n\n")[-1]
    assert chunks[1].text.split("\n\n")[1] == first_last_para


def test_chunk_ids_are_stable_and_unique():
    a = chunk_document(Document(DOC, "export.md"), 1200, 100)
    b = chunk_document(Document(DOC, "export.md"), 1200, 100)
    assert [c.id for c in a] == [c.id for c in b]
    assert len({c.id for c in a}) == len(a)


def test_plain_text_is_not_parsed_as_markdown():
    chunks = chunk_document(Document("# not a heading in a .txt file", "notes.txt"), 1200, 100)
    assert chunks[0].section == ""
