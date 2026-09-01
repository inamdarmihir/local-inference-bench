"""
corpus.py

Loads the real document corpus for this benchmark: every markdown file in
/Users/apple/Downloads/articles/ and /Users/apple/Downloads/aihive-posts-ready/.
These are this project's own Qdrant Stars article drafts and their published
versions (same five pieces, draft and publish-ready copies), used as-is
because they're real prose sitting on disk, not text generated for this repo.
To point this at a different corpus, edit CORPUS_DIRS below to your own
directory of markdown files; nothing else in this module is corpus-specific.

No synthetic text anywhere in this file. Chunking is paragraph-based: walk
each document's paragraphs in order and pack them into a chunk until adding
the next paragraph would push the chunk over MAX_WORDS, then start a new
chunk. A single paragraph longer than MAX_WORDS (this corpus has a few,
mostly fenced code blocks) is kept whole rather than cut mid-sentence, so
chunk length is a target, not a hard cap.
"""

import re
from dataclasses import dataclass
from pathlib import Path

CORPUS_DIRS = [
    Path("/Users/apple/Downloads/articles"),
    Path("/Users/apple/Downloads/aihive-posts-ready"),
]

MIN_WORDS = 200
MAX_WORDS = 400

FRONTMATTER_RE = re.compile(r"\A---\n.*?\n---\n", re.DOTALL)


@dataclass
class Chunk:
    source_file: str
    chunk_index: int
    text: str
    word_count: int


def strip_frontmatter(text: str) -> str:
    """Removes YAML frontmatter (title/date/tags/etc.) if present.

    Only the aihive-posts-ready/ copies have it; the articles/ drafts don't.
    Either way this leaves plain markdown prose behind.
    """
    return FRONTMATTER_RE.sub("", text, count=1)


def split_paragraphs(text: str) -> list[str]:
    """Splits on blank lines. Keeps fenced code blocks intact by treating
    the whole fenced region as one paragraph, so a code sample doesn't get
    sliced across chunk boundaries."""
    lines = text.split("\n")
    paragraphs: list[str] = []
    buf: list[str] = []
    in_code_block = False

    def flush():
        joined = "\n".join(buf).strip()
        if joined:
            paragraphs.append(joined)
        buf.clear()

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("```"):
            in_code_block = not in_code_block
            buf.append(line)
            if not in_code_block:
                flush()
            continue
        if not in_code_block and stripped == "":
            flush()
        else:
            buf.append(line)
    flush()
    return paragraphs


def chunk_document(source_file: str, text: str) -> list[Chunk]:
    paragraphs = split_paragraphs(strip_frontmatter(text))

    chunks: list[Chunk] = []
    current: list[str] = []
    current_words = 0
    idx = 0

    def flush():
        nonlocal current, current_words, idx
        if not current:
            return
        joined = "\n\n".join(current)
        chunks.append(
            Chunk(
                source_file=source_file,
                chunk_index=idx,
                text=joined,
                word_count=current_words,
            )
        )
        idx += 1
        current = []
        current_words = 0

    for para in paragraphs:
        para_words = len(para.split())
        if current_words > 0 and current_words + para_words > MAX_WORDS:
            flush()
        current.append(para)
        current_words += para_words
        if current_words >= MIN_WORDS:
            flush()

    flush()
    return chunks


def load_corpus() -> list[Chunk]:
    all_chunks: list[Chunk] = []
    for directory in CORPUS_DIRS:
        for path in sorted(directory.glob("*.md")):
            text = path.read_text(encoding="utf-8")
            all_chunks.extend(chunk_document(str(path), text))
    return all_chunks


if __name__ == "__main__":
    chunks = load_corpus()
    total_words = sum(c.word_count for c in chunks)
    print(f"{len(chunks)} chunks from {len(set(c.source_file for c in chunks))} files")
    print(f"{total_words} total words")
    print(f"{total_words / len(chunks):.0f} avg words/chunk")
