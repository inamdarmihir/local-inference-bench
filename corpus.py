"""
corpus.py

Loads and chunks a directory (or directories) of markdown files. Which
directories to load is a caller decision, not a module-level constant: see
`load_corpus(dirs)` below and `run_local_benchmark.py --corpus-dir` for how
callers point this at a corpus. This module has no default corpus and no
machine-specific paths baked in.

Chunking is paragraph-based: walk each document's paragraphs in order and
pack them into a chunk until adding the next paragraph would push the chunk
over MAX_WORDS, then start a new chunk. A single paragraph longer than
MAX_WORDS (e.g. a large fenced code block) is kept whole rather than cut
mid-sentence, so chunk length is a target, not a hard cap.
"""

import re
from dataclasses import dataclass
from pathlib import Path

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


def load_corpus(dirs: list[Path]) -> list[Chunk]:
    """Loads and chunks every ``*.md`` file in each of ``dirs``.

    Args:
        dirs: Directories to scan for markdown files. Each directory is
            scanned non-recursively (``directory.glob("*.md")``); pass
            multiple directories to combine several sources into one corpus.

    Returns:
        All chunks from all files, in directory order then filename order.
    """
    all_chunks: list[Chunk] = []
    for directory in dirs:
        for path in sorted(directory.glob("*.md")):
            text = path.read_text(encoding="utf-8")
            all_chunks.extend(chunk_document(str(path), text))
    return all_chunks


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Load and chunk a markdown corpus, print summary stats."
    )
    parser.add_argument(
        "--corpus-dir",
        type=Path,
        action="append",
        dest="corpus_dirs",
        metavar="PATH",
        help="Directory of markdown files to load. Repeatable. "
        "Defaults to the in-repo sample_corpus/ directory if omitted.",
    )
    args = parser.parse_args()
    corpus_dirs = args.corpus_dirs or [Path("sample_corpus")]

    chunks = load_corpus(corpus_dirs)
    total_words = sum(c.word_count for c in chunks)
    print(f"{len(chunks)} chunks from {len(set(c.source_file for c in chunks))} files")
    print(f"{total_words} total words")
    if chunks:
        print(f"{total_words / len(chunks):.0f} avg words/chunk")
