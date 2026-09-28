"""
multi_dir_corpus.py

corpus.load_corpus(dirs) scans each directory non-recursively
(directory.glob("*.md")) by design -- see corpus.py's own docstring. A
deeply-nested real-world corpus (e.g. a documentation site with
concepts/, tasks/, tutorials/ subtrees, each holding its own .md files)
needs its leaf directories discovered and passed in individually. This
module does that discovery and fans out to the existing, unmodified
load_corpus() once per leaf directory -- it does not reimplement chunking
and does not modify corpus.py.

Used by notebooks/searchai_colab_benchmark.ipynb to load a large, real
corpus (e.g. kubernetes/website's content/en/docs/ tree) for the GPU/
SearchAI benchmark legs, while still driving run_local_benchmark.py's own
--corpus-dir (repeatable) for the CPU baseline so that leg's methodology
stays identical to the rest of this repo.
"""

import random
from pathlib import Path

from corpus import Chunk, load_corpus


def find_md_leaf_dirs(root: Path) -> list[Path]:
    """Every directory at or under root that directly contains >=1 *.md
    file, sorted for deterministic ordering."""
    root = Path(root)
    dirs = {md_file.parent for md_file in root.rglob("*.md")}
    return sorted(dirs)


def load_corpus_recursive(
    root: Path,
    max_chunks: int | None = None,
    seed: int = 42,
) -> tuple[list[Chunk], list[Path]]:
    """Discovers every leaf directory under root containing markdown
    files and loads/chunks all of them via corpus.load_corpus(), one
    directory at a time (matching its documented non-recursive-per-
    directory contract), then concatenates in leaf-dir order.

    If max_chunks is given and there are more chunks than that, takes a
    reproducible (seeded) random subsample so notebook runtime stays
    predictable regardless of how large the source corpus happens to be
    on a given day.

    Returns (chunks, leaf_dirs) so callers can pass leaf_dirs straight
    into run_local_benchmark.py's own --corpus-dir (repeatable) for a
    methodology-identical CPU baseline over the same directories.
    """
    leaf_dirs = find_md_leaf_dirs(root)
    if not leaf_dirs:
        raise SystemExit(f"No markdown files found anywhere under {root}.")

    all_chunks: list[Chunk] = []
    for directory in leaf_dirs:
        all_chunks.extend(load_corpus([directory]))

    if max_chunks is not None and len(all_chunks) > max_chunks:
        rng = random.Random(seed)
        all_chunks = rng.sample(all_chunks, max_chunks)

    return all_chunks, leaf_dirs


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Recursively discover and load a nested markdown "
        "corpus (e.g. a documentation site) by fanning out to "
        "corpus.load_corpus() once per leaf directory, since that "
        "function only scans one directory non-recursively."
    )
    parser.add_argument("root", type=Path, help="Root directory to search under.")
    parser.add_argument(
        "--max-chunks",
        type=int,
        default=None,
        help="Cap the corpus to this many chunks via a reproducible "
        "random subsample (default: no cap).",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Subsample seed (default: 42)."
    )
    args = parser.parse_args()

    chunks, leaf_dirs = load_corpus_recursive(
        args.root, max_chunks=args.max_chunks, seed=args.seed
    )
    total_words = sum(c.word_count for c in chunks)
    print(f"{len(leaf_dirs)} leaf directories with markdown files under {args.root}")
    print(f"{len(chunks)} chunks from {len(set(c.source_file for c in chunks))} files")
    print(f"{total_words} total words")
    if chunks:
        print(f"{total_words / len(chunks):.0f} avg words/chunk")
