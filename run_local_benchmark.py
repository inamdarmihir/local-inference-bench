"""
run_local_benchmark.py

Runs FastEmbed's default model (BAAI/bge-small-en-v1.5) against a corpus of
markdown files and times it. This is the half of the comparison that's
actually executed on this machine: no API key needed, no mocked timing.
FastEmbed's TextEmbedding.embed() call and its batch_size kwarg are used as
documented in fastembed==0.8.0.

Usage:
    python3 run_local_benchmark.py --help
    python3 run_local_benchmark.py --corpus-dir sample_corpus
    python3 run_local_benchmark.py --corpus-dir path/to/your/docs --out results_local.json

With no --corpus-dir given, this defaults to the in-repo sample_corpus/
directory, never a machine-specific path. See README.md for how this
in-repo sample corpus differs from the corpus used for the published
results_local.json / results_local_warm.json numbers.
"""

import argparse
import json
import os
import platform
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from fastembed import TextEmbedding

from corpus import load_corpus

DEFAULT_CORPUS_DIR = Path("sample_corpus")


@dataclass
class LocalBenchmarkResult:
    model_name: str
    documents_embedded: int
    embedding_dim: int
    wall_clock_seconds: float
    documents_per_second: float
    model_load_seconds: float
    cpu_brand: str
    cpu_cores: int
    platform: str
    python_version: str
    fastembed_batch_size: int


def get_cpu_brand() -> str:
    """macOS-specific: sysctl reports the real CPU model string.
    Falls back to platform.processor() on other OSes, which is often
    empty or generic on Linux/Windows."""
    if platform.system() == "Darwin":
        try:
            out = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
                check=True,
            )
            return out.stdout.strip()
        except Exception:
            pass
    return platform.processor() or "unknown"


def run_local_fastembed(
    documents: list[str],
    model_name: str = "BAAI/bge-small-en-v1.5",
    batch_size: int = 256,
) -> LocalBenchmarkResult:
    load_start = time.perf_counter()
    model = TextEmbedding(model_name=model_name)
    model_load_seconds = time.perf_counter() - load_start

    start = time.perf_counter()
    embeddings = list(model.embed(documents, batch_size=batch_size))
    elapsed = time.perf_counter() - start

    embedding_dim = len(embeddings[0]) if embeddings else 0

    return LocalBenchmarkResult(
        model_name=model_name,
        documents_embedded=len(embeddings),
        embedding_dim=embedding_dim,
        wall_clock_seconds=elapsed,
        documents_per_second=len(documents) / elapsed,
        model_load_seconds=model_load_seconds,
        cpu_brand=get_cpu_brand(),
        cpu_cores=os.cpu_count() or 0,
        platform=platform.platform(),
        python_version=platform.python_version(),
        fastembed_batch_size=batch_size,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run FastEmbed locally against a markdown corpus and time it."
    )
    parser.add_argument(
        "--corpus-dir",
        type=Path,
        action="append",
        dest="corpus_dirs",
        metavar="PATH",
        help="Directory of markdown files to embed. Repeatable to combine "
        "multiple directories. Defaults to the in-repo sample_corpus/ "
        "directory if omitted.",
    )
    parser.add_argument(
        "--out",
        default="results_local.json",
        metavar="PATH",
        help="Where to write the JSON results (default: results_local.json).",
    )
    parser.add_argument(
        "--model",
        default="BAAI/bge-small-en-v1.5",
        help="FastEmbed model name to run (default: BAAI/bge-small-en-v1.5).",
    )
    args = parser.parse_args()

    corpus_dirs = args.corpus_dirs or [DEFAULT_CORPUS_DIR]

    chunks = load_corpus(corpus_dirs)
    if not chunks:
        raise SystemExit(
            f"No markdown files found in {[str(d) for d in corpus_dirs]}. "
            f"Pass --corpus-dir pointing at a directory of .md files."
        )
    documents = [c.text for c in chunks]
    total_words = sum(c.word_count for c in chunks)

    dirs_desc = ", ".join(str(d) for d in corpus_dirs)
    print(f"Loaded {len(documents)} chunks from {dirs_desc}, {total_words} total words")
    print(f"Loading and running {args.model} via FastEmbed...")

    result = run_local_fastembed(documents, model_name=args.model)

    print()
    print(f"model load time:       {result.model_load_seconds:.2f}s")
    print(f"embedding wall clock:  {result.wall_clock_seconds:.3f}s")
    print(f"documents/sec:         {result.documents_per_second:.2f}")
    print(f"embedding dimension:   {result.embedding_dim}")
    print(f"CPU:                   {result.cpu_brand} ({result.cpu_cores} cores)")
    print(f"platform:              {result.platform}")

    out = {
        "corpus": {
            "num_chunks": len(chunks),
            "total_words": total_words,
            "num_source_files": len(set(c.source_file for c in chunks)),
            "corpus_dirs": [str(d) for d in corpus_dirs],
        },
        "benchmark": asdict(result),
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
