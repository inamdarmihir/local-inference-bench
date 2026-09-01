"""
run_local_benchmark.py

Runs FastEmbed's default model against the real corpus in corpus.py and
times it. This is the half of the comparison that's actually executed on
this machine: no API key needed, no mocked timing.

Usage:
    python3 run_local_benchmark.py [--out results_local.json]

FastEmbed usage verified against the installed package directly this
session (`python3 -c "from fastembed import TextEmbedding; import
inspect; print(inspect.signature(TextEmbedding.__init__))"` on
fastembed==0.8.0), not recalled from memory: TextEmbedding defaults to
model_name="BAAI/bge-small-en-v1.5", and .embed() takes a batch_size
kwarg (default 256) and returns an iterable of numpy arrays, one per
input document.
"""

import argparse
import json
import platform
import subprocess
import time
from dataclasses import asdict, dataclass

from fastembed import TextEmbedding

from corpus import load_corpus


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
        cpu_cores=__import__("os").cpu_count() or 0,
        platform=platform.platform(),
        python_version=platform.python_version(),
        fastembed_batch_size=batch_size,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="results_local.json")
    parser.add_argument("--model", default="BAAI/bge-small-en-v1.5")
    args = parser.parse_args()

    chunks = load_corpus()
    documents = [c.text for c in chunks]
    total_words = sum(c.word_count for c in chunks)

    print(f"Loaded {len(documents)} chunks from real corpus, {total_words} total words")
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
        },
        "benchmark": asdict(result),
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
