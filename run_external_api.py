"""
run_external_api.py

The external-API counterpart to run_local_benchmark.py, for anyone who has
a real OpenAI API key and wants to measure live latency themselves against
their own corpus. Uses `client.embeddings.create(input=..., model=...)`
from openai==3.6.0, which returns a CreateEmbeddingResponse with `.data`
and `.usage.total_tokens`.

This script was never run to produce this repo's published numbers
(results_local.json, results_local_warm.json, or the README's cost table):
no OPENAI_API_KEY was available when those were generated, so no billed
call was made. Running this script yourself makes a real, billed OpenAI
API call against whatever corpus you point it at.

Usage (requires OPENAI_API_KEY in the environment):
    python3 run_external_api.py --corpus-dir sample_corpus --confirm
"""

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from openai import OpenAI

from corpus import load_corpus
from pricing import OPENAI_TEXT_EMBEDDING_3_SMALL

DEFAULT_CORPUS_DIR = Path("sample_corpus")


@dataclass
class ApiBenchmarkResult:
    model: str
    documents_embedded: int
    total_tokens: int
    wall_clock_seconds: float
    documents_per_second: float
    total_cost_usd: float
    batch_size: int


def run_external_api(
    documents: list[str],
    model: str = "text-embedding-3-small",
    batch_size: int = 100,
) -> ApiBenchmarkResult:
    client = OpenAI()

    start = time.perf_counter()
    total_tokens = 0
    for i in range(0, len(documents), batch_size):
        batch = documents[i : i + batch_size]
        response = client.embeddings.create(input=batch, model=model)
        total_tokens += response.usage.total_tokens
    elapsed = time.perf_counter() - start

    cost = (total_tokens / 1_000_000) * OPENAI_TEXT_EMBEDDING_3_SMALL.usd

    return ApiBenchmarkResult(
        model=model,
        documents_embedded=len(documents),
        total_tokens=total_tokens,
        wall_clock_seconds=elapsed,
        documents_per_second=len(documents) / elapsed,
        total_cost_usd=cost,
        batch_size=batch_size,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Makes real, billed OpenAI API calls against a markdown "
        "corpus. Requires OPENAI_API_KEY. Never run for this repo's "
        "published numbers."
    )
    parser.add_argument(
        "--corpus-dir",
        type=Path,
        action="append",
        dest="corpus_dirs",
        metavar="PATH",
        help="Directory of markdown files to embed. Repeatable. Defaults "
        "to the in-repo sample_corpus/ directory if omitted.",
    )
    parser.add_argument(
        "--out",
        default="results_api.json",
        metavar="PATH",
        help="Where to write the JSON results (default: results_api.json).",
    )
    parser.add_argument(
        "--model",
        default="text-embedding-3-small",
        help="OpenAI embedding model to call (default: text-embedding-3-small).",
    )
    parser.add_argument(
        "--confirm",
        action="store_true",
        help="Required flag, acknowledges this makes a real billed API call.",
    )
    args = parser.parse_args()

    if not args.confirm:
        raise SystemExit(
            "This makes a real, billed OpenAI API call. Rerun with --confirm "
            "once OPENAI_API_KEY is set and you intend to spend real money."
        )

    corpus_dirs = args.corpus_dirs or [DEFAULT_CORPUS_DIR]
    documents = [c.text for c in load_corpus(corpus_dirs)]
    result = run_external_api(documents, model=args.model)

    print(f"{result.model}: {result.documents_per_second:.2f} docs/sec, "
          f"{result.total_tokens} tokens, ${result.total_cost_usd:.6f}")

    with open(args.out, "w") as f:
        json.dump(asdict(result), f, indent=2)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
