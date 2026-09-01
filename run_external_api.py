"""
run_external_api.py

The external-API counterpart to run_local_benchmark.py, for anyone who
has a real OpenAI API key and wants to measure live latency themselves
against this same corpus. This repo's own README does NOT report numbers
from this script, because this environment has no API key and no billed
call was made. Running it makes a real, billed OpenAI API call.

Call signature verified against the installed SDK directly this session
(`python3 -c "from openai.resources.embeddings import Embeddings; import
inspect; print(inspect.signature(Embeddings.create))"` on openai==3.6.0):
client.embeddings.create(input=..., model=...) returns a
CreateEmbeddingResponse with .data (one entry per input) and
.usage.total_tokens.

Usage (requires OPENAI_API_KEY in the environment):
    python3 run_external_api.py --out results_api.json
"""

import argparse
import json
import time
from dataclasses import asdict, dataclass

from openai import OpenAI

from corpus import load_corpus

# Cited, dated. See cost_comparison.py for the source and check date.
# Kept here too so this script's own cost math doesn't depend on importing
# from a file most people won't read before running a billed call.
OPENAI_TEXT_EMBEDDING_3_SMALL_USD_PER_1M_TOKENS = 0.02


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

    cost = (total_tokens / 1_000_000) * OPENAI_TEXT_EMBEDDING_3_SMALL_USD_PER_1M_TOKENS

    return ApiBenchmarkResult(
        model=model,
        documents_embedded=len(documents),
        total_tokens=total_tokens,
        wall_clock_seconds=elapsed,
        documents_per_second=len(documents) / elapsed,
        total_cost_usd=cost,
        batch_size=batch_size,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Makes real, billed OpenAI API calls. Requires OPENAI_API_KEY."
    )
    parser.add_argument("--out", default="results_api.json")
    parser.add_argument("--model", default="text-embedding-3-small")
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

    documents = [c.text for c in load_corpus()]
    result = run_external_api(documents, model=args.model)

    print(f"{result.model}: {result.documents_per_second:.2f} docs/sec, "
          f"{result.total_tokens} tokens, ${result.total_cost_usd:.6f}")

    with open(args.out, "w") as f:
        json.dump(asdict(result), f, indent=2)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
