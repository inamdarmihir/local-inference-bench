"""
cost_comparison.py

Combines a real, locally measured FastEmbed throughput number (from
run_local_benchmark.py's JSON output) with cited, dated external prices for
OpenAI's API and a comparable rented cloud CPU instance, to compare local
vs. hosted embedding cost. Every number printed or written is tagged as
"measured" (this machine, this run), "cited" (external, dated, not
reproduced here), or "projected" (linear extrapolation of a measured rate),
so the three are never blended into one unlabeled figure.

Sources, checked 2026-09-01:
  - OpenAI text-embedding-3-small price: $0.02 / 1M tokens.
    https://platform.openai.com/docs/models/text-embedding-3-small
  - AWS EC2 c7g.xlarge (4 vCPU, 8 GiB, Graviton3) on-demand price:
    $0.145/hr, us-east-1. https://instances.vantage.sh/aws/ec2/c7g.xlarge
    Picked as a roughly comparable small CPU instance, not an exact
    hardware match to whatever machine produced the local numbers.
"""

import argparse
import json
from pathlib import Path

import tiktoken

from corpus import load_corpus

DEFAULT_CORPUS_DIR = Path("sample_corpus")

# --- cited, dated external numbers (not measured on this machine) ---
OPENAI_TEXT_EMBEDDING_3_SMALL_USD_PER_1M_TOKENS = 0.02
OPENAI_PRICING_SOURCE = "https://platform.openai.com/docs/models/text-embedding-3-small"
OPENAI_PRICING_CHECKED_DATE = "2026-09-01"

AWS_C7G_XLARGE_USD_PER_HOUR = 0.145
AWS_PRICING_SOURCE = "https://instances.vantage.sh/aws/ec2/c7g.xlarge"
AWS_PRICING_CHECKED_DATE = "2026-09-01"

TOKENIZER_MODEL = "text-embedding-3-small"  # tiktoken maps this to cl100k_base


def real_token_count(documents: list[str]) -> int:
    """Real tokenizer, real corpus. cl100k_base is the encoding
    tiktoken.encoding_for_model('text-embedding-3-small') actually
    returns on the installed tiktoken (checked this session), which is
    the correct encoding for that model, not an approximation."""
    enc = tiktoken.encoding_for_model(TOKENIZER_MODEL)
    return sum(len(enc.encode(doc)) for doc in documents)


def load_local_results(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare measured local FastEmbed cost against cited "
        "hosted-API pricing for the same corpus."
    )
    parser.add_argument(
        "local_results",
        nargs="?",
        default="results_local_warm.json",
        metavar="LOCAL_RESULTS_JSON",
        help="Path to a JSON file written by run_local_benchmark.py "
        "(default: results_local_warm.json, the published warm-run "
        "numbers). Must be measured against the same corpus passed to "
        "--corpus-dir, or the token count and throughput won't match.",
    )
    parser.add_argument(
        "--corpus-dir",
        type=Path,
        action="append",
        dest="corpus_dirs",
        metavar="PATH",
        help="Directory of markdown files to count tokens for. Repeatable. "
        "Defaults to the in-repo sample_corpus/ directory if omitted; use "
        "the same --corpus-dir you passed to run_local_benchmark.py.",
    )
    parser.add_argument(
        "--out",
        metavar="PATH",
        help="Optional path to also write the comparison as JSON, with an "
        "explicit 'source' field (measured/cited/projected/"
        "measured+cited) on every numeric group.",
    )
    args = parser.parse_args()

    local_results_path = args.local_results
    try:
        local = load_local_results(local_results_path)
    except FileNotFoundError:
        raise SystemExit(
            f"{local_results_path} not found. Run run_local_benchmark.py "
            f"first (it writes results_local.json by default), then pass "
            f"that path here."
        )

    corpus_dirs = args.corpus_dirs or [DEFAULT_CORPUS_DIR]
    chunks = load_corpus(corpus_dirs)
    documents = [c.text for c in chunks]
    total_tokens = real_token_count(documents)

    published_num_chunks = local.get("corpus", {}).get("num_chunks")
    if published_num_chunks is not None and published_num_chunks != len(chunks):
        print(
            f"warning: {local_results_path} was measured against "
            f"{published_num_chunks} chunks, but {[str(d) for d in corpus_dirs]} "
            f"produced {len(chunks)} chunks. Token counts and cost below are "
            f"computed from the corpus you pointed --corpus-dir at, not the "
            f"corpus that produced {local_results_path}'s throughput numbers. "
            f"Pass the matching --corpus-dir for an apples-to-apples "
            f"comparison, e.g. the --corpus-dir you gave run_local_benchmark.py.\n"
        )

    bench = local["benchmark"]
    docs_per_second = bench["documents_per_second"]
    wall_clock = bench["wall_clock_seconds"]

    api_cost_this_corpus = (
        total_tokens / 1_000_000
    ) * OPENAI_TEXT_EMBEDDING_3_SMALL_USD_PER_1M_TOKENS

    local_cost_rented_compute = (wall_clock / 3600) * AWS_C7G_XLARGE_USD_PER_HOUR

    print("=" * 72)
    print("MEASURED (this machine, this corpus, this run)")
    print("=" * 72)
    print(f"corpus:                 {len(chunks)} chunks, {total_tokens} tokens (cl100k_base)")
    print(f"local model:            {bench['model_name']} via FastEmbed")
    print(f"local wall clock:       {wall_clock:.3f}s ({docs_per_second:.2f} docs/sec)")
    print(f"local hardware:         {bench['cpu_brand']} ({bench['cpu_cores']} cores)")
    print()

    print("=" * 72)
    print("CITED (external, dated, not measured here)")
    print("=" * 72)
    print(
        f"OpenAI text-embedding-3-small: "
        f"${OPENAI_TEXT_EMBEDDING_3_SMALL_USD_PER_1M_TOKENS}/1M tokens "
        f"({OPENAI_PRICING_SOURCE}, checked {OPENAI_PRICING_CHECKED_DATE})"
    )
    print(
        f"AWS c7g.xlarge on-demand: ${AWS_C7G_XLARGE_USD_PER_HOUR}/hr "
        f"({AWS_PRICING_SOURCE}, checked {AWS_PRICING_CHECKED_DATE})"
    )
    print()

    print("=" * 72)
    print(f"COST AT THIS CORPUS'S ACTUAL SIZE ({total_tokens} tokens, {len(chunks)} chunks)")
    print("=" * 72)
    print(f"API cost (OpenAI, cited rate):              ${api_cost_this_corpus:.6f}")
    print(f"local cost, already-owned machine:           $0.00 (marginal cost)")
    print(
        f"local cost, if renting equivalent compute:   "
        f"${local_cost_rented_compute:.6f} "
        f"({wall_clock:.2f}s of ${AWS_C7G_XLARGE_USD_PER_HOUR}/hr compute)"
    )
    print(
        "\nAt this corpus size the API cost is a fraction of a cent either "
        "way. This comparison only becomes interesting at volume, so the "
        "next block projects the same measured per-token rate forward."
    )
    print()

    # --- projection, explicitly labeled as a projection, not a new measurement ---
    projections: list[dict] = []
    for scale_docs in (10_000, 1_000_000):
        scale_factor = scale_docs / len(chunks)
        proj_tokens = total_tokens * scale_factor
        proj_api_cost = (
            proj_tokens / 1_000_000
        ) * OPENAI_TEXT_EMBEDDING_3_SMALL_USD_PER_1M_TOKENS
        proj_local_seconds = scale_docs / docs_per_second
        proj_local_rented_cost = (proj_local_seconds / 3600) * AWS_C7G_XLARGE_USD_PER_HOUR

        print(
            f"PROJECTION to {scale_docs:,} chunks (linear scaling of the "
            f"measured {docs_per_second:.2f} docs/sec and {total_tokens/len(chunks):.0f} "
            f"avg tokens/chunk, not a new run):"
        )
        print(f"  projected tokens:                    {proj_tokens:,.0f}")
        print(f"  projected API cost:                  ${proj_api_cost:,.2f}")
        print(
            f"  projected local time:                {proj_local_seconds/3600:.2f} hours "
            f"({proj_local_seconds:,.0f}s)"
        )
        print(
            f"  projected local cost (rented, {AWS_C7G_XLARGE_USD_PER_HOUR}/hr):  "
            f"${proj_local_rented_cost:,.2f}"
        )
        print(
            f"  projected local cost (owned machine): $0.00 marginal "
            f"(hardware already sunk, see README caveats)"
        )
        print()

        projections.append(
            {
                "source": "projected",
                "scale_docs": scale_docs,
                "projected_tokens": proj_tokens,
                "projected_api_cost_usd": proj_api_cost,
                "projected_local_seconds": proj_local_seconds,
                "projected_local_rented_cost_usd": proj_local_rented_cost,
                "projected_local_owned_cost_usd": 0.0,
            }
        )

    print("=" * 72)
    print("LATENCY: NOT INDEPENDENTLY MEASURED FOR THE API PATH")
    print("=" * 72)
    print(
        "No API key is available in this environment, so no live OpenAI "
        "embedding call was made and no real API latency number is "
        "reported here. See README.md for why, and for the Zep survey "
        "citation as a related-but-not-directly-applicable data point "
        "(different model, tested in 2023, single-sentence inputs)."
    )

    if args.out:
        out = {
            "corpus": {
                "source": "measured",
                "num_chunks": len(chunks),
                "total_tokens": total_tokens,
                "corpus_dirs": [str(d) for d in corpus_dirs],
                "local_results_file": local_results_path,
            },
            "local_benchmark": {
                "source": "measured",
                "model_name": bench["model_name"],
                "wall_clock_seconds": wall_clock,
                "documents_per_second": docs_per_second,
                "cpu_brand": bench["cpu_brand"],
                "cpu_cores": bench["cpu_cores"],
            },
            "cited_pricing": {
                "source": "cited",
                "openai_usd_per_1m_tokens": OPENAI_TEXT_EMBEDDING_3_SMALL_USD_PER_1M_TOKENS,
                "openai_pricing_source": OPENAI_PRICING_SOURCE,
                "openai_pricing_checked_date": OPENAI_PRICING_CHECKED_DATE,
                "aws_c7g_xlarge_usd_per_hour": AWS_C7G_XLARGE_USD_PER_HOUR,
                "aws_pricing_source": AWS_PRICING_SOURCE,
                "aws_pricing_checked_date": AWS_PRICING_CHECKED_DATE,
            },
            "cost_at_corpus_size": {
                "source": "measured+cited",
                "api_cost_usd": api_cost_this_corpus,
                "local_cost_rented_usd": local_cost_rented_compute,
                "local_cost_owned_usd": 0.0,
            },
            "projections": projections,
        }
        with open(args.out, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
