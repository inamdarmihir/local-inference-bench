"""
cost_comparison.py

Computes the cost side of the local-vs-API comparison using:
  - the real corpus (corpus.py) and a real token count from tiktoken
  - the real measured local throughput from run_local_benchmark.py's output
    (results_local_warm.json by default)
  - OpenAI's own published price for text-embedding-3-small, cited below
    with the date it was checked
  - a cited real cloud on-demand hourly rate, as the stated "rent
    equivalent compute" hardware-cost assumption

No number here is invented. What's measured (this machine, this corpus)
and what's cited (external, dated) is kept in separate variables and
labeled as such in the printed output, not blended into one figure.

Sources, checked 2026-09-01:
  - OpenAI text-embedding-3-small price: $0.02 / 1M tokens.
    https://platform.openai.com/docs/models/text-embedding-3-small
    (redirects to https://developers.openai.com/api/docs/models/text-embedding-3-small)
  - AWS EC2 c7g.xlarge (4 vCPU, 8 GiB, Graviton3) on-demand price:
    $0.145/hr, us-east-1.
    https://instances.vantage.sh/aws/ec2/c7g.xlarge
    Picked as a roughly comparable small CPU instance to this dev
    machine (Apple M2, 8 cores / 8 GB), not an exact hardware match,
    vCPU count differs (4 vs 8).
"""

import json
import sys

import tiktoken

from corpus import load_corpus

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


def main():
    local_results_path = sys.argv[1] if len(sys.argv) > 1 else "results_local_warm.json"
    try:
        local = load_local_results(local_results_path)
    except FileNotFoundError:
        print(
            f"{local_results_path} not found. Run run_local_benchmark.py "
            f"first (it writes results_local.json by default; pass that "
            f"path here, or rerun with --out results_local_warm.json)."
        )
        sys.exit(1)

    chunks = load_corpus()
    documents = [c.text for c in chunks]
    total_tokens = real_token_count(documents)

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


if __name__ == "__main__":
    main()
