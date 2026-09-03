"""
cost_comparison.py

Prices a local FastEmbed run against cited hosted-API costs. The normal
path reads everything it needs — num_chunks, num_tokens,
wall_clock_seconds, documents_per_second — straight out of a
run_local_benchmark.py results JSON via --results, fully offline, with no
need for the original markdown corpus to exist on this machine.

--results must point at a file written by the current run_local_
benchmark.py to use that offline path: it records "num_tokens" (via
tokens.py) alongside "total_words". Files written before that field
existed, including this repo's own committed results_local.json /
results_local_warm.json, don't have it. For those, pass --corpus-dir
pointing at the exact corpus that produced the file, and this script will
load and tokenize it instead — the only case where markdown is read.
Otherwise, see README.md for the published numbers directly.

Every number printed or written is tagged with an explicit "source":
"measured" (this machine's recorded local run), "cited" (external, dated
pricing, from pricing.py), "projected" (linear extrapolation to a larger
corpus), or "measured+cited" (cost figures that combine the two), so a
reader can't blend them.
"""

import argparse
import json
from pathlib import Path

from pricing import AWS_C7G_XLARGE, OPENAI_TEXT_EMBEDDING_3_SMALL
from tokens import count_tokens


def load_results(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def resolve_num_chunks_and_tokens(
    local: dict, results_path: str, corpus_dirs: list[Path] | None
) -> tuple[int, int]:
    """Returns (num_chunks, num_tokens), preferring the values already
    recorded in the results JSON (fully offline) and falling back to
    loading --corpus-dir and re-tokenizing only if the JSON predates
    'num_tokens'.
    """
    corpus = local.get("corpus", {})
    num_chunks = corpus.get("num_chunks")
    num_tokens = corpus.get("num_tokens")

    if num_chunks is not None and num_tokens is not None:
        return num_chunks, num_tokens

    if not corpus_dirs:
        raise SystemExit(
            f"{results_path} has no 'num_tokens' in its 'corpus' section. "
            f"This is expected for files written before token counts were "
            f"tracked, including this repo's committed results_local.json "
            f"and results_local_warm.json. Pass --corpus-dir pointing at "
            f"the exact corpus that produced {results_path} so tokens can "
            f"be counted from the markdown directly, or see README.md for "
            f"the published numbers without needing that corpus at all."
        )

    from corpus import load_corpus  # only needed for this fallback path

    chunks = load_corpus(corpus_dirs)
    if num_chunks is not None and num_chunks != len(chunks):
        print(
            f"warning: {results_path} was measured against {num_chunks} "
            f"chunks, but {[str(d) for d in corpus_dirs]} produced "
            f"{len(chunks)} chunks just now. The token count below may not "
            f"match the corpus that actually produced {results_path}'s "
            f"throughput numbers.\n"
        )
    documents = [c.text for c in chunks]
    return len(chunks), count_tokens(documents)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare a run_local_benchmark.py results file's "
        "measured cost against cited hosted-API pricing. Runs fully "
        "offline when --results already has 'num_tokens'."
    )
    parser.add_argument(
        "--results",
        default="results_local.json",
        metavar="PATH",
        help="Path to a JSON file written by run_local_benchmark.py "
        "(default: results_local.json). If it includes a 'num_tokens' "
        "field, this runs fully offline with no corpus needed. Older "
        "files without that field (e.g. this repo's committed "
        "results_local_warm.json) need --corpus-dir instead.",
    )
    parser.add_argument(
        "--corpus-dir",
        type=Path,
        action="append",
        dest="corpus_dirs",
        metavar="PATH",
        help="Directory of markdown files to tokenize. Only used as a "
        "fallback when --results has no 'num_tokens' field; ignored "
        "otherwise. Repeatable. Must be the same corpus that produced "
        "--results, or the token count won't match its throughput numbers.",
    )
    parser.add_argument(
        "--out",
        metavar="PATH",
        help="Optional path to also write the comparison as JSON, with an "
        "explicit 'source' field (measured/cited/projected/"
        "measured+cited) on every numeric group.",
    )
    args = parser.parse_args()

    try:
        local = load_results(args.results)
    except FileNotFoundError:
        raise SystemExit(
            f"{args.results} not found. Run run_local_benchmark.py first "
            f"(it writes results_local.json by default), then pass that "
            f"path here with --results."
        )

    bench = local["benchmark"]
    num_chunks, total_tokens = resolve_num_chunks_and_tokens(
        local, args.results, args.corpus_dirs
    )

    docs_per_second = bench["documents_per_second"]
    wall_clock = bench["wall_clock_seconds"]

    api_cost_this_corpus = (total_tokens / 1_000_000) * OPENAI_TEXT_EMBEDDING_3_SMALL.usd
    local_cost_rented_compute = (wall_clock / 3600) * AWS_C7G_XLARGE.usd

    print("=" * 72)
    print("MEASURED (recorded in --results, this machine's run)")
    print("=" * 72)
    print(f"corpus:                 {num_chunks} chunks, {total_tokens} tokens (cl100k_base)")
    print(f"local model:            {bench['model_name']} via FastEmbed")
    print(f"local wall clock:       {wall_clock:.3f}s ({docs_per_second:.2f} docs/sec)")
    print(f"local hardware:         {bench['cpu_brand']} ({bench['cpu_cores']} cores)")
    print()

    print("=" * 72)
    print("CITED (external, dated, not measured here)")
    print("=" * 72)
    print(
        f"OpenAI text-embedding-3-small: "
        f"${OPENAI_TEXT_EMBEDDING_3_SMALL.usd} {OPENAI_TEXT_EMBEDDING_3_SMALL.unit} "
        f"({OPENAI_TEXT_EMBEDDING_3_SMALL.source}, checked "
        f"{OPENAI_TEXT_EMBEDDING_3_SMALL.checked_date})"
    )
    print(
        f"AWS c7g.xlarge on-demand: ${AWS_C7G_XLARGE.usd} {AWS_C7G_XLARGE.unit} "
        f"({AWS_C7G_XLARGE.source}, checked {AWS_C7G_XLARGE.checked_date})"
    )
    print()

    print("=" * 72)
    print(f"COST AT THIS CORPUS'S ACTUAL SIZE ({total_tokens} tokens, {num_chunks} chunks)")
    print("=" * 72)
    print(f"API cost (OpenAI, cited rate):              ${api_cost_this_corpus:.6f}")
    print(f"local cost, already-owned machine:           $0.00 (marginal cost)")
    print(
        f"local cost, if renting equivalent compute:   "
        f"${local_cost_rented_compute:.6f} "
        f"({wall_clock:.2f}s of ${AWS_C7G_XLARGE.usd}/hr compute)"
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
        scale_factor = scale_docs / num_chunks
        proj_tokens = total_tokens * scale_factor
        proj_api_cost = (proj_tokens / 1_000_000) * OPENAI_TEXT_EMBEDDING_3_SMALL.usd
        proj_local_seconds = scale_docs / docs_per_second
        proj_local_rented_cost = (proj_local_seconds / 3600) * AWS_C7G_XLARGE.usd

        print(
            f"PROJECTION to {scale_docs:,} chunks (linear scaling of the "
            f"measured {docs_per_second:.2f} docs/sec and {total_tokens/num_chunks:.0f} "
            f"avg tokens/chunk, not a new run):"
        )
        print(f"  projected tokens:                    {proj_tokens:,.0f}")
        print(f"  projected API cost:                  ${proj_api_cost:,.2f}")
        print(
            f"  projected local time:                {proj_local_seconds/3600:.2f} hours "
            f"({proj_local_seconds:,.0f}s)"
        )
        print(
            f"  projected local cost (rented, {AWS_C7G_XLARGE.usd}/hr):  "
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
                "num_chunks": num_chunks,
                "num_tokens": total_tokens,
                "results_file": args.results,
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
                "openai_usd_per_1m_tokens": OPENAI_TEXT_EMBEDDING_3_SMALL.usd,
                "openai_pricing_source": OPENAI_TEXT_EMBEDDING_3_SMALL.source,
                "openai_pricing_checked_date": OPENAI_TEXT_EMBEDDING_3_SMALL.checked_date,
                "aws_c7g_xlarge_usd_per_hour": AWS_C7G_XLARGE.usd,
                "aws_pricing_source": AWS_C7G_XLARGE.source,
                "aws_pricing_checked_date": AWS_C7G_XLARGE.checked_date,
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
