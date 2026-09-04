"""
push_to_qdrant.py

Optional demo: takes the same FastEmbed vectors run_local_benchmark.py
measures and actually puts them in a Qdrant collection, so "real
corpus/collection shape" isn't just a phrase in the README. Uses
QdrantClient(":memory:") by default, an in-process, no-server, no-network
backend, so this stays a local demo with zero external calls and zero
extra cost, consistent with the rest of this repo.

This script does not affect, recompute, or re-tag any of the timing/cost
numbers elsewhere in this repo. It's a separate, optional demonstration
that the vectors FastEmbed produces here are ordinary Qdrant points:
same dimension, same distance metric choice, insertable and searchable
like any other collection.

Usage:
    python3 push_to_qdrant.py --help
    python3 push_to_qdrant.py --corpus-dir sample_corpus --query "how does chunking work?"
    python3 push_to_qdrant.py --corpus-dir sample_corpus --location localhost:6333  # real server
"""

import argparse
from pathlib import Path

from fastembed import TextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from corpus import load_corpus

DEFAULT_CORPUS_DIR = Path("sample_corpus")
DEFAULT_COLLECTION = "local_inference_bench_demo"


def build_collection(
    client: QdrantClient,
    collection_name: str,
    documents: list[str],
    model_name: str = "BAAI/bge-small-en-v1.5",
) -> tuple[TextEmbedding, int]:
    """Embeds ``documents`` with FastEmbed and upserts them into a fresh
    Qdrant collection. Returns the loaded model (reused for query
    embedding) and the vector dimension actually written."""
    model = TextEmbedding(model_name=model_name)
    vectors = list(model.embed(documents))
    dim = len(vectors[0]) if vectors else 0

    if client.collection_exists(collection_name):
        client.delete_collection(collection_name)
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
    )

    points = [
        PointStruct(
            id=i,
            vector=vector.tolist(),
            payload={"text": doc[:300], "chunk_index": i},
        )
        for i, (vector, doc) in enumerate(zip(vectors, documents))
    ]
    client.upsert(collection_name=collection_name, points=points)
    return model, dim


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Embed a markdown corpus with FastEmbed and upsert it "
        "into a real Qdrant collection (in-memory by default, no server "
        "required), then optionally run one demo search. Does not touch "
        "this repo's published results_local*.json numbers."
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
        "--location",
        default=":memory:",
        metavar="ADDR",
        help="QdrantClient location (default: ':memory:', an in-process "
        "backend with no server or network needed). Pass e.g. "
        "'localhost:6333' to use a real running Qdrant instance instead.",
    )
    parser.add_argument(
        "--collection",
        default=DEFAULT_COLLECTION,
        help=f"Collection name to create/overwrite (default: {DEFAULT_COLLECTION}).",
    )
    parser.add_argument(
        "--model",
        default="BAAI/bge-small-en-v1.5",
        help="FastEmbed model name (default: BAAI/bge-small-en-v1.5, "
        "matches run_local_benchmark.py's default).",
    )
    parser.add_argument(
        "--query",
        metavar="TEXT",
        help="Optional query text. If given, embeds it with the same "
        "model and prints the top-3 nearest chunks from the collection.",
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

    client = QdrantClient(location=args.location)
    print(f"Embedding {len(documents)} chunks with {args.model} via FastEmbed...")
    model, dim = build_collection(client, args.collection, documents, model_name=args.model)

    info = client.get_collection(args.collection)
    print(
        f"Collection '{args.collection}' ready at {args.location}: "
        f"{info.points_count} points, dim={dim}, distance=Cosine"
    )

    if args.query:
        query_vector = list(model.embed([args.query]))[0].tolist()
        hits = client.query_points(
            collection_name=args.collection,
            query=query_vector,
            limit=3,
        ).points
        print(f"\nTop {len(hits)} matches for: {args.query!r}\n")
        for rank, hit in enumerate(hits, start=1):
            preview = hit.payload.get("text", "")[:120].replace("\n", " ")
            print(f"{rank}. score={hit.score:.4f}  {preview}...")
    else:
        print("\nPass --query \"some text\" to run a demo similarity search.")


if __name__ == "__main__":
    main()
