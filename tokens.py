"""
tokens.py

Real tokenizer, shared by run_local_benchmark.py (to record a token count
alongside timing) and cost_comparison.py (to price those tokens), so both
count tokens the same way instead of duplicating the tiktoken call.
"""

import tiktoken

TOKENIZER_MODEL = "text-embedding-3-small"  # tiktoken maps this to cl100k_base


def count_tokens(documents: list[str]) -> int:
    """Counts tokens across ``documents`` using the real tiktoken encoding
    for TOKENIZER_MODEL (cl100k_base), not an approximation."""
    enc = tiktoken.encoding_for_model(TOKENIZER_MODEL)
    return sum(len(enc.encode(doc)) for doc in documents)
