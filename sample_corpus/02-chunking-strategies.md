# Chunking Strategies for Retrieval

Most documents are too long to embed as a single vector without losing
useful detail, so retrieval systems split source documents into smaller
pieces, called chunks, before embedding each one separately. The choice of
chunk size and boundary rule has a real effect on retrieval quality, and
there's no single setting that's correct for every corpus.

A fixed-size chunker splits text every N characters or tokens, regardless
of where sentences or paragraphs fall. It's simple to implement and gives
predictable chunk sizes, but it can cut a sentence in half or separate a
claim from the evidence that supports it, landing the two halves in
different chunks that might not both be retrieved together.

A paragraph-based chunker instead treats paragraph breaks as natural
boundaries, packing whole paragraphs into a chunk until a target word or
token count is reached, then starting a new chunk at the next paragraph
boundary. This keeps each chunk's content coherent, at the cost of some
variance in chunk size, since paragraphs are rarely the same length. A
long paragraph that already exceeds the target size on its own is usually
kept intact rather than split further, since arbitrarily cutting it
would likely be worse than leaving one oversized chunk.

Code blocks and other pre-formatted content deserve special handling in
technical documentation. Splitting a fenced code block across two chunks
usually makes both chunks harder to use: neither one shows a runnable
example, and re-assembling the original code from two separate retrieval
hits is not something most downstream systems attempt. Treating a fenced
block as a single atomic unit during chunking, even when it pushes past
the target chunk size, tends to produce better results than a size limit
enforced without exception.

There is a real tradeoff between chunk size and retrieval precision.
Smaller chunks let a search return more narrowly relevant text, since each
result carries a single idea rather than several unrelated ones bundled
together. Larger chunks preserve more surrounding context per result,
which can matter when a downstream reader needs that context to make sense
of the retrieved text. A common practical range for prose is somewhere
between 200 and 500 words per chunk, though the right number for a given
application depends on the kind of documents being indexed and how the
retrieved chunks get used afterward.
