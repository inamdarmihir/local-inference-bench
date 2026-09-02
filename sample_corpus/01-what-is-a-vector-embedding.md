# What Is a Vector Embedding?

A vector embedding is a list of numbers that represents a piece of content
in a way a computer can compare mathematically. Instead of storing a
sentence as literal characters, an embedding model maps that sentence to a
point in a high-dimensional space, typically somewhere between 128 and 1536
numbers depending on the model. Two sentences with similar meaning end up
as points that are close together in that space, even if they don't share
a single word in common. Two sentences with unrelated meaning end up far
apart.

This property is what makes embeddings useful for search. A traditional
keyword search index matches on exact terms or their stems, so a query for
"laptop won't turn on" might miss a document that only says "notebook
fails to power up." An embedding-based search compares meaning instead of
tokens, so the two phrases land near each other in vector space and the
match succeeds. This is usually called semantic search, to distinguish it
from lexical or keyword search.

Embeddings are produced by a trained neural network, usually a
transformer, that has learned during training to place semantically
similar inputs near each other and dissimilar inputs far apart. The
network itself doesn't need to run again at search time for documents that
were already embedded; embedding is done once per document up front, and
the resulting vectors are stored in an index for fast lookup later. Only
the query needs to be embedded live, at query time, which is one reason
embedding-based search can still be fast even though the underlying model
is doing real neural network inference.

Where an embedding is computed matters for both latency and cost. A
hosted API call sends text over the network to a remote model and waits
for the response, which adds network round-trip time and usually a
per-token fee. Running the same model locally, on the same machine that's
doing the rest of the pipeline, avoids the network round trip and the
per-call billing, at the cost of using local CPU or GPU cycles and needing
the model weights on disk. Neither option is free: the local path has a
real hardware and electricity cost, it's just not itemized per API call
the way a hosted model's bill is. Comparing the two fairly means measuring
both sides on their own terms rather than assuming one is free because its
cost isn't shown on an invoice.
