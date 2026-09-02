# Approximate Nearest Neighbor Search

Once documents are represented as vectors, finding the ones most similar
to a query vector is, at its core, a nearest-neighbor search problem:
given a query point, find the K stored points closest to it by some
distance measure, usually cosine similarity or Euclidean distance. Doing
this exactly, by comparing the query against every stored vector, is
called a brute-force or exact search, and it scales linearly with the
number of stored vectors. For a few thousand vectors that's fast enough to
not matter. For tens of millions, checking every single one on every
query becomes too slow for an interactive application.

Approximate nearest neighbor, or ANN, search trades a small amount of
accuracy for a large amount of speed. Instead of comparing a query against
every stored vector, an ANN index organizes vectors ahead of time into a
structure that lets a search skip large portions of the dataset that are
unlikely to contain the true nearest neighbors, at the cost of
occasionally missing one of them. In practice, well-tuned ANN indexes
recover the vast majority of the true nearest neighbors while running
orders of magnitude faster than brute force at large scale, which is why
essentially every production vector search system uses some form of ANN
rather than exact search.

One common ANN approach builds a graph where each vector is a node
connected to a small set of its approximate neighbors, then a query walks
that graph greedily, hopping toward closer and closer nodes, until it
converges on a good answer without ever touching most of the dataset.
Another common approach partitions the vector space into clusters ahead of
time, and a query only searches the clusters closest to it rather than the
whole dataset. Both families expose tunable parameters that trade index
build time, memory use, and query speed against recall, the fraction of
true nearest neighbors an approximate search actually finds.

Choosing an ANN index and its parameters is a separate decision from
choosing an embedding model, but the two interact: a higher-dimensional
embedding produces vectors that take more memory to index and more time
per distance calculation, so the embedding model's output size is one of
the practical inputs to sizing and tuning the index that stores it.
