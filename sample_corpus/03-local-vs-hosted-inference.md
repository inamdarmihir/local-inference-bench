# Local vs. Hosted Inference for Embeddings

Running an embedding model falls into one of two broad categories: calling
a hosted API that runs the model on someone else's infrastructure, or
running the model directly on hardware you control. Both approaches
produce the same kind of output, a vector per input document, but the
operational and cost tradeoffs differ enough that the choice is worth
making deliberately rather than by default.

A hosted API is easy to start with. There's no model to download, no
runtime to install beyond an HTTP client, and no local compute to
provision or maintain. The provider handles scaling, and the cost shows up
as a metered per-token or per-request fee on a bill. The tradeoff is that
every embedding call leaves the local network, which adds latency, and
the ongoing per-token fee scales linearly with usage in a way that's easy
to forecast but doesn't go away as volume grows.

Local inference trades that convenience for control. A small embedding
model can run entirely on a laptop's CPU using an optimized runtime,
without a GPU or a network connection, and without an API key or account.
The tradeoff is that the machine running it now has a job to do: it's
using CPU cycles and, for larger models, meaningful amounts of memory,
that would otherwise be idle or available for something else. That cost is
real even when there's no separate invoice for it, since the hardware
still has to be bought, powered, and eventually replaced whether it's
busy or idle.

Neither approach is categorically cheaper. At low volume, a hosted
provider's per-token price is often a fraction of a cent per document,
cheap enough that hardware cost dwarfs it. At very high volume, a
local model running on already-owned, already-idle hardware can have an
effectively zero marginal cost per additional document, while the hosted
provider's bill keeps growing every time it's called. Where the crossover
point sits depends on the actual per-token price, the actual local
throughput on real hardware, and how much of that local hardware capacity
would otherwise go unused. Any comparison that skips measuring one side or
the other, or that assumes local compute is free just because it isn't
billed per call, is not actually comparing the two options on equal
footing.
