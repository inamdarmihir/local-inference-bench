# Measuring a Benchmark Honestly

A benchmark is only as useful as the honesty of its labels. Three
different kinds of numbers tend to show up side by side in a performance
or cost comparison, and conflating them is one of the most common ways a
benchmark ends up misleading, even when every individual number in it is
technically correct.

The first kind is a measured number: something a script actually ran and
timed on real hardware, against a real input, during this session. A
measured wall-clock time for embedding a specific set of documents on a
specific machine is about as solid as a number gets, but it's also
narrow. It describes that run, on that hardware, with that software
version, and it may not generalize to a different machine, a different
corpus size, or a different day, when a network path or thermal
throttling behaves differently.

The second kind is a cited number: a fact pulled from an external,
dated source rather than reproduced locally, such as a vendor's published
price per unit. A cited number can be perfectly accurate and still not be
independently verified by the person citing it, so the honest thing to do
is label it with where it came from and when it was checked, since prices
and specifications do change over time and a stale citation can quietly
become wrong.

The third kind is a projected number: an extrapolation of a measured rate
to a different scale, computed with arithmetic rather than a new
measurement. Scaling a measured throughput or cost linearly out to a much
larger input size is a reasonable way to estimate what a larger run might
look like, but it assumes the measured rate holds constant at that new
scale, which is not guaranteed. Larger inputs can hit different bottlenecks,
like memory pressure or network contention, that a small-scale measurement
never encounters.

None of these three kinds of number is inherently better or worse than the
others, they answer different questions. The failure mode is presenting
them without labels, so a reader can't tell that one figure in a table was
timed five minutes ago and the next one was extrapolated from it, or
copied from a webpage. Keeping the three kinds in clearly separate,
explicitly labeled groups, even at the cost of a slightly less tidy table,
is what makes a benchmark something a reader can actually trust and reuse.
