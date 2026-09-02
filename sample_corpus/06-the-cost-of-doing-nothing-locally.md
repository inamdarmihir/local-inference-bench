# The Cost of Doing Nothing, Locally

It's tempting to describe local compute that's already paid for as free,
since running one more job on an idle machine doesn't generate a new
invoice. That framing is useful for a specific question, marginal cost,
but it quietly hides a different question that matters just as much:
utilization. A machine that's idle ninety percent of the time and running
an embedding job the other ten percent really is paying close to zero
marginal cost for that job, because the hardware would have been sitting
there powered on regardless.

The same local machine dedicated full time to running embedding jobs is
in a very different situation. Now the hardware cost, the electricity, and
the eventual replacement cost all need to be amortized across the
work it does, and the comparison against a hosted API's per-token price
has to include that amortized cost, not just call it zero. A hosted
provider's price already bakes in their own hardware, power, and
utilization economics; ignoring the equivalent costs on the local side
because they don't appear on a monthly bill produces a rigged comparison
that favors local by construction rather than by evidence.

A more honest way to frame the choice is to price two versions of the
local option side by side: marginal cost on genuinely idle,
already-owned hardware, which really can be close to zero, and rented
cost on equivalent hardware, using a real published price for a
comparable cloud instance, as a stand-in for what the local option costs
when the hardware isn't already sitting there for free. The rented number
will usually be more pessimistic than the true cost for a team with
already-idle hardware, and more realistic for a team deciding whether to
provision new hardware specifically for this workload. Reporting both,
labeled clearly as two different scenarios rather than one blended figure,
gives a reader enough information to substitute in their own utilization
assumptions instead of inheriting the author's.
