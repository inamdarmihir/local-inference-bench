# local-inference-bench

**A real, reproducible measurement of FastEmbed running locally next to
Qdrant, priced against cited external API rates — not a synthetic
benchmark, and honest about which numbers you can rerun vs. only inspect.**

Running an embedding model locally alongside Qdrant, instead of calling an
external API for every embedding, is a real latency and cost lever — but
most write-ups either skip measuring the local side or invent numbers for
the API side. This repo measures one side of it for real: FastEmbed's
default model, run locally with `fastembed==0.8.0` against a real corpus
of markdown chunks (the same shape you'd hand to a Qdrant collection),
timed on real hardware — and prices the other side from real, dated,
published rates, because no billed API key is available in this
environment. Every number below is tagged `measured`, `cited`, or
`projected` so the two are never blended.

> [!IMPORTANT]
> **`sample_corpus/` is a demo corpus for clone-and-run only.** The
> published numbers in this README came from a separate, uncommitted
> 38-chunk corpus and are **inspect-only** — see
> [§3](#3-inspect-the-published-numbers). Running `./demo.sh` will *not*
> reproduce them, by design.

## Contents

1. [Published results, at a glance](#1-published-results-at-a-glance)
2. [Run this on your machine](#2-run-this-on-your-machine)
3. [Inspect the published numbers](#3-inspect-the-published-numbers)
4. [Run this against your own corpus](#4-run-this-against-your-own-corpus)
5. [What's here](#5-whats-here)
6. [What was actually measured (published run)](#6-what-was-actually-measured-published-run-38-chunk-corpus)
7. [Cost comparison](#7-cost-comparison)
8. [How the JSON maps to these numbers](#8-how-the-json-maps-to-these-numbers)
9. [Measured vs. cited vs. projected vs. unverified](#9-measured-vs-cited-vs-projected-vs-unverified)
10. [Known limitations](#10-known-limitations)
11. [Versions](#11-versions)

## 1. Published results, at a glance

Published corpus (38 chunks, 13,111 tokens, Apple M2), cited pricing
checked 2026-09-01. Full detail, sources, and caveats in
[§6](#6-what-was-actually-measured-published-run-38-chunk-corpus) and
[§7](#7-cost-comparison).

| | source | this corpus (13,111 tokens) | projected to 1M chunks |
|---|---|---|---|
| FastEmbed local, `BAAI/bge-small-en-v1.5` | `measured` | 4.26 docs/sec, 8.93s wall clock | — |
| OpenAI `text-embedding-3-small` API cost | `measured+cited` | $0.000262 | $6.90 |
| FastEmbed local, rented AWS `c7g.xlarge` ($0.145/hr) | `measured+cited` | $0.000360 | $9.46 |
| FastEmbed local, already-owned machine | `measured` | $0.00 marginal | $0.00 marginal |

**Headline finding:** at this measured throughput, renting equivalent
cloud compute to run FastEmbed is *not* cheaper than OpenAI's API for this
model and pricing. Local inference only wins once the compute is already
owned and the marginal cost is really zero — see
[§7](#7-cost-comparison) for why that caveat matters.

These numbers can be inspected (not reproduced) from the committed
[`results_local.json`](results_local.json) /
[`results_local_warm.json`](results_local_warm.json); see
[§3](#3-inspect-the-published-numbers).

## 2. Run this on your machine

Clone-and-run works with zero setup beyond `pip install`, using a small
sample corpus checked into this repo. One command:

```bash
./demo.sh
# or: make demo
```

That's equivalent to, and exactly what it runs under the hood:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python3 run_local_benchmark.py --corpus-dir sample_corpus --out results_local_sample.json
python3 cost_comparison.py --results results_local_sample.json
```

**`sample_corpus/` is not the corpus behind the published numbers in
[§1](#1-published-results-at-a-glance).** It's 6 short original markdown
files written for this repo so a stranger can run the pipeline end to end
without any files from Mihir's laptop. It'll produce a different chunk
count, different timing, and a different cost than
`results_local.json` / `results_local_warm.json`. See
[§4](#4-run-this-against-your-own-corpus) to point the same commands at a
real corpus of your own, and [§3](#3-inspect-the-published-numbers) for
the numbers that were actually published for this repo.

`cost_comparison.py --results` runs fully offline: `run_local_benchmark.py`
writes a real tiktoken `num_tokens` count into its output JSON's `corpus`
section, so `cost_comparison.py` never needs the markdown files themselves,
only that JSON. Only pass `--corpus-dir` to `cost_comparison.py` as a
fallback, for older results files (like the committed
`results_local_warm.json`) that predate `num_tokens` — see
[§3](#3-inspect-the-published-numbers).

Run `python3 run_local_benchmark.py --help` and `python3 cost_comparison.py
--help` for the full flag list.

### Optional: see the vectors in an actual Qdrant collection

`push_to_qdrant.py` takes the same FastEmbed vectors and upserts them into
a real Qdrant collection — `QdrantClient(":memory:")` by default, so it's
still zero-server, zero-network, zero-cost — then runs one demo similarity
search:

```bash
python3 push_to_qdrant.py --corpus-dir sample_corpus --query "how does chunking work?"
# or: make qdrant-demo
```

Pass `--location localhost:6333` (or any Qdrant Cloud URL) instead of the
default `:memory:` to push into a real running instance.

## 3. Inspect the published numbers

The numbers quoted in [§1](#1-published-results-at-a-glance) and
[§6](#6-what-was-actually-measured-published-run-38-chunk-corpus) came
from a corpus of 38 chunks — Mihir's own Qdrant Stars article drafts and
their publish-ready copies — that lives on his machine and is **not
checked into this repo**. That corpus can't be re-downloaded or
regenerated by a clone, so those numbers can't be reproduced by rerunning
anything here, only inspected.

No re-run is required to inspect them:

- [`results_local.json`](results_local.json) — the actual, unedited output
  of the first (cold) run against that 38-chunk corpus.
- [`results_local_warm.json`](results_local_warm.json) — the actual,
  unedited output of the second (warm) run against the same corpus.
- [§6](#6-what-was-actually-measured-published-run-38-chunk-corpus) and
  [§7](#7-cost-comparison) below are computed from those two files and
  from the cited pricing in `pricing.py`; nothing there was re-run to
  write this README. See [§8](#8-how-the-json-maps-to-these-numbers) for
  the exact field-by-field mapping.

Both files predate the `num_tokens` field (see
[§2](#2-run-this-on-your-machine)), since they were written before
`run_local_benchmark.py` tracked token counts, and they're kept
byte-for-byte as originally written — not backfilled.
`cost_comparison.py --results results_local_warm.json` on its own will
tell you exactly that and stop, rather than silently loading a corpus
that isn't in this repo. Mihir can still price them locally with
`cost_comparison.py --results results_local_warm.json --corpus-dir
/path/to/that/original/corpus`; a clone can't, since that corpus was
never checked in, which is exactly why the numbers above are quoted
directly instead.

## 4. Run this against your own corpus

`--corpus-dir` is repeatable, so you can point it at one or more real
directories of markdown files — this is the one public interface every
script in this repo shares, and there is no machine-specific default
baked into any of them:

```bash
python3 run_local_benchmark.py --corpus-dir path/to/your/docs --out results_local.json
python3 cost_comparison.py --results results_local.json
```

Chunking is paragraph-based with a 200-400 word target, generic to any
markdown corpus (see [`corpus.py`](corpus.py)). If you have a real
`OPENAI_API_KEY` and want an actual (not cited) API-side number for your
own corpus, `run_external_api.py --corpus-dir path/to/your/docs --confirm`
embeds it through `text-embedding-3-small` and writes a results file in
the same shape, so it can be compared directly instead of against a cited
price alone. It requires `--confirm` because it spends real money, and its
own docstring is explicit that it was never run for this repo's published
numbers.

The one hard requirement carried over from this repo's own run: whatever
local model you use, keep the vector dimension consistent across whatever
you're comparing it to, so a downstream Qdrant recall comparison stays
apples to apples. `push_to_qdrant.py --corpus-dir path/to/your/docs` (see
[§2](#2-run-this-on-your-machine)) is one way to check that directly: it
prints the actual vector dimension it wrote to the collection.

## 5. What's here

```
corpus.py                 loads and chunks a markdown corpus (CLI: --corpus-dir)
tokens.py                 shared tiktoken counting, used by run_local_benchmark.py and cost_comparison.py
pricing.py                shared cited OpenAI/AWS prices, used by cost_comparison.py and run_external_api.py
run_local_benchmark.py    runs FastEmbed locally against it, times the run, records num_tokens
run_external_api.py       the OpenAI-side script, real code, not run for the published numbers (no API key)
cost_comparison.py        prices a results JSON's numbers against cited pricing, offline, tags each figure's source
push_to_qdrant.py         optional: upserts the embedded corpus into a real (in-memory by default) Qdrant collection
demo.sh                   one-command clone-and-run demo (venv + pip install + sample_corpus + cost_comparison)
Makefile                  make demo / make qdrant-demo / make clean, thin wrappers around demo.sh and push_to_qdrant.py
sample_corpus/            6 short original markdown files, for clone-and-run only (not the published corpus)
results_local.json        actual output of the first (cold) published run, checked in
results_local_warm.json   actual output of the second (warm) published run, checked in
requirements.txt          pinned dependency versions (fastembed, openai, tiktoken, qdrant-client)
.github/workflows/ci.yml  smoke test: runs the demo pipeline against sample_corpus/ only, on every push/PR
```

## 6. What was actually measured (published run, 38-chunk corpus)

```
Loaded 38 chunks from real corpus, 8091 total words
Loading and running BAAI/bge-small-en-v1.5 via FastEmbed...

model load time:       0.06s   (22.56s on a cold/first-time HF download)
embedding wall clock:  8.928s
documents/sec:         4.26
embedding dimension:   384
CPU:                   Apple M2 (8 cores)
platform:              macOS-26.6.2-arm64-arm-64bit-Mach-O
```

That's fastembed's default settings: `batch_size=256` (the whole corpus
fits in one batch), no manual ONNX thread tuning, CPU only, no GPU/ANE
path. 4.26 docs/sec on 38 chunks averaging 345 tokens each is not fast.
It's what the library does out of the box on this machine with zero
tuning, which is the realistic case for most people trying this, not the
ceiling of what `bge-small-en-v1.5` can do with a tuned server and batched
concurrent requests.

Two runs were made: a cold one (fresh venv, no cached model) and a warm
one (model already on disk). Model *load* time drops from 22.56s to 0.06s
between them, as expected. Embedding throughput does not move (4.19 vs
4.26 docs/sec), so the slow part is inference itself, not a cold-start
artifact.

## 7. Cost comparison

Published corpus, cited pricing, both checked 2026-09-01:

| | this corpus (13,111 tokens) | projected to 1M chunks |
|---|---|---|
| OpenAI `text-embedding-3-small` API | $0.000262 | $6.90 |
| FastEmbed local, rented AWS `c7g.xlarge` ($0.145/hr) | $0.000360 | $9.46 |
| FastEmbed local, already-owned machine | $0.00 marginal | $0.00 marginal |

The projection scales the measured 4.26 docs/sec and 345 avg tokens/chunk
linearly to 1M chunks. It is not a second measurement; `cost_comparison.py`
labels it `"source": "projected"` in its own JSON output and prints it
under a `PROJECTION` header.

The result that matters here: at this measured throughput, **renting
equivalent cloud compute to run FastEmbed is not cheaper than OpenAI's API**
for this model and pricing. It only becomes a win when the compute is
already owned and the marginal cost really is zero: local inference cost
depends on utilization, not just per-request cost, and whether it's
cheaper depends on whether you're paying for the hardware either way.

The `c7g.xlarge` (4 vCPU, 8 GiB, Graviton3, $0.145/hr, us-east-1) is not
the same hardware as the M2 this actually ran on (8 cores, 8 GB); it's a
stated proxy for "rent a small general-purpose CPU instance," not a
hardware-matched equivalent. Sources:
[instances.vantage.sh/aws/ec2/c7g.xlarge](https://instances.vantage.sh/aws/ec2/c7g.xlarge)
and
[platform.openai.com/docs/models/text-embedding-3-small](https://platform.openai.com/docs/models/text-embedding-3-small)
($0.02 per 1M tokens), both checked 2026-09-01.

## 8. How the JSON maps to these numbers

Every figure in [§1](#1-published-results-at-a-glance),
[§6](#6-what-was-actually-measured-published-run-38-chunk-corpus), and
[§7](#7-cost-comparison) traces back to one of two committed files, plus
the cited constants in [`pricing.py`](pricing.py). Nothing in this README
was hand-typed from a source other than these:

| README figure | JSON source | field | source tag |
|---|---|---|---|
| 38 chunks | `results_local_warm.json` | `corpus.num_chunks` | `measured` |
| 8,091 total words | `results_local_warm.json` | `corpus.total_words` | `measured` |
| 13,111 tokens | *(not in either file — see note below)* | — | `measured` |
| 4.26 docs/sec (warm) | `results_local_warm.json` | `benchmark.documents_per_second` | `measured` |
| 8.928s wall clock (warm) | `results_local_warm.json` | `benchmark.wall_clock_seconds` | `measured` |
| 0.06s model load (warm) | `results_local_warm.json` | `benchmark.model_load_seconds` | `measured` |
| 4.19 docs/sec (cold) | `results_local.json` | `benchmark.documents_per_second` | `measured` |
| 22.56s model load (cold) | `results_local.json` | `benchmark.model_load_seconds` | `measured` |
| 384 embedding dimension | `results_local_warm.json` | `benchmark.embedding_dim` | `measured` |
| Apple M2 (8 cores) | `results_local_warm.json` | `benchmark.cpu_brand`, `cpu_cores` | `measured` |
| $0.02/1M tokens (OpenAI) | [`pricing.py`](pricing.py) | `OPENAI_TEXT_EMBEDDING_3_SMALL.usd` | `cited` |
| $0.145/hr (AWS `c7g.xlarge`) | [`pricing.py`](pricing.py) | `AWS_C7G_XLARGE.usd` | `cited` |
| $0.000262 API cost, this corpus | computed by `cost_comparison.py` | `cost_at_corpus_size.api_cost_usd` | `measured+cited` |
| $0.000360 rented-compute cost, this corpus | computed by `cost_comparison.py` | `cost_at_corpus_size.local_cost_rented_usd` | `measured+cited` |
| $6.90 / $9.46 projected to 1M chunks | computed by `cost_comparison.py` | `projections[].projected_api_cost_usd`, `projected_local_rented_cost_usd` | `projected` |

**Note on the 13,111-token figure:** `results_local.json` /
`results_local_warm.json` both predate the `num_tokens` field (see
[§3](#3-inspect-the-published-numbers)), so it isn't literally present in
either committed file — it's what `cost_comparison.py --corpus-dir
/path/to/that/corpus` computed and printed when it was run against the
original 38-chunk corpus, which is not itself checked in. It's still a
real tiktoken count of real chunks, just not one you can regenerate from
this repo alone; see [§3](#3-inspect-the-published-numbers) for exactly
why.

Run `cost_comparison.py --results <file> --out <file>.json` yourself
(e.g. against `results_local_sample.json` from
[§2](#2-run-this-on-your-machine)) to see this same table's shape,
computed fresh and tagged the same way, in machine-readable form.

## 9. Measured vs. cited vs. projected vs. unverified

This repo mixes several different kinds of numbers on purpose, and keeps
them labeled separately rather than blending them into one table.
`cost_comparison.py --out` writes an explicit `"source"` field
(`measured`, `cited`, `projected`, or `measured+cited`) on every numeric
group in its JSON output for exactly this reason — see
[§8](#8-how-the-json-maps-to-these-numbers) for the full mapping.

- **Measured**: local wall-clock, throughput, and token count, actually
  run on real hardware against a real corpus (the "What was actually
  measured" section above, or your own `run_local_benchmark.py
  --corpus-dir` run).
- **Cited**: the $0.02/1M token OpenAI price and the $0.145/hr AWS price,
  both published rates checked 2026-09-01, linked above. Neither was
  independently reproduced by a live call in this environment.
- **Projected**: the 1M-chunk (and 10k-chunk) figures, a linear
  extrapolation of a measured rate, not a new measurement.
- **Explicitly unverified**: API latency for `text-embedding-3-small`. No
  `OPENAI_API_KEY` is available in this environment, so no live call was
  made and no latency number is reported for it. The closest real,
  citable data point is Zep's
  ["A Survey of Embedding Models"](https://blog.getzep.com/text-embedding-latency-a-semi-scientific-look/)
  (June 2023) — it covers `text-embedding-ada-002` and Google's
  `textembedding-gecko@001` on single ~20-word sentences, not
  `text-embedding-3-small` on ~345-token batches, so it's a related data
  point, not a stand-in measurement.

## 10. Known limitations

- The published corpus is small (38 chunks, 13,111 tokens) because it's
  the actual real markdown that existed for this project, not padded to a
  round number. It's also not in git (see [§3](#3-inspect-the-published-numbers)),
  so it can't be regenerated by a clone; `sample_corpus/` is a stand-in for
  trying the pipeline, not a substitute for those numbers.
- `run_local_benchmark.py` uses fastembed's defaults; no attempt was made
  to tune batch size, thread count, or try a GPU/ANE execution provider.
  A tuned local server would likely change this comparison's outcome;
  that's future work, not something this repo claims to have tried.
- The AWS instance used for the "rent equivalent compute" cost line is a
  stated proxy, not a hardware-matched benchmark machine.
- No API latency or API embedding correctness was verified in this
  environment. `run_external_api.py` is real, runnable code for someone
  with a key; its own numbers are not reported here because it was never
  run for the published numbers.
- `push_to_qdrant.py` demonstrates the vectors are valid Qdrant points
  (right shape, insertable, searchable); it is not a recall or ranking
  benchmark and makes no accuracy claims.

## 11. Versions

Pinned in [`requirements.txt`](requirements.txt) so a clone-and-run today
behaves the same as it did when this repo was built:

| dependency | pinned version | used by |
|---|---|---|
| Python | 3.10–3.13 tested (see [`.python-version`](.python-version)) | everything |
| `fastembed` | 0.8.0 | `run_local_benchmark.py`, `push_to_qdrant.py` |
| `tiktoken` | 0.14.0 | `tokens.py` |
| `openai` | 3.6.0 | `run_external_api.py` |
| `qdrant-client` | 1.15.1 (optional) | `push_to_qdrant.py` only |

The published numbers in this README were run on Python 3.13.15 (macOS);
CI (`.github/workflows/ci.yml`) and this repo's own sandbox verification
of `demo.sh` / `push_to_qdrant.py` used Python 3.11 and 3.12 on Linux, to
confirm the pipeline isn't accidentally macOS- or version-specific beyond
`get_cpu_brand()`'s documented `sysctl` fallback (see
[`run_local_benchmark.py`](run_local_benchmark.py)).

## License

[MIT](LICENSE) © Mihir Inamdar.
