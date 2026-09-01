# qdrant-local-inference-bench

Running an embedding model locally alongside Qdrant, instead of calling an
external API for every embedding, is a real latency and cost lever. Most
write-ups skip measuring it because it's easier to demo with an API key
than a local model server. This repo measures one side of it for real
(FastEmbed running locally, on a real corpus, on a real machine) and prices
the other side from real published numbers, because no billed API key is
available in this environment.

The corpus is real: every markdown file in `/Users/apple/Downloads/articles/`
and `/Users/apple/Downloads/aihive-posts-ready/`, which are Mihir's own
Qdrant Stars article drafts and their publish-ready copies (five pieces,
twice each: a draft and a formatted version with frontmatter). 38 chunks,
8,091 words, 13,111 tokens by the actual `text-embedding-3-small` tokenizer.
Not a synthetic benchmark corpus built to look a certain size.

## What's here

```
corpus.py                 loads and chunks the real markdown corpus
run_local_benchmark.py    runs FastEmbed locally against it, times the run
run_external_api.py       the OpenAI-side script, real code, not run here (no API key)
cost_comparison.py        combines the real local numbers with cited API pricing
results_local.json        actual output of the first (cold) run, checked in
results_local_warm.json   actual output of the second (warm) run, checked in
requirements.txt
```

Four scripts, no notebook, no mock data files. The two `results_*.json`
files are the real output already committed, so the numbers below can be
checked without rerunning anything.

## Running it yourself

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python3 run_local_benchmark.py --out results_local.json
python3 cost_comparison.py results_local.json
```

`run_local_benchmark.py` downloads `BAAI/bge-small-en-v1.5` from Hugging
Face on first run (a few seconds to tens of seconds depending on network),
then embeds the full corpus and reports real wall-clock timing.

If you have a real `OPENAI_API_KEY`, `run_external_api.py` will make a real,
billed call against the same corpus and write `results_api.json` in the
same shape. It requires `--confirm` because it spends real money. This
repo's own numbers below don't include anything from that script, since it
was never run here.

## What was actually measured (this machine, this run)

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

## Cost comparison

Real corpus, cited pricing, both checked 2026-09-01:

| | this corpus (13,111 tokens) | projected to 1M chunks |
|---|---|---|
| OpenAI `text-embedding-3-small` API | $0.000262 | $6.90 |
| FastEmbed local, rented AWS `c7g.xlarge` ($0.145/hr) | $0.000360 | $9.46 |
| FastEmbed local, already-owned machine | $0.00 marginal | $0.00 marginal |

The projection scales the measured 4.26 docs/sec and 345 avg tokens/chunk
linearly to 1M chunks. It is not a second measurement, `cost_comparison.py`
labels it as a projection in its own output.

The result that matters here: at this measured throughput, **renting
equivalent cloud compute to run FastEmbed is not cheaper than OpenAI's API**
for this model and pricing. It only becomes a win when the compute is
already owned and the marginal cost really is zero, which is the
qualification the original spec for this repo called out going in: local
inference cost depends on utilization, not just per-request cost, and
whether it's cheaper depends on whether you're paying for the hardware
either way.

The `c7g.xlarge` (4 vCPU, 8 GiB, Graviton3, $0.145/hr, us-east-1) is not
the same hardware as the M2 this actually ran on (8 cores, 8 GB), it's a
stated proxy for "rent a small general-purpose CPU instance," not a
hardware-matched equivalent. Source:
[instances.vantage.sh/aws/ec2/c7g.xlarge](https://instances.vantage.sh/aws/ec2/c7g.xlarge).
OpenAI pricing source:
[platform.openai.com/docs/models/text-embedding-3-small](https://platform.openai.com/docs/models/text-embedding-3-small),
$0.02 per 1M tokens, both checked 2026-09-01.

## Measured vs. cited vs. unverified

This repo mixes three different kinds of numbers on purpose, and keeps
them labeled separately rather than blending them into one table:

- **Independently measured on real hardware**: everything under "What was
  actually measured" above. `run_local_benchmark.py` ran on this machine,
  today, against the real corpus.
- **Cited from a real, dated external source**: the $0.02/1M token OpenAI
  price, and the $0.145/hr AWS price. Both are published rates, checked
  2026-09-01, linked above. Neither was independently reproduced by a live
  call in this environment.
- **Explicitly unverified here**: API latency for `text-embedding-3-small`.
  No `OPENAI_API_KEY` is available in this environment, so no live call
  was made and no latency number is reported for it. The closest real,
  citable data point is Zep's
  ["A Survey of Embedding Models"](https://blog.getzep.com/text-embedding-latency-a-semi-scientific-look/),
  published June 14, 2023. It doesn't cover `text-embedding-3-small`,
  which didn't exist yet, it tested `text-embedding-ada-002` and Google's
  `textembedding-gecko@001`, on single ~20-word sentences (not the ~345
  token batches this corpus produces), 50 iterations, from a GCP
  `n1-standard-4`, an AWS `ml.t3.large`, and a MacBook Pro M1. Their
  reported OpenAI p95 varied wildly by network path in their test, almost
  600ms from AWS and nearly a minute from GCP. That's a real finding from
  a real (if dated) benchmark, not a stand-in for measuring the current
  model. If API latency matters for a real decision, it needs a real key
  and a real call, that's a limitation of this repo, not a gap papered
  over with an estimate.

## Known limitations

- The corpus is small (38 chunks, 13,111 tokens) because it's the actual
  real markdown on disk for this project, not padded to a round number.
  Cost and throughput projections beyond that size are linear
  extrapolations of the measured rate, explicitly labeled as such in
  `cost_comparison.py`'s output, not new measurements.
- `run_local_benchmark.py` uses fastembed's defaults, no attempt was made
  to tune batch size, thread count, or try a GPU/ANE execution provider
  for higher local throughput. A tuned local server would likely change
  this comparison's outcome; that's future work, not something this repo
  claims to have already tried.
- The AWS instance used for the "rent equivalent compute" cost line is a
  stated proxy, not a hardware-matched benchmark machine.
- No API latency or API embedding correctness was verified in this
  environment. `run_external_api.py` is real, runnable code for someone
  with a key, its own numbers are not reported here because it was never
  run.
