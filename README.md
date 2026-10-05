<p align="center">
  <a href="https://github.com/inamdarmihir/local-inference-bench">
    <img src="docs/assets/banner.svg" width="800px" alt="Local Inference Bench: FastEmbed next to Qdrant, priced against cited API rates">
  </a>
</p>

<p align="center">
  <a href="#quickstart">Quickstart</a>
  ·
  <a href="#published-results">Results</a>
  ·
  <a href="#cost-comparison">Cost comparison</a>
  ·
  <a href="#gpu-and-searchai-leg">GPU / SearchAI</a>
  ·
  <a href="#known-limitations">Limitations</a>
</p>

<p align="center">
  <a href="https://github.com/inamdarmihir/local-inference-bench/actions/workflows/ci.yml">
    <img src="https://github.com/inamdarmihir/local-inference-bench/actions/workflows/ci.yml/badge.svg" alt="CI">
  </a>
  <a href=".python-version">
    <img src="https://img.shields.io/badge/python-3.10--3.13-blue.svg" alt="Python 3.10 to 3.13">
  </a>
  <a href="https://github.com/qdrant/fastembed">
    <img src="https://img.shields.io/badge/embeddings-FastEmbed-6f42c1.svg" alt="FastEmbed">
  </a>
  <a href="https://qdrant.tech">
    <img src="https://img.shields.io/badge/vector%20db-Qdrant-dc244c.svg" alt="Qdrant">
  </a>
  <a href="LICENSE">
    <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT">
  </a>
</p>

# Local Inference Bench

Running an embedding model next to Qdrant instead of calling an external API for every embedding is a real latency and cost lever. Most write-ups skip measuring the local side or invent numbers for the API side. **This repo measures the local side for real and prices the API side from published, dated rates**, because no billed API key was available. Every figure is tagged `measured`, `cited` or `projected`, so the three are never blended.

| | |
| --- | --- |
| **Measured** | FastEmbed `BAAI/bge-small-en-v1.5` (`fastembed==0.8.0`) run locally over a real corpus of markdown chunks, timed on real hardware. |
| **Cited** | OpenAI and AWS prices with source links and check dates, kept in one file ([`pricing.py`](pricing.py)). |
| **Projected** | Linear extrapolations of a measured rate, labeled as projections everywhere they appear. |
| **Reproducible** | One command runs the whole pipeline on a bundled sample corpus. Published numbers can be inspected from committed JSON. |

> [!IMPORTANT]
> **`sample_corpus/` is a demo corpus for clone-and-run only.** The published numbers below came from a separate, uncommitted 38-chunk corpus and are **inspect-only**, not reproducible by rerunning anything here. Running `./demo.sh` will not reproduce them, by design. See [Inspect the published numbers](#inspect-the-published-numbers).

## Contents

- [Data and ground truth](#data-and-ground-truth)
- [Published results](#published-results)
- [Quickstart](#quickstart)
- [Inspect the published numbers](#inspect-the-published-numbers)
- [Run it on your own corpus](#run-it-on-your-own-corpus)
- [GPU and SearchAI leg](#gpu-and-searchai-leg)
- [What was measured](#what-was-measured)
- [Cost comparison](#cost-comparison)
- [How the JSON maps to these numbers](#how-the-json-maps-to-these-numbers)
- [Measured vs cited vs projected](#measured-vs-cited-vs-projected-vs-unverified)
- [Known limitations](#known-limitations)
- [Repository layout](#repository-layout)
- [Versions](#versions)
- [Development](#development)

## Data and ground truth

**This repo has no labeled benchmark dataset, because it does not measure answer or retrieval quality.** It measures embedding throughput and prices it. The "ground truth" is therefore hardware timings and published prices.

| Item | Source | Notes |
| --- | --- | --- |
| Published-run corpus (38 chunks, 8,091 words, 13,111 tokens) | The author's own markdown article drafts | **Private and not committed.** The published numbers are inspect-only; see [Inspect the published numbers](#inspect-the-published-numbers). Unedited outputs are in [`results_local.json`](results_local.json) and [`results_local_warm.json`](results_local_warm.json). |
| Demo corpus | [`sample_corpus/`](sample_corpus/) | 6 short original files written for this repo. For trying the pipeline only. |
| GPU leg corpus | [`kubernetes/website`](https://github.com/kubernetes/website) docs, CC-BY-4.0 (as stated in the notebook) | Fetched by the Colab notebook at run time; not committed. |
| Embedding model | [`BAAI/bge-small-en-v1.5`](https://huggingface.co/BAAI/bge-small-en-v1.5) via [FastEmbed](https://github.com/qdrant/fastembed) | Default FastEmbed model, `fastembed==0.8.0`. |
| Token counts | [tiktoken](https://github.com/openai/tiktoken), `cl100k_base` | Real tokenizer, not an estimate. |
| OpenAI price ($0.02 per 1M tokens) | [OpenAI model page](https://platform.openai.com/docs/models/text-embedding-3-small) | Cited, checked 2026-09-01. Prices change; re-check before reuse. |
| AWS price ($0.145/hr, `c7g.xlarge`, us-east-1) | [Vantage instance page](https://instances.vantage.sh/aws/ec2/c7g.xlarge) | Cited, checked 2026-09-01. |
| API latency context | Zep, [*Text embedding latency: a semi-scientific look*](https://blog.getzep.com/text-embedding-latency-a-semi-scientific-look/) | Related data point only; different models. No API latency was measured here. |

**Real vs derived.** Timings, throughput and token counts are measured. Prices are cited. The 10k and 1M-chunk figures are linear projections and are labeled as such. No OpenAI API call was ever made for the published numbers.

## Published results

Corpus: 38 chunks, 13,111 tokens, Apple M2. Cited pricing checked 2026-09-01.

| | source | this corpus (13,111 tokens) | projected to 1M chunks |
| --- | --- | --- | --- |
| FastEmbed local, `BAAI/bge-small-en-v1.5` | `measured` | 4.26 docs/sec, 8.93 s wall clock | n/a |
| OpenAI `text-embedding-3-small` API cost | `measured+cited` | $0.000262 | $6.90 |
| FastEmbed local, rented AWS `c7g.xlarge` ($0.145/hr) | `measured+cited` | $0.000360 | $9.46 |
| FastEmbed local, already-owned machine | `measured` | $0.00 marginal | $0.00 marginal |

**Headline finding:** at this measured throughput, renting equivalent cloud compute to run FastEmbed is *not* cheaper than OpenAI's API for this model and pricing. Local inference only wins once the compute is already owned and the marginal cost really is zero. See [Cost comparison](#cost-comparison) for why that caveat matters.

## Quickstart

Clone-and-run needs nothing beyond `pip install`, using the sample corpus in this repo.

```bash
git clone https://github.com/inamdarmihir/local-inference-bench.git
cd local-inference-bench
./demo.sh          # or: make demo
```

That creates a virtual environment, installs the pinned dependencies, embeds `sample_corpus/` with FastEmbed and prices the run offline. It is equivalent to:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python3 run_local_benchmark.py --corpus-dir sample_corpus --out results_local_sample.json
python3 cost_comparison.py --results results_local_sample.json
```

The sample corpus is 6 short original markdown files, so chunk count, timing and cost differ from the published run. `cost_comparison.py --results` runs fully offline: `run_local_benchmark.py` writes a real tiktoken `num_tokens` count into its output JSON, so pricing never needs the markdown files. Run any script with `--help` for its flags.

### See the vectors in a Qdrant collection

`push_to_qdrant.py` upserts the same FastEmbed vectors into a Qdrant collection (in-memory by default: no server, no network, no cost) and runs one similarity search:

```bash
python3 push_to_qdrant.py --corpus-dir sample_corpus --query "how does chunking work?"
# or: make qdrant-demo
```

Pass `--location localhost:6333` (or a Qdrant Cloud URL) to push into a running instance. This shows the vectors are valid Qdrant points; it is not a recall or ranking benchmark.

## Inspect the published numbers

The published figures came from 38 chunks of the author's own article drafts, which live on one machine and are **not checked in**. A clone cannot regenerate them, only inspect them:

- [`results_local.json`](results_local.json): unedited output of the first (cold) run.
- [`results_local_warm.json`](results_local_warm.json): unedited output of the second (warm) run.
- The tables in this README are computed from those two files and the cited prices in [`pricing.py`](pricing.py). [`tests/test_cost_comparison.py`](tests/test_cost_comparison.py) asserts that the code reproduces the published cost figures from them.

Both files predate the `num_tokens` field and are kept byte-for-byte as written. `cost_comparison.py --results results_local_warm.json` says so and stops rather than silently loading a corpus that is not in this repo.

## Run it on your own corpus

`--corpus-dir` is repeatable and is the one public interface every script shares. No script has a machine-specific default.

```bash
python3 run_local_benchmark.py --corpus-dir path/to/your/docs --out results_local.json
python3 cost_comparison.py --results results_local.json
```

Chunking is paragraph-based with a 200 to 400 word target and works on any markdown corpus (see [`corpus.py`](corpus.py)).

If you have an `OPENAI_API_KEY` and want an actual (not cited) API-side number, run:

```bash
python3 run_external_api.py --corpus-dir path/to/your/docs --confirm
```

It embeds through `text-embedding-3-small` and writes a results file in the same shape. It requires `--confirm` because it spends real money, and it was **never run** for this repo's published numbers.

Whatever local model you use, keep the vector dimension consistent with whatever you compare it to, so a downstream Qdrant recall comparison stays apples to apples. `push_to_qdrant.py` prints the dimension it wrote.

## GPU and SearchAI leg

[`notebooks/searchai_colab_benchmark.ipynb`](notebooks/searchai_colab_benchmark.ipynb) extends the benchmark in Google Colab with a GPU and a self-hosted comparison. It is committed **without outputs and without published numbers**: treat it as runnable tooling, not as results.

| Section | What it does |
| --- | --- |
| Environment fingerprint | Records CPU, GPU, driver and Python so every number can be tied to its hardware |
| CPU baseline | Drives the unmodified `run_local_benchmark.py` over a large real corpus (Kubernetes docs) |
| GPU FastEmbed | `fastembed-gpu` with ONNX Runtime CUDA, installed in an order that avoids the CPU/GPU package conflict |
| Local Qdrant | On-disk Qdrant collection, closer to a real self-hosted deployment than `:memory:` |
| SearchAI Index Server | Installs and probes SearchBlox's SearchAI Index Server and times ingest, only through endpoints that validated |
| Results and download | Combined comparison table and an export before the Colab runtime recycles |

Supporting modules: [`searchai_client.py`](searchai_client.py) (install, run and probe helpers that return the raw HTTP status and body with every verdict) and [`multi_dir_corpus.py`](multi_dir_corpus.py) (discovers nested markdown directories and fans out to the unmodified `corpus.load_corpus()`).

## What was measured

Published run, 38-chunk corpus:

```text
Loaded 38 chunks from real corpus, 8091 total words
Loading and running BAAI/bge-small-en-v1.5 via FastEmbed...

model load time:       0.06s   (22.56s on a cold/first-time HF download)
embedding wall clock:  8.928s
documents/sec:         4.26
embedding dimension:   384
CPU:                   Apple M2 (8 cores)
platform:              macOS-26.6.2-arm64-arm-64bit-Mach-O
```

These are FastEmbed's defaults: `batch_size=256` (the whole corpus fits in one batch), no ONNX thread tuning, CPU only, no GPU or ANE path. 4.26 docs/sec on chunks averaging 345 tokens is not fast. It is what the library does out of the box with zero tuning, which is the realistic case for most people, not the ceiling for this model with a tuned server and concurrent batched requests.

Two runs were made: cold (fresh venv, no cached model) and warm (model on disk). Model *load* time drops from 22.56 s to 0.06 s, as expected. Throughput does not move (4.19 vs 4.26 docs/sec), so the slow part is inference, not a cold-start artifact.

## Cost comparison

Published corpus, cited pricing, both checked 2026-09-01:

| | this corpus (13,111 tokens) | projected to 1M chunks |
| --- | --- | --- |
| OpenAI `text-embedding-3-small` API | $0.000262 | $6.90 |
| FastEmbed local, rented AWS `c7g.xlarge` ($0.145/hr) | $0.000360 | $9.46 |
| FastEmbed local, already-owned machine | $0.00 marginal | $0.00 marginal |

The projection scales the measured 4.26 docs/sec and 345 average tokens per chunk linearly to 1M chunks. It is not a second measurement; `cost_comparison.py` tags it `"source": "projected"` and prints it under a `PROJECTION` header.

The result that matters: renting equivalent cloud compute to run FastEmbed is not cheaper than OpenAI's API here. It becomes a win only when the compute is already owned. Local inference cost depends on utilization, not just per-request cost, and whether it is cheaper depends on whether you pay for the hardware either way.

The `c7g.xlarge` (4 vCPU, 8 GiB, Graviton3, $0.145/hr, us-east-1) is not the M2 the run used. It is a stated proxy for "rent a small general-purpose CPU instance", not a hardware-matched equivalent. Sources: [instances.vantage.sh/aws/ec2/c7g.xlarge](https://instances.vantage.sh/aws/ec2/c7g.xlarge) and [platform.openai.com/docs/models/text-embedding-3-small](https://platform.openai.com/docs/models/text-embedding-3-small) ($0.02 per 1M tokens).

## How the JSON maps to these numbers

Every figure above traces to one of two committed files plus the cited constants in [`pricing.py`](pricing.py):

| README figure | JSON source | field | source tag |
| --- | --- | --- | --- |
| 38 chunks | `results_local_warm.json` | `corpus.num_chunks` | `measured` |
| 8,091 total words | `results_local_warm.json` | `corpus.total_words` | `measured` |
| 13,111 tokens | not in either file (see note) | n/a | `measured` |
| 4.26 docs/sec (warm) | `results_local_warm.json` | `benchmark.documents_per_second` | `measured` |
| 8.928 s wall clock (warm) | `results_local_warm.json` | `benchmark.wall_clock_seconds` | `measured` |
| 0.06 s model load (warm) | `results_local_warm.json` | `benchmark.model_load_seconds` | `measured` |
| 4.19 docs/sec (cold) | `results_local.json` | `benchmark.documents_per_second` | `measured` |
| 22.56 s model load (cold) | `results_local.json` | `benchmark.model_load_seconds` | `measured` |
| 384 embedding dimension | `results_local_warm.json` | `benchmark.embedding_dim` | `measured` |
| Apple M2 (8 cores) | `results_local_warm.json` | `benchmark.cpu_brand`, `cpu_cores` | `measured` |
| $0.02 / 1M tokens (OpenAI) | [`pricing.py`](pricing.py) | `OPENAI_TEXT_EMBEDDING_3_SMALL.usd` | `cited` |
| $0.145 / hr (AWS `c7g.xlarge`) | [`pricing.py`](pricing.py) | `AWS_C7G_XLARGE.usd` | `cited` |
| $0.000262 API cost | `cost_comparison.py` | `cost_at_corpus_size.api_cost_usd` | `measured+cited` |
| $0.000360 rented-compute cost | `cost_comparison.py` | `cost_at_corpus_size.local_cost_rented_usd` | `measured+cited` |
| $6.90 / $9.46 at 1M chunks | `cost_comparison.py` | `projections[].projected_api_cost_usd`, `projected_local_rented_cost_usd` | `projected` |

**Note on the 13,111-token figure.** Both JSON files predate the `num_tokens` field, so it is not literally in either. It is what `cost_comparison.py --corpus-dir <original corpus>` computed when run against the original 38-chunk corpus. It is a real tiktoken count of real chunks, but not one you can regenerate from this repo alone.

Run `cost_comparison.py --results <file> --out <file>.json` against `results_local_sample.json` to see the same table's shape, computed fresh and tagged the same way, in machine-readable form.

## Measured vs cited vs projected vs unverified

- **Measured**: local wall clock, throughput and token count, run on real hardware against a real corpus (the section above, or your own `--corpus-dir` run).
- **Cited**: the $0.02 per 1M token OpenAI price and the $0.145/hr AWS price, published rates checked 2026-09-01. Neither was reproduced by a live call here.
- **Projected**: the 1M-chunk and 10k-chunk figures, linear extrapolations of a measured rate.
- **Explicitly unverified**: API latency for `text-embedding-3-small`. No `OPENAI_API_KEY` was available, so no live call was made and no latency is reported. The closest citable data point is Zep's ["A Survey of Embedding Models"](https://blog.getzep.com/text-embedding-latency-a-semi-scientific-look/) (June 2023), which covers `text-embedding-ada-002` and `textembedding-gecko@001` on single ~20-word sentences. It is a related data point, not a stand-in measurement.

## Known limitations

- The published corpus is small (38 chunks, 13,111 tokens) because it is the real markdown that existed for the project, not padded to a round number. It is not in git, so a clone cannot regenerate it. `sample_corpus/` is a stand-in for trying the pipeline.
- `run_local_benchmark.py` uses FastEmbed defaults. No tuning of batch size or threads, no GPU or ANE. A tuned local server would likely change the comparison.
- The AWS instance is a stated proxy, not a hardware-matched benchmark machine.
- No API latency or API embedding correctness was verified. `run_external_api.py` is real, runnable code for someone with a key; its numbers are not reported because it was never run for the published results.
- `push_to_qdrant.py` shows the vectors are valid Qdrant points. It makes no accuracy claims.
- The Colab notebook ships without outputs; no GPU or SearchAI numbers are published.

## Repository layout

| Path | Purpose |
| --- | --- |
| `corpus.py` | Loads and chunks a markdown corpus (`--corpus-dir`) |
| `multi_dir_corpus.py` | Discovers nested markdown directories for large corpora |
| `tokens.py` | Shared tiktoken counting |
| `pricing.py` | Cited OpenAI and AWS prices, single source of truth |
| `run_local_benchmark.py` | Runs FastEmbed locally, times it, records `num_tokens` |
| `run_external_api.py` | The OpenAI-side script; real code, not run for published numbers |
| `cost_comparison.py` | Prices a results JSON offline and tags each figure's source |
| `push_to_qdrant.py` | Optional: upserts the embedded corpus into a Qdrant collection |
| `searchai_client.py` | Install and probe helpers for SearchAI Index Server |
| `notebooks/` | Colab notebook for the GPU and SearchAI leg |
| `demo.sh`, `Makefile` | One-command clone-and-run demo and shortcuts |
| `sample_corpus/` | 6 short original markdown files, for clone-and-run only |
| `results_local*.json` | Committed outputs of the cold and warm published runs |
| `tests/` | Offline tests, including the check that code reproduces the published costs |

## Versions

Pinned in [`requirements.txt`](requirements.txt) so a clone-and-run today behaves like the original build.

| Dependency | Pinned version | Used by |
| --- | --- | --- |
| Python | 3.10 to 3.13 tested (see [`.python-version`](.python-version)) | everything |
| `fastembed` | 0.8.0 | `run_local_benchmark.py`, `push_to_qdrant.py` |
| `tiktoken` | 0.14.0 | `tokens.py` |
| `openai` | 3.6.0 | `run_external_api.py` |
| `qdrant-client` | 1.15.1 (optional) | `push_to_qdrant.py` only |

The published numbers ran on Python 3.13.15 (macOS). CI and sandbox verification used Python 3.11 and 3.12 on Linux, to confirm the pipeline is not macOS-specific beyond `get_cpu_brand()`'s documented `sysctl` fallback.

## Development

```bash
pip install -r requirements.txt pytest
pytest -q          # or: make test
make demo          # end-to-end smoke on sample_corpus/
make clean         # remove the venv and sample results
```

CI ([`.github/workflows/ci.yml`](.github/workflows/ci.yml)) runs the tests and the demo pipeline on `sample_corpus/` for every push and pull request. It is a pipeline smoke test, not a benchmark, and does not compare against published numbers.

## License

[MIT](LICENSE) © Mihir Inamdar.
