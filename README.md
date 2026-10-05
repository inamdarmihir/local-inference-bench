# Local Inference Bench

**Measure local embedding throughput before deciding what it costs.**

This repo times FastEmbed on your Markdown corpus, records the hardware and vector dimensions, then calculates a cost comparison using dated API and compute prices. An optional Qdrant demo inserts the vectors and runs a similarity query.

The important boundary: **local inference was measured; the OpenAI API was not benchmarked for the committed results.** API cost is calculated from cited pricing. API latency and retrieval-quality equivalence are not established here.

[Quickstart](#quickstart) · [Committed results](#committed-results) · [Your own corpus](#your-own-corpus) · [Limits](#limits) · [Article](https://aihive.hashnode.dev/fastembed-vs-openai-embedding-cost-qdrant-docs)

## Quickstart

Use Python with virtual-environment support and internet access for dependency and model downloads. Pinned dependencies are in [`requirements.txt`](requirements.txt).

```bash
git clone https://github.com/inamdarmihir/local-inference-bench.git
cd local-inference-bench
./demo.sh
```

The script creates `.venv`, embeds the six Markdown files in `sample_corpus/`, writes `results_local_sample.json`, and prices that run. It does not call a billed embedding API or overwrite the committed result files.

**The demo corpus is not the corpus behind the committed measurements.** Expect your own timing and chunk count, not the table below.

### Optional Qdrant demo

```bash
make qdrant-demo
```

This uses `QdrantClient(":memory:")` by default. It demonstrates insertion and a similarity query, not retrieval accuracy. A server is not required for this path.

## Committed results

The two JSON files contain runs on an Apple M2, eight cores, with `BAAI/bge-small-en-v1.5`, 384 dimensions and FastEmbed batch size 256. The corpus contained 38 chunks and 8,091 words across ten source files.

| Run | Model load | Embedding time | Chunks/second |
| --- | ---: | ---: | ---: |
| First run | 22.56 s | 9.06 s | 4.19 |
| Warm run | 0.06 s | 8.93 s | 4.26 |

Sources: [`results_local.json`](results_local.json) and [`results_local_warm.json`](results_local_warm.json). Loading and embedding are separate timings, not a single end-to-end total.

**These measurements are inspectable, not exactly reproducible from this checkout.** The original 38-chunk corpus is not committed. The JSON also predates token-count recording, so it cannot independently reproduce the original token-priced comparison. The linked September article describes a larger run; that run's raw artifacts are not in this checkout, and its numbers are not substituted into this table.

## Your own corpus

```bash
source .venv/bin/activate
python3 run_local_benchmark.py --corpus-dir path/to/docs --out my_results.json
python3 cost_comparison.py --results my_results.json --out my_costs.json
```

`--corpus-dir` can be repeated. [`corpus.py`](corpus.py) loads Markdown and chunks it by paragraphs with a 200-400-word target. New result files include a tiktoken count, allowing the cost script to run from the JSON alone.

```text
Markdown -> chunks -> FastEmbed -> measured timing + token count
                                         |
                                 dated price constants
                                         |
                              cost calculation and projection
```

### What the cost calculation means

[`pricing.py`](pricing.py) stores rates checked **September 1, 2026**:

- OpenAI `text-embedding-3-small`: $0.02 per million tokens.
- AWS `c7g.xlarge`, us-east-1 on-demand: $0.145 per hour.

These are historical inputs, not a current-price promise. Verify and update them before making a purchasing decision. Sources recorded in the code: [OpenAI model page](https://platform.openai.com/docs/models/text-embedding-3-small) and [AWS instance listing](https://instances.vantage.sh/aws/ec2/c7g.xlarge).

The script labels measurements, cited inputs and projections separately. A rented-compute calculation applies the AWS hourly rate to your measured duration; it is **not an AWS throughput measurement**. The M2 and Graviton instance are different hardware. Linear projections to larger corpora are estimates, not additional runs.

An already-owned machine can avoid a per-request API bill, but electricity, hardware, maintenance and opportunity cost do not become zero.

### Optional billed API measurement

[`run_external_api.py`](run_external_api.py) can measure OpenAI on your corpus when you supply an `OPENAI_API_KEY`. It requires `--confirm` because it spends money. It was not run for the committed results. Review its options, price and data disclosure before using it.

Do not infer equal quality from equal vector dimensions. A ranking comparison needs the same retrieval task and relevance judgments.

## Project map

| File | Purpose |
| --- | --- |
| `run_local_benchmark.py` | FastEmbed timing and machine metadata |
| `corpus.py`, `tokens.py` | Markdown chunking and token counting |
| `cost_comparison.py`, `pricing.py` | Offline calculations and dated rates |
| `run_external_api.py` | Optional paid API-side measurement |
| `push_to_qdrant.py` | In-memory or server-backed insertion demo |
| `demo.sh`, `Makefile` | Clone-and-run demo and wrappers |
| `sample_corpus/` | Small original demo documents |
| `results_local*.json` | Older measured outputs |

## Limits

- The committed corpus is small and unavailable in git. Its throughput is not a production-capacity claim.
- No committed API latency, recall comparison, concurrency test or tuned hardware comparison.
- Large batches may exhaust memory on small machines. The historical batch size is not a recommendation for every host.
- `tokens.py` uses tiktoken's default special-token handling. Markdown containing reserved token strings can raise an error; this README does not claim that code issue is fixed.
- The Qdrant demo proves insertability and search execution, not answer quality.

## Contributing

Useful additions are a redistributable real corpus, raw artifacts for the larger article run, safer special-token handling and paired retrieval-quality evaluation. Keep billed API runs optional and preserve the measured/cited/projected distinction.

## License

[MIT](LICENSE).
