.PHONY: demo qdrant-demo smoke clean

# One-command clone-and-run demo: venv + pip install + sample_corpus +
# cost_comparison, end to end. See demo.sh for details.
demo:
	./demo.sh

# Optional: same sample_corpus embeddings, pushed into a real (in-memory)
# Qdrant collection, plus one demo similarity search.
qdrant-demo: demo
	. .venv/bin/activate && python3 push_to_qdrant.py --corpus-dir sample_corpus \
		--query "how does chunking work?"

# Fast CI-style check: run the demo pipeline against sample_corpus only,
# fail if either script errors. Does not compare against published numbers.
smoke: demo

clean:
	rm -rf .venv results_local_sample.json
