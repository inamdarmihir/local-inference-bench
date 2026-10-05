.PHONY: demo qdrant-demo test smoke clean help

help:
	@echo "make demo        embed sample_corpus/ and price it (creates .venv)"
	@echo "make qdrant-demo demo + push vectors into an in-memory Qdrant collection"
	@echo "make test        run the offline unit tests"
	@echo "make clean       remove .venv and sample results"

# One-command clone-and-run demo: venv + pip install + sample_corpus +
# cost_comparison, end to end. See demo.sh for details.
demo:
	./demo.sh

# Optional: same sample_corpus embeddings, pushed into a real (in-memory)
# Qdrant collection, plus one demo similarity search.
qdrant-demo: demo
	. .venv/bin/activate && python3 push_to_qdrant.py --corpus-dir sample_corpus \
		--query "how does chunking work?"

# Offline unit tests: chunking, cost math, and the published README figures.
test: demo-venv
	. .venv/bin/activate && pip install --quiet pytest && pytest -q

# Fast CI-style check: run the demo pipeline against sample_corpus only,
# fail if either script errors. Does not compare against published numbers.
smoke: demo

demo-venv:
	@test -d .venv || (python3 -m venv .venv && . .venv/bin/activate && pip install --quiet -r requirements.txt)

clean:
	rm -rf .venv results_local_sample.json
