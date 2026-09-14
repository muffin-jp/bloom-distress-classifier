.PHONY: seed propose propose-run generate generate-run review stats splits vendor-model baselines train bench test lint types check

# Import the golden cases from the companion repo into data/seed.jsonl.
# Re-run only if bloom-langgraph's eval dataset changes.
seed:
	uv run python scripts/import_seed.py

# Ask the teacher what it thinks, to order the review queue. Costs money;
# both scripts refuse to send anything without --yes.
# `make propose` only prints an estimate. `make propose-run` actually spends.
# Extra flags go through ARGS, e.g. `make propose-run ARGS="--votes 5"`.
propose:
	uv run --extra teacher python scripts/propose_labels.py --dry-run $(ARGS)

propose-run:
	uv run --extra teacher python scripts/propose_labels.py --yes $(ARGS)

# Expand the curated taxonomy into more candidates for review.
# `make generate` estimates; `make generate-run` spends.
generate:
	uv run --extra teacher python scripts/generate_candidates.py --dry-run $(ARGS)

generate-run:
	uv run --extra teacher python scripts/generate_candidates.py --yes $(ARGS)

# The only path from candidate to training data.
review:
	uv run python scripts/review.py --reviewed-by "$(REVIEWER)"

stats:
	uv run python scripts/review.py --stats

# (Re)build the frozen train/test assignment and commit data/splits.json.
# Adding rows does not reshuffle existing groups — see src/dc/splits.py.
splits:
	uv run python -m dc.splits

# One ~90MB download of the pinned MiniLM weights, needed only for baseline 3
# and the model itself. Everything else in this repo runs offline without it.
vendor-model:
	uv run --extra embed python -m dc.features

# Baselines 0-4, 5-fold CV on train. Never touches the test split.
# Drop the extra (or pass --skip-embedding) to run without the embedder.
baselines:
	uv run --extra embed python scripts/run_baselines.py $(ARGS)

# The train.py spine: select by grouped CV, refit, write artifacts/ and
# reports/training.*. Never touches the test split. Run after committing source
# so the artifact's provenance records a clean commit.
train:
	uv run --extra embed python -m dc.train

# Latency snapshot to reports/latency.md. Local path only by default; add
# ARGS="--llm-calls 20 --yes" to time real production calls (costs money).
bench:
	uv run --extra embed --extra teacher python scripts/bench_latency.py $(ARGS)

test:
	uv run pytest -q

lint:
	uv run ruff check .
	uv run ruff format --check .

types:
	uv run pyright

check: lint types test
