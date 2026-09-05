.PHONY: seed propose generate review stats splits test lint types check

# Import the golden cases from the companion repo into data/seed.jsonl.
# Re-run only if bloom-langgraph's eval dataset changes.
seed:
	uv run python scripts/import_seed.py

# Ask the teacher what it thinks, to order the review queue. Costs money;
# both scripts refuse to send anything without --yes.
propose:
	uv run --extra teacher python scripts/propose_labels.py --dry-run

# Expand the curated taxonomy into more candidates for review. Costs money.
generate:
	uv run --extra teacher python scripts/generate_candidates.py --dry-run

# The only path from candidate to training data.
review:
	uv run python scripts/review.py --reviewed-by "$(REVIEWER)"

stats:
	uv run python scripts/review.py --stats

# (Re)build the frozen train/test assignment and commit data/splits.json.
# Adding rows does not reshuffle existing groups — see src/dc/splits.py.
splits:
	uv run python -m dc.splits

test:
	uv run pytest -q

lint:
	uv run ruff check .
	uv run ruff format --check .

types:
	uv run pyright

check: lint types test
