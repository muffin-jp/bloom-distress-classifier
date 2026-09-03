.PHONY: seed splits test lint types check

# Import the golden cases from the companion repo into data/seed.jsonl.
# Re-run only if bloom-langgraph's eval dataset changes.
seed:
	uv run python scripts/import_seed.py

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
