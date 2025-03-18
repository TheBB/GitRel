# Test targets

.PHONY: pytest
pytest:
	uv run pytest

.PHONY: mypy
mypy:
	uv run mypy

.PHONY: lint-check
lint-check:
	uv run ruff check
	uv run ruff format --check

.PHONY: test
test: pytest mypy lint-check
