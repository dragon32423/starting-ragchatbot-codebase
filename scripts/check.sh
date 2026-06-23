#!/usr/bin/env bash
# Run all quality checks without modifying files. Suitable for CI / pre-commit.
# Fails if formatting, imports, or lint rules are violated, or if tests fail.
set -euo pipefail
cd "$(dirname "$0")/.."

echo "==> Checking formatting (black --check)..."
uv run black --check .

echo "==> Checking import order (ruff --select I)..."
uv run ruff check --select I .

echo "==> Linting (ruff)..."
uv run ruff check .

echo "==> Running tests (pytest, excluding integration)..."
# Integration tests hit the real Anthropic API / ChromaDB and need data loaded,
# so they are excluded from the quality gate. Run them with `uv run pytest`.
uv run pytest -m "not integration"

echo "All quality checks passed."
