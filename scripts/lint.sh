#!/usr/bin/env bash
# Lint the codebase without modifying files (ruff).
set -euo pipefail
cd "$(dirname "$0")/.."

echo "==> Linting code (ruff)..."
uv run ruff check .

echo "Done. No lint errors."
