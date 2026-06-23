#!/usr/bin/env bash
# Auto-format the codebase: sort imports (ruff) then format (black).
set -euo pipefail
cd "$(dirname "$0")/.."

echo "==> Sorting imports (ruff --fix)..."
uv run ruff check --select I --fix .

echo "==> Formatting code (black)..."
uv run black .

echo "Done. Code formatted."
