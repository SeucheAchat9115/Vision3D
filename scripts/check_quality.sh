#!/usr/bin/env bash
set -euo pipefail

# Run lint, format check, type check, and tests.
uv run ruff check .
uv run ruff format --check .
uv run mypy src/
uv run pytest
