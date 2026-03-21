#!/usr/bin/env bash
set -euo pipefail

# Install core + development dependencies.
uv sync --group dev

# Install pre-commit hooks for local commits.
uv run pre-commit install
