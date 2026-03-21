#!/usr/bin/env bash
set -euo pipefail

# Train with the default config (dummy dataset workflow).
uv run python tools/train.py
