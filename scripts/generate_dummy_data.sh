#!/usr/bin/env bash
set -euo pipefail

# Generate synthetic data for local testing and smoke checks.
OUTPUT_DIR="data/dummy"
NUM_FRAMES="50"

uv run python tools/generate_dummy_dataset.py \
  --output-dir "$OUTPUT_DIR" \
  --num-frames "$NUM_FRAMES"
