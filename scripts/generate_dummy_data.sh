#!/usr/bin/env bash
set -euo pipefail

# Generate synthetic data for local testing and smoke checks.
OUTPUT_ROOT="data/dummy"
NUM_FRAMES="50"

uv run python tools/generate_dummy_dataset.py \
  --output_root "$OUTPUT_ROOT" \
  --num_frames "$NUM_FRAMES"
