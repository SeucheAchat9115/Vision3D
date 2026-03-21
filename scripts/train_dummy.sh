#!/usr/bin/env bash
set -euo pipefail

# Train with the default config (dummy dataset workflow).
uv run python tools/train.py \
  dataset=dummy \
  max_epochs=3 \
  batch_size=2 \
  num_workers=0
