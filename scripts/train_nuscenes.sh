#!/usr/bin/env bash
set -euo pipefail

# Train with NuScenes dataset settings.
uv run python tools/train.py \
  dataset=nuscenes \
  max_epochs=24 \
  batch_size=4
