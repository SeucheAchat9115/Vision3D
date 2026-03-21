#!/usr/bin/env bash
set -euo pipefail

# Convert the NuScenes mini dataset into Vision3D's internal format.
NUSCENES_ROOT="D:/06_Datasets/01_nuscenes/v1.0-mini"
OUTPUT_ROOT="data/nuscenes_mini"
VERSION="v1.0-mini"

uv run python tools/convert_nuscenes.py \
  --nuscenes_root "$NUSCENES_ROOT" \
  --output_root "$OUTPUT_ROOT" \
  --version "$VERSION"
