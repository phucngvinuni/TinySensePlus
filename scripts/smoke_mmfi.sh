#!/usr/bin/env bash
# Run a small MM-Fi training and evaluation check.
#   MMFI_ROOT=/path/to/MMFi_Dataset bash scripts/smoke_mmfi.sh
set -euo pipefail
cd "$(dirname "$0")/.."

: "${MMFI_ROOT:?Set MMFI_ROOT to the MM-Fi dataset folder (the one containing E01..E04)}"

mkdir -p runs
python train.py \
  --data-root "$MMFI_ROOT" \
  --output-dir runs/smoke \
  --epochs 3 \
  --max-samples 256 \
  --prefix-ratios 1.0,0.5 \
  --test-error-rates 0.0,0.3 \
  "$@" \
  2>&1 | tee runs/smoke_mmfi.log
