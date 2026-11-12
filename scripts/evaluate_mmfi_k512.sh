#!/usr/bin/env bash
# Evaluate a TinySense+ checkpoint on the MM-Fi test split.
#   MMFI_ROOT=/path/to/MMFi_Dataset bash scripts/evaluate_mmfi_k512.sh
# Set CHECKPOINT to evaluate a different checkpoint; extra arguments override defaults.
set -euo pipefail
cd "$(dirname "$0")/.."

: "${MMFI_ROOT:?Set MMFI_ROOT to the MM-Fi dataset folder (the one containing E01..E04)}"

mkdir -p runs
python evaluate.py "${CHECKPOINT:-runs/mmfi/K512/best.pt}" \
  --data-root "$MMFI_ROOT" \
  --prefix-ratios 1.0,0.8,0.6,0.4,0.2,0.1 \
  --test-error-rates 0.0,0.1,0.3 \
  --output-json results.json \
  "$@" \
  2>&1 | tee runs/mmfi_k512_eval.log
