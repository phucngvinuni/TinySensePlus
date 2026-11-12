#!/usr/bin/env bash
# Train TinySense+ with a 512-entry codebook on MM-Fi.
#   MMFI_ROOT=/path/to/MMFi_Dataset CUDA_VISIBLE_DEVICES=0 bash scripts/train_mmfi_k512.sh
set -euo pipefail
cd "$(dirname "$0")/.."

: "${MMFI_ROOT:?Set MMFI_ROOT to the MM-Fi dataset folder (the one containing E01..E04)}"

mkdir -p runs
python train.py \
  --data-root "$MMFI_ROOT" \
  --output-dir runs/mmfi \
  --k 512 \
  --epochs 100 \
  --tau 0.01 \
  --lambda-rate 0.1 \
  --lambda-ent 1 \
  --warmup-epochs 2 \
  "$@" \
  2>&1 | tee runs/mmfi_k512.log
