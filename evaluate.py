"""Evaluate a TinySense+ checkpoint on the MM-Fi test split (rate-distortion sweep)."""

import argparse
import os

import torch

from tinysense_plus import evaluate, load_checkpoint
from tinysense_plus.data import build_mmfi_dataloaders
from tinysense_plus.engine import TrainConfig


def parse_floats(text: str):
    return [float(x) for x in text.split(",") if x.strip()]


def main() -> None:
    d = TrainConfig()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", help="Path to best.pt / last.pt written by train.py")
    parser.add_argument("--data-root", default=os.environ.get("MMFI_ROOT", "data/MMFi_Dataset"))
    parser.add_argument("--split-config", default="configs/mmfi_split.yaml")
    parser.add_argument("--prefix-ratios", type=parse_floats, default=d.prefix_ratios)
    parser.add_argument("--test-error-rates", type=parse_floats, default=d.test_error_rates)
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = load_checkpoint(args.checkpoint, device)
    loaders = build_mmfi_dataloaders(args.data_root, args.split_config, args.max_samples)
    evaluate(
        model, loaders["test"], device, args.prefix_ratios, args.test_error_rates,
        out_json=args.output_json,
    )


if __name__ == "__main__":
    main()
