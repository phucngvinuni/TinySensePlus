"""Train TinySense+ on MM-Fi (WiFi-CSI) and run the rate-distortion evaluation."""

import argparse
import json
import os
from dataclasses import asdict

import torch

from tinysense_plus import Discriminator, TinySensePlus, TrainConfig, evaluate, load_checkpoint, train
from tinysense_plus.data import build_mmfi_dataloaders
from tinysense_plus.utils import set_reproducibility


def parse_floats(text: str):
    return [float(x) for x in text.split(",") if x.strip()]


def main() -> None:
    d = TrainConfig()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default=os.environ.get("MMFI_ROOT", "data/MMFi_Dataset"),
                        help="MM-Fi root folder (contains E01..E04). Env: MMFI_ROOT")
    parser.add_argument("--split-config", default="configs/mmfi_split.yaml")
    parser.add_argument("--output-dir", default="runs/mmfi",
                        help="Checkpoints and results are written to <output-dir>/K<k>")
    parser.add_argument("--k", type=int, default=d.codebook_size, help="Codebook size K")
    parser.add_argument("--epochs", type=int, default=d.epochs)
    parser.add_argument("--lr", type=float, default=d.lr)
    parser.add_argument("--tau", type=float, default=d.tau, help="Soft-assignment temperature")
    parser.add_argument("--lambda-rate", type=float, default=d.lambda_rate, help="Rate-loss weight")
    parser.add_argument("--lambda-ent", type=float, default=d.lambda_ent, help="Entropy-model loss weight")
    parser.add_argument("--lambda-rec", type=float, default=d.lambda_rec, help="CSI reconstruction weight")
    parser.add_argument("--warmup-epochs", type=int, default=d.warmup_epochs,
                        help="Epochs of representation (first half) and entropy-model (second half) warm-up")
    parser.add_argument("--train-index-error-rate", type=float, default=d.train_index_error_rate,
                        help="Probability of replacing a VQ index by a random one during training")
    parser.add_argument("--adv-epsilon", type=float, default=d.adv_epsilon)
    parser.add_argument("--prefix-ratios", type=parse_floats, default=d.prefix_ratios)
    parser.add_argument("--test-error-rates", type=parse_floats, default=d.test_error_rates)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Cap samples per partition (quick smoke test)")
    args = parser.parse_args()

    cfg = TrainConfig(
        codebook_size=args.k,
        epochs=args.epochs,
        lr=args.lr,
        tau=args.tau,
        lambda_rate=args.lambda_rate,
        lambda_ent=args.lambda_ent,
        lambda_rec=args.lambda_rec,
        warmup_epochs=args.warmup_epochs,
        train_index_error_rate=args.train_index_error_rate,
        adv_epsilon=args.adv_epsilon,
        prefix_ratios=args.prefix_ratios,
        test_error_rates=args.test_error_rates,
    )
    out_dir = os.path.join(args.output_dir, f"K{cfg.codebook_size}")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(asdict(cfg), f, indent=2)

    set_reproducibility(args.seed, args.deterministic)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Run] config={asdict(cfg)}\n[Run] output={out_dir}", flush=True)

    loaders = build_mmfi_dataloaders(args.data_root, args.split_config, args.max_samples)
    model = TinySensePlus(cfg.codebook_size, train_index_error_rate=cfg.train_index_error_rate).to(device)
    discriminator = Discriminator().to(device)

    best_path = train(model, discriminator, loaders, cfg, device, out_dir)

    print("Training complete. Running rate-distortion evaluation with the best checkpoint.")
    best_model, _ = load_checkpoint(best_path, device)
    evaluate(
        best_model, loaders["test"], device, cfg.prefix_ratios, cfg.test_error_rates,
        out_json=os.path.join(out_dir, "results.json"),
    )


if __name__ == "__main__":
    main()
