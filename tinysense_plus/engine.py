"""Training and rate-distortion evaluation loops."""

import json
import os
from dataclasses import asdict, dataclass, field
from typing import Dict, Iterable, List, Optional

import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from .metrics import (
    PCK_OVERALL_INDEX,
    compute_mpjpe,
    compute_pck,
    nmse_db,
    nmse_loss,
)
from .models.networks import Discriminator
from .models.tinysense_plus import TinySensePlus

PCK_THRESHOLDS = {"pck_50": 0.5, "pck_40": 0.4, "pck_30": 0.3, "pck_20": 0.2, "pck_10": 0.1, "pck_5": 0.05}


@dataclass
class TrainConfig:
    codebook_size: int = 512
    epochs: int = 100
    lr: float = 1e-3
    lr_step: int = 40
    lr_gamma: float = 0.1
    discriminator_lr: float = 1e-3
    lambda_rec: float = 0.1
    lambda_rate: float = 0.1
    lambda_ent: float = 1.0
    warmup_epochs: int = 2
    tau: float = 0.01
    train_index_error_rate: float = 0.1
    adv_epsilon: float = 0.0
    prefix_ratios: List[float] = field(default_factory=lambda: [1.0, 0.8, 0.6, 0.4, 0.2, 0.1])
    test_error_rates: List[float] = field(
        default_factory=lambda: [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    )


def _set_trainable(module: Optional[nn.Module], trainable: bool) -> None:
    if module is not None:
        for parameter in module.parameters():
            parameter.requires_grad_(trainable)


def training_stage(epoch: int, warmup_epochs: int) -> str:
    """``representation`` -> ``entropy`` -> ``joint``; only ``joint`` if no warm-up."""
    if warmup_epochs <= 0:
        return "joint"
    if epoch < warmup_epochs // 2:
        return "representation"
    if epoch < warmup_epochs:
        return "entropy"
    return "joint"


def _apply_stage(model: TinySensePlus, stage: str) -> None:
    representation = [
        model.encoder,
        model.pre_vq_conv,
        model.latent_pool,
        model.pose_head,
        model.decoder,
    ]
    if stage == "representation":
        for m in representation + [model.vq]:
            _set_trainable(m, True)
    elif stage == "entropy":
        for m in representation + [model.vq]:
            _set_trainable(m, False)
    elif stage == "joint":
        for m in representation:
            _set_trainable(m, True)
        _set_trainable(model.vq, False)
    else:
        raise ValueError(f"Unknown training stage: {stage}")
    _set_trainable(model.entropy_model, True)


def _lambda_rate(cfg: TrainConfig, stage: str, epoch: int) -> float:
    """Rate-loss weight: 0 in the first stage, constant in the entropy stage and
    linearly ramped from 0 to ``lambda_rate`` over the joint stage."""
    if stage == "representation":
        return 0.0
    if stage == "entropy" or cfg.warmup_epochs <= 0:
        return cfg.lambda_rate
    joint_epochs = max(1, cfg.epochs - cfg.warmup_epochs - 1)
    progress = max(0.0, min(1.0, (epoch - cfg.warmup_epochs) / joint_epochs))
    return progress * cfg.lambda_rate


def _train_step(model, discriminator, batch, optimizer, optimizer_d, cfg, stage, lambda_rate, device):
    bce = nn.BCELoss().to(device)
    mse = nn.MSELoss().to(device)

    optimizer_d.zero_grad()
    csi = batch["input_wifi-csi"].clone().detach().to(device).float()
    target = batch["output"][:, :, 0:2].to(device)
    batch_size = csi.size(0)
    real_label = torch.ones(batch_size, 1, device=device)
    fake_label = torch.zeros(batch_size, 1, device=device)
    detach_rate_target = stage == "entropy"

    out = model(csi, is_test=False, tau=cfg.tau, detach_rate_target=detach_rate_target)

    # Discriminator terms on real CSI and on the reconstruction.
    bce(discriminator(csi), real_label).backward(retain_graph=True)
    bce(discriminator(out.recon), fake_label).backward(retain_graph=True)

    # Second and third forward passes. The model's gradients are cleared and then
    # re-accumulated from (i) the NMSE of a second reconstruction, (ii) the
    # discriminator loss on a third reconstruction, and (iii) the total loss below.
    # With ``adv_epsilon == 0`` the "perturbed" CSI equals the input; with
    # ``adv_epsilon > 0`` it is a one-step FGSM perturbation of the input.
    csi_adv = csi.clone().requires_grad_(True)
    recon_adv = model(csi_adv, is_test=False, tau=cfg.tau, detach_rate_target=detach_rate_target).recon
    optimizer.zero_grad()
    nmse_loss(csi_adv, recon_adv).backward(retain_graph=True)
    perturbed = (csi_adv + cfg.adv_epsilon * csi_adv.grad.sign()).detach()

    recon_perturbed = model(
        perturbed, is_test=False, tau=cfg.tau, detach_rate_target=detach_rate_target
    ).recon
    bce(discriminator(recon_perturbed), fake_label).backward(retain_graph=True)

    pose_loss = mse(out.pose, target)
    rec_loss = mse(out.recon, perturbed)
    if stage == "representation":
        loss = out.vq_loss + cfg.lambda_rec * rec_loss + pose_loss + cfg.lambda_ent * out.entropy_loss
    elif stage == "entropy":
        loss = lambda_rate * out.rate_loss + cfg.lambda_ent * out.entropy_loss
    else:
        loss = (
            lambda_rate * out.rate_loss
            + cfg.lambda_ent * out.entropy_loss
            + out.vq_loss
            + cfg.lambda_rec * rec_loss
            + pose_loss
        )
    loss.backward()
    optimizer.step()
    optimizer_d.step()
    return loss.item()


@torch.no_grad()
def _validate(model, loader, cfg, device) -> Dict[str, float]:
    """Validation pass.

    This uses the *training* forward path (``is_test=False``): VQ indices are
    corrupted with ``train_index_error_rate`` and no entropy coding is performed.
    """
    model.eval()
    errors, pck_20, pck_5, nmse = [], [], [], []
    for batch in loader:
        csi = batch["input_wifi-csi"].clone().detach().to(device).float()
        target = batch["output"][:, :, 0:2]
        out = model(csi, is_test=False, tau=cfg.tau)
        pred = out.pose.cpu()
        errors.append(compute_mpjpe(pred, target))
        pck_20.append(compute_pck(pred.transpose(1, 2), target.transpose(1, 2), 0.2))
        pck_5.append(compute_pck(pred.transpose(1, 2), target.transpose(1, 2), 0.05))
        nmse.append(nmse_db(csi, out.recon).item())
    return {
        "mpjpe": float(np.mean(errors, 0)[0]),
        "pck_20": float(np.mean(pck_20, 0)[PCK_OVERALL_INDEX]),
        "pck_5": float(np.mean(pck_5, 0)[PCK_OVERALL_INDEX]),
        "nmse": float(np.mean(nmse)),
    }


def save_checkpoint(path: str, model: TinySensePlus, cfg: TrainConfig, **extra) -> None:
    torch.save(
        {"state_dict": model.state_dict(), "config": asdict(cfg), **extra}, path
    )


def load_checkpoint(path: str, device) -> tuple:
    """Rebuild a model from a checkpoint written by :func:`save_checkpoint`."""
    ckpt = torch.load(path, map_location=device, weights_only=True)
    cfg = TrainConfig(**ckpt["config"])
    model = TinySensePlus(
        cfg.codebook_size, train_index_error_rate=cfg.train_index_error_rate
    ).to(device)
    model.load_state_dict(ckpt["state_dict"])
    return model, cfg


def train(
    model: TinySensePlus,
    discriminator: Discriminator,
    loaders: Dict[str, Iterable],
    cfg: TrainConfig,
    device: torch.device,
    out_dir: str,
) -> str:
    """Train for ``cfg.epochs`` epochs; returns the path of the best checkpoint.

    The best epoch (searched from the end of the warm-up on) maximises
    ``PCK20 - 100 * MPJPE - NMSE(dB)`` on the validation set.
    """
    os.makedirs(out_dir, exist_ok=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=cfg.discriminator_lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg.lr_step, gamma=cfg.lr_gamma)

    best_score = -float("inf")
    best_path = os.path.join(out_dir, "best.pt")
    message = ""

    for epoch in tqdm(range(cfg.epochs)):
        stage = training_stage(epoch, cfg.warmup_epochs)
        _apply_stage(model, stage)
        lambda_rate = _lambda_rate(cfg, stage, epoch)
        print(
            f"[Train] stage={stage} codebook={cfg.codebook_size} "
            f"epoch={epoch + 1}/{cfg.epochs} "
            f"lambda_rate={lambda_rate:.3f} lambda_ent={cfg.lambda_ent:.3f}",
            flush=True,
        )

        model.train()
        for batch in loaders["train"]:
            _train_step(model, discriminator, batch, optimizer, optimizer_d, cfg, stage, lambda_rate, device)

        val = _validate(model, loaders["valid"], cfg, device)

        if cfg.warmup_epochs > 0 and epoch == cfg.warmup_epochs:
            best_score = -float("inf")
        if epoch >= cfg.warmup_epochs:
            score = val["pck_20"] - val["mpjpe"] * 100.0 - val["nmse"]
            if score > best_score:
                best_score = score
                save_checkpoint(best_path, model, cfg, epoch=epoch + 1, score=score)
                message = (
                    f"\n[NEW BEST] Epoch {epoch + 1} with codebook size {cfg.codebook_size} "
                    f"Score: {score:.2f}, PCK5: {val['pck_5']:.3f}, PCK20: {val['pck_20']:.3f}, "
                    f"MPJPE(valid): {val['mpjpe']:.3f}, NMSE(valid): {val['nmse']:.3f}"
                )

        save_checkpoint(os.path.join(out_dir, "last.pt"), model, cfg, epoch=epoch + 1)
        scheduler.step()

    print(message)
    return best_path


@torch.no_grad()
def evaluate(
    model: TinySensePlus,
    loader: Iterable,
    device: torch.device,
    prefix_ratios: List[float],
    error_rates: List[float],
    out_json: Optional[str] = None,
) -> List[dict]:
    """Rate-distortion sweep over prefix ratios and index-error rates on the test set.

    For each (prefix ratio, error rate) pair the first ``prefix_ratio`` of the VQ
    indices is arithmetic-coded, optionally corrupted by index substitution, and the
    missing suffix is generated by the entropy model.
    """
    model.eval()
    results = []
    reference_bytes = None

    for prefix_ratio in prefix_ratios:
        for error_rate in error_rates:
            print(
                f"\n--- Testing with Prefix Ratio: {prefix_ratio * 100:.1f}%, Error Rate: {error_rate:.1f} ---"
            )
            errors, nmse, coded, fixed, raw = [], [], [], [], []
            pck = {name: [] for name in PCK_THRESHOLDS}

            for batch in loader:
                csi = batch["input_wifi-csi"].clone().detach().to(device).float()
                target = batch["output"][:, :, 0:2]
                out = model(csi, is_test=True, prefix_ratio=prefix_ratio, error_rate=error_rate)

                coded.append(out.coded_bytes)
                fixed.append(out.fixed_bytes)
                raw.append(float(csi.numel() * csi.element_size() / max(1, csi.size(0))))
                nmse.append(nmse_db(csi, out.recon).item())
                pred = out.pose.cpu()
                errors.append(compute_mpjpe(pred, target))
                for name, threshold in PCK_THRESHOLDS.items():
                    pck[name].append(
                        compute_pck(pred.transpose(1, 2), target.transpose(1, 2), threshold)
                    )

            mpjpe, pa_mpjpe = np.mean(errors, 0)
            pck_mean = {n: float(np.mean(v, 0)[PCK_OVERALL_INDEX]) for n, v in pck.items()}
            mean_coded, mean_fixed, mean_raw = (float(np.mean(v)) for v in (coded, fixed, raw))
            if prefix_ratio == 1.0 and reference_bytes is None:
                reference_bytes = mean_coded

            row = {
                "prefix_ratio": prefix_ratio,
                "error_rate": error_rate,
                "codebook_size": model.codebook_size,
                "nmse": float(np.mean(nmse)),
                **pck_mean,
                "mpjpe": float(mpjpe),
                "pa_mpjpe": float(pa_mpjpe),
                "raw_bytes": mean_raw,
                "coded_bytes": mean_coded,
                "fixed_bytes": mean_fixed,
                "compression_ratio": mean_raw / max(mean_coded, 1e-9),
                "gain_vs_fixed": mean_fixed / max(mean_coded, 1e-9),
                "gain_vs_prefix_100": (reference_bytes / max(mean_coded, 1e-9))
                if reference_bytes is not None
                else 1.0,
            }
            results.append(row)
            _print_row(row)

    if out_json:
        with open(out_json, "w") as f:
            json.dump(results, f, indent=2)
    return results


def _print_row(r: dict) -> None:
    print(
        "test result: nmse: %.3f, pck_50: %.3f, pck_40: %.3f, pck_30: %.3f, pck_20: %.3f, "
        "pck_10: %.3f, pck_5: %.3f, mpjpe: %.3f, pa_mpjpe: %.3f"
        % (r["nmse"], r["pck_50"], r["pck_40"], r["pck_30"], r["pck_20"], r["pck_10"], r["pck_5"], r["mpjpe"], r["pa_mpjpe"])
    )
    print("Average Original Bytes per CSI sample: %.2f bytes" % r["raw_bytes"])
    print("Average Transmitted Bytes per CSI sample: %.2f bytes" % r["coded_bytes"])
    print("Compression Ratio (Original/Transmitted): %.2fx" % r["compression_ratio"])
    print("Bitrate Reduction vs Original: %.2f%%" % ((1.0 - r["coded_bytes"] / max(r["raw_bytes"], 1e-9)) * 100.0))
    print("Fixed-Length Bytes per CSI sample: %.2f bytes" % r["fixed_bytes"])
    print("VLC Compression Gain vs Fixed-Length: %.2fx" % r["gain_vs_fixed"])
    print("VLC Bitrate Saving vs Fixed-Length: %.2f%%" % ((1.0 - r["coded_bytes"] / max(r["fixed_bytes"], 1e-9)) * 100.0))
    print("Compression Gain vs Prefix 100%%: %.2fx" % r["gain_vs_prefix_100"])
