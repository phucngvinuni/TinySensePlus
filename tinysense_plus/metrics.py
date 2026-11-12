"""Reconstruction and pose-estimation metrics.

Pose tensors follow the layout used by the dataloader: predictions and targets
are (N, J, 2) 2-D keypoints in the dataset's own coordinate units.
"""

from typing import Tuple

import numpy as np
import torch

# MM-Fi (COCO-style, 17 joints): PCK is normalised by the distance between
# joints 1 and 11 (torso length).
PCK_REF_JOINTS = (1, 11)
NUM_JOINTS = 17
PCK_OVERALL_INDEX = NUM_JOINTS


def nmse_loss(input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean squared error divided by the variance of ``target`` (training loss)."""
    return torch.mean((input - target) ** 2) / torch.var(target)


def nmse_db(x: torch.Tensor, x_hat: torch.Tensor) -> torch.Tensor:
    """Per-sample NMSE averaged over the batch, in dB (lower is better)."""
    return 10 * torch.log10(
        torch.mean(
            torch.mean(torch.square(x - x_hat), dim=(1, 2, 3))
            / torch.mean(torch.square(x), dim=(1, 2, 3))
        )
    )


def compute_pck(pred: torch.Tensor, target: torch.Tensor, threshold: float) -> np.ndarray:
    """Percentage of Correct Keypoints.

    Args:
        pred, target: (N, 2, J) tensors.
        threshold: fraction of the torso length.

    Returns:
        Array of length J + 1: PCK (%) per joint, then the average over all joints.
    """
    dt = pred.detach().cpu().numpy()
    gt = target.detach().cpu().numpy()
    assert dt.shape[0] == gt.shape[0]
    num_joints = gt.shape[2]

    a, b = PCK_REF_JOINTS
    scale = np.sqrt(np.sum(np.square(gt[:, :, a] - gt[:, :, b]), 1))
    dist = np.sqrt(np.sum(np.square(dt - gt), 1)) / np.tile(scale, (num_joints, 1)).T

    pck = np.zeros(num_joints + 1)
    for j in range(num_joints):
        pck[j] = 100 * np.mean(dist[:, j] <= threshold)
    pck[num_joints] = 100 * np.mean(dist <= threshold)
    return pck


def _similarity_transform(X: np.ndarray, Y: np.ndarray):
    """Optimal similarity transform (rotation, scale, translation) of Y onto X.

    NumPy port of MATLAB's ``procrustes`` with optimal scaling.
    """
    muX, muY = X.mean(0), Y.mean(0)
    X0, Y0 = X - muX, Y - muY
    normX, normY = np.sqrt((X0**2.0).sum()), np.sqrt((Y0**2.0).sum())
    X0 /= normX
    Y0 /= normY

    U, s, Vt = np.linalg.svd(np.dot(X0.T, Y0), full_matrices=False)
    V = Vt.T
    T = np.dot(V, U.T)
    if np.linalg.det(T) < 0:
        V[:, -1] *= -1
        s[-1] *= -1
        T = np.dot(V, U.T)

    b = s.sum() * normX / normY
    c = muX - b * np.dot(muY, T)
    return T, b, c


def compute_mpjpe(pred: torch.Tensor, target: torch.Tensor) -> Tuple[float, float]:
    """MPJPE and Procrustes-aligned MPJPE for (N, J, 2) keypoints (CPU tensors)."""
    pred = pred.detach().cpu().numpy()
    target = target.detach().cpu().numpy()
    if np.isnan(pred).any() or np.isinf(pred).any():
        pred = np.nan_to_num(pred) + 1e-8
    assert pred.shape == target.shape, "Input shapes must match"

    mpjpe = np.mean(np.sqrt(np.sum(np.square(pred - target), axis=2)))

    pa = np.zeros(pred.shape[0])
    for n in range(pred.shape[0]):
        T, b, c = _similarity_transform(target[n], pred[n])
        aligned = (b * pred[n] @ T) + c
        pa[n] = np.mean(np.sqrt(np.sum(np.square(aligned - target[n]), axis=1)))
    return mpjpe, np.mean(pa)
