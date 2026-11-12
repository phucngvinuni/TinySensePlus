"""MM-Fi WiFi-CSI dataset (frame-level).

Derived from the MM-Fi toolbox, reduced to the WiFi-CSI modality. Split
definitions follow the toolbox: ``random_split``, ``cross_scene_split``,
``cross_subject_split`` and ``manual_split`` (see ``configs/mmfi_split.yaml``).
"""

import os
from typing import Dict, List

import numpy as np
import scipy.io as scio
import torch
from torch.utils.data import DataLoader, Dataset

MODALITY = "wifi-csi"
FRAMES_PER_SEQUENCE = 297

ALL_SUBJECTS = [f"S{i:02d}" for i in range(1, 41)]
ALL_ACTIONS = [f"A{i:02d}" for i in range(1, 28)]
SCENES = {
    "E01": ALL_SUBJECTS[0:10],
    "E02": ALL_SUBJECTS[10:20],
    "E03": ALL_SUBJECTS[20:30],
    "E04": ALL_SUBJECTS[30:40],
}
DAILY_ACTIONS = [
    "A02", "A03", "A04", "A05", "A13", "A14", "A17",
    "A18", "A19", "A20", "A21", "A22", "A23", "A27",
]  # fmt: skip
REHAB_ACTIONS = [
    "A01", "A06", "A07", "A08", "A09", "A10", "A11",
    "A12", "A15", "A16", "A24", "A25", "A26",
]  # fmt: skip
CROSS_SCENE_TRAIN = ALL_SUBJECTS[:30]
CROSS_SCENE_VAL = ALL_SUBJECTS[30:]


def scene_of(subject: str) -> str:
    for scene, subjects in SCENES.items():
        if subject in subjects:
            return scene
    raise ValueError("Subject does not exist in this dataset.")


def decode_split_config(config: dict) -> Dict[str, Dict[str, List[str]]]:
    """Return ``{subject: [actions]}`` for the training and held-out partitions."""
    protocol = config["protocol"]
    if protocol == "protocol1":
        actions = DAILY_ACTIONS
    elif protocol == "protocol2":
        actions = REHAB_ACTIONS
    else:
        actions = ALL_ACTIONS

    train_form: Dict[str, List[str]] = {}
    val_form: Dict[str, List[str]] = {}
    split = config["split_to_use"]

    if split == "random_split":
        seed = config["random_split"]["random_seed"]
        ratio = config["random_split"]["ratio"]
        for action in actions:
            # A fresh subject permutation is drawn for every action (seed + 1 each).
            idx = np.random.RandomState(seed).permutation(len(ALL_SUBJECTS))
            n_train = int(np.floor(ratio * len(ALL_SUBJECTS)))
            subjects_train = np.array(ALL_SUBJECTS)[idx[:n_train]].tolist()
            subjects_val = np.array(ALL_SUBJECTS)[idx[n_train:]].tolist()
            for subject in ALL_SUBJECTS:
                if subject in subjects_train:
                    train_form.setdefault(subject, []).append(action)
                if subject in subjects_val:
                    val_form.setdefault(subject, []).append(action)
            seed += 1
    elif split == "cross_scene_split":
        for subject in CROSS_SCENE_TRAIN:
            train_form[subject] = actions
        for subject in CROSS_SCENE_VAL:
            val_form[subject] = actions
    elif split == "cross_subject_split":
        for subject in config["cross_subject_split"]["train_dataset"]["subjects"]:
            train_form[subject] = actions
        for subject in config["cross_subject_split"]["val_dataset"]["subjects"]:
            val_form[subject] = actions
    else:
        manual = config["manual_split"]
        for subject in manual["train_dataset"]["subjects"]:
            train_form[subject] = manual["train_dataset"]["actions"]
        for subject in manual["val_dataset"]["subjects"]:
            val_form[subject] = manual["val_dataset"]["actions"]

    return {"train": train_form, "val": val_form}


def read_csi_frame(path: str) -> np.ndarray:
    """Load one CSI amplitude frame (3 antennas x 114 subcarriers x 10 packets),
    fill NaN/inf by the per-packet mean and min-max normalise the frame to [0, 1].
    """
    data = scio.loadmat(path)["CSIamp"]
    data[np.isinf(data)] = np.nan
    for i in range(10):
        column = data[:, :, i]
        if np.count_nonzero(column != column) != 0:
            valid = column[column == column]
            column[np.isnan(column)] = valid.mean()
    return (data - np.min(data)) / (np.max(data) - np.min(data))


class MMFiCSIDataset(Dataset):
    def __init__(self, data_root: str, data_form: Dict[str, List[str]]):
        self.data_root = data_root
        self.items = self._index(data_form)

    def _index(self, data_form: Dict[str, List[str]]) -> List[dict]:
        items = []
        for subject, actions in data_form.items():
            scene = scene_of(subject)
            for action in actions:
                sequence_dir = os.path.join(self.data_root, scene, subject, action)
                for idx in range(FRAMES_PER_SEQUENCE):
                    frame = os.path.join(
                        sequence_dir, MODALITY, "frame{:03d}.mat".format(idx + 1)
                    )
                    if os.path.getsize(frame) == 0:
                        continue
                    items.append(
                        {
                            "scene": scene,
                            "subject": subject,
                            "action": action,
                            "idx": idx,
                            "gt_path": os.path.join(sequence_dir, "ground_truth.npy"),
                            "csi_path": frame,
                        }
                    )
        return items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int) -> dict:
        item = self.items[i]
        keypoints = torch.from_numpy(np.load(item["gt_path"]))[item["idx"]]
        return {
            "scene": item["scene"],
            "subject": item["subject"],
            "action": item["action"],
            "idx": item["idx"],
            "output": keypoints,
            "input_wifi-csi": read_csi_frame(item["csi_path"]),
        }


def collate_batch(batch: List[dict]) -> dict:
    return {
        "scene": [s["scene"] for s in batch],
        "subject": [s["subject"] for s in batch],
        "action": [s["action"] for s in batch],
        "idx": [s["idx"] for s in batch],
        "output": torch.FloatTensor(np.array([np.array(s["output"]) for s in batch])),
        "input_wifi-csi": torch.FloatTensor(
            np.array([np.array(s["input_wifi-csi"]) for s in batch])
        ),
    }


def make_datasets(data_root: str, config: dict):
    forms = decode_split_config(config)
    return (
        MMFiCSIDataset(data_root, forms["train"]),
        MMFiCSIDataset(data_root, forms["val"]),
    )


def make_dataloader(dataset, is_training: bool, generator, batch_size: int) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collate_batch,
        shuffle=is_training,
        drop_last=is_training,
        generator=generator,
    )
