# TinySense+

Code for **TinySense+: Entropy-Aware CSI Compression for Scalable and Adaptive Wi-Fi Sensing**.

TinySense+ extends [TinySense](https://github.com/Toandinh1/TinySense-PerCom2026): the edge device
encodes Wi-Fi CSI into a short sequence of VQ indices, a learned causal entropy model drives
arithmetic coding of those indices, rate-aware training makes the indices cheaper to code, and a
prefix of the sequence can be sent with the rest completed at the server. The server reconstructs
the CSI and predicts 2-D human pose.

- Project page: <https://phucngvinuni.github.io/TinySense/>
- Preprint: <https://arxiv.org/abs/2601.15838>

This repository covers the **MM-Fi** experiments (WiFi-CSI amplitude, 17 keypoints).

## Repository layout

```
train.py                      train + rate-distortion evaluation
evaluate.py                   evaluate a saved checkpoint
configs/mmfi_split.yaml       MM-Fi split and loader settings
scripts/train_mmfi_k512.sh    example run (K = 512)
tinysense_plus/
  models/                     encoder, decoder, discriminator, pose head, VQ + entropy model
  data/                       MM-Fi WiFi-CSI dataset and loaders
  engine.py                   training loop, validation, rate-distortion evaluation
  metrics.py                  NMSE, PCK, MPJPE, PA-MPJPE
  channel.py                  synthetic index-corruption channel
```

## Installation

```bash
pip install -r requirements.txt
```

Tested with Python 3.12, PyTorch 2.12 (CUDA), `vector-quantize-pytorch` 1.29.1 and `torchac` 0.9.3.
`torchac` compiles a small C++ extension on first import, so a C++ compiler and `ninja` must be
available. Training needs a CUDA GPU (about 8 GB of memory at batch size 128) and roughly 8 GB of
host RAM, because the validation and test frames are held in memory.

## Data

Download MM-Fi and point the code at the folder that contains `E01 ... E04`:

```
<MMFI_ROOT>/E01/S01/A01/wifi-csi/frame001.mat
<MMFI_ROOT>/E01/S01/A01/ground_truth.npy
```

Only the WiFi-CSI frames and `ground_truth.npy` are used; the first two coordinates of the 17
keypoints are the regression target.

## Training

```bash
export MMFI_ROOT=/path/to/MMFi_Dataset
bash scripts/train_mmfi_k512.sh
```

which runs

```bash
python train.py --data-root "$MMFI_ROOT" --k 512 --epochs 100 --tau 0.01 \
    --lambda-rate 0.1 --lambda-ent 1 --warmup-epochs 2
```

Checkpoints (`best.pt`, `last.pt`), `config.json` and `results.json` are written to
`runs/mmfi/K<k>/`. One epoch took about 8 minutes on an RTX 4070; the final evaluation sweep over
all prefix ratios and error rates takes an hour or more.

A quick check that the setup works (a few seconds):

```bash
python train.py --data-root "$MMFI_ROOT" --output-dir runs/smoke --epochs 3 \
    --max-samples 256 --prefix-ratios 1.0,0.5 --test-error-rates 0.0,0.3
```

### Evaluation

```bash
python evaluate.py runs/mmfi/K512/best.pt --data-root "$MMFI_ROOT" \
    --prefix-ratios 1.0,0.8,0.6,0.4,0.2,0.1 --test-error-rates 0.0,0.1,0.3 \
    --output-json results.json
```

For every (prefix ratio, error rate) pair it reports NMSE (dB), PCK at thresholds 0.5 to 0.05 of
the torso length, MPJPE, PA-MPJPE, and the average coded size per CSI frame together with the
compression ratio against the 13,680-byte raw frame and against fixed-length index coding.

Checkpoints are plain `state_dict`s plus the training config and are loaded with
`torch.load(..., weights_only=True)`.


## Citation

```bibtex
@misc{nguyen2026tinysenseplus,
  title  = {TinySense+: Entropy-Aware CSI Compression for Scalable and Adaptive Wi-Fi Sensing},
  author = {Nguyen, Phuc H. and Gian, Toan and Do, Cuong D. and Chatzinotas, Symeon and Nguyen, Van-Dinh},
  year   = {2026},
  note   = {Preprint},
  url    = {https://arxiv.org/abs/2601.15838}
}
```

See [NOTICE.md](NOTICE.md) for third-party code and dependency notices.
