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

## What the code does

These details affect how the reported numbers should be read.

- **Representation.** The encoder output is average-pooled to a 256 x 7 x 1 latent and reshaped into
  32 tokens of dimension 56, so each CSI frame becomes 32 VQ indices (9 bits each with K = 512).
  The codebook is updated by exponential moving average.
- **Training stages.** With `--warmup-epochs W`, the first `W/2` epochs train the representation
  with the entropy model, the next `W/2` epochs train only the entropy model with the rate loss, and
  all remaining epochs train everything jointly while the rate-loss weight ramps linearly from 0 to
  `--lambda-rate`.
- **Rate loss.** The expected code length of a latent vector under the entropy model, using a
  softmax over negative Euclidean distances to the codewords with temperature `--tau`. It is
  averaged over tokens and measured in nats.
- **Index corruption.** `--train-index-error-rate` (default 0.1) replaces VQ indices by uniformly
  random ones during training. At test time the same substitution is applied to the decoded
  indices with probability equal to the error rate. It acts on the index map, not on the
  arithmetic-coded bitstream, and does not model packetisation.
- **Validation.** Per-epoch validation uses the training forward path, so it includes the index
  corruption and no entropy coding. The best epoch maximises `PCK20 - 100 * MPJPE - NMSE(dB)`.
- **Coded size.** `coded_bytes` is the length of the `torchac` payload per frame; headers, framing
  and model synchronisation are not counted. Streams are verified by decoding with probabilities
  computed from the true indices (teacher forcing), which checks that a stream round-trips but is
  not a sequential receiver.
- **Prefix completion.** The missing suffix is generated greedily (argmax) by the entropy model.
- **Pose metrics.** PCK is normalised by the torso length (joints 1 and 11); MPJPE and PA-MPJPE are
  in the dataset's 2-D coordinate units.
- **Data split.** `configs/mmfi_split.yaml` uses all 27 actions with a `random_split`: for each
  action a separate random permutation of the 40 subjects puts 80% of subjects in training. The
  held-out frames are then split 50/50 at frame level into validation and test sets. Other
  splits supported by the MM-Fi toolbox (`cross_scene_split`, `cross_subject_split`,
  `manual_split`) can be selected in the config.
- **Training step.** Each step runs three forward passes of the generator. Generator gradients
  accumulated before the second pass are cleared, and the update uses the NMSE of the second
  reconstruction, the discriminator loss on the third reconstruction, and the total loss.
  `--adv-epsilon` (default 0) sets the size of an optional FGSM perturbation used by the second
  and third passes.

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
