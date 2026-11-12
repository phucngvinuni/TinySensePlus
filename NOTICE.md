# Third-party notices

This repository builds on the following work.

- **TinySense** (PerCom 2026) — https://github.com/Toandinh1/TinySense-PerCom2026.
  The encoder/decoder design, SK blocks and the data pipeline originate from the
  TinySense code base.
- **MM-Fi** — the dataset reader in `tinysense_plus/data/mmfi.py` is derived from the
  MM-Fi toolbox and reduced to the WiFi-CSI modality. If you use MM-Fi, cite the
  MM-Fi paper (Yang et al., NeurIPS 2023, Datasets and Benchmarks Track).
- **minGPT** — `tinysense_plus/models/gpt.py` is adapted from Andrej Karpathy's
  minGPT (as vendored in taming-transformers), which is released under the MIT
  license:

  > MIT License
  >
  > Copyright (c) 2020 Andrej Karpathy
  >
  > Permission is hereby granted, free of charge, to any person obtaining a copy
  > of this software and associated documentation files (the "Software"), to deal
  > in the Software without restriction, including without limitation the rights
  > to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
  > copies of the Software, and to permit persons to whom the Software is
  > furnished to do so, subject to the following conditions:
  >
  > The above copyright notice and this permission notice shall be included in all
  > copies or substantial portions of the Software.
  >
  > THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
  > IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
  > FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
  > AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
  > LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
  > OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
  > SOFTWARE.

- **vector-quantize-pytorch** (Phil Wang) provides the EMA vector quantiser.
- **torchac** (Mentzer et al.) provides the arithmetic coder; it is distributed under
  the GNU GPL, which matters if you redistribute this code together with it.
- `tinysense_plus/metrics.py` includes a NumPy port of MATLAB's `procrustes` for
  PA-MPJPE.
