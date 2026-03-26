# Complex-LR

**Complex-Valued CNN for SAR Automatic Target Recognition, trained with Adaptive Phase Gradient Descent**

> Wirtinger gradients · Complex learning rates η = ρe^{iθ} · Armijo backtracking · Phase-space saddle escape

*Forked from [Sayan Ghosh](https://github.com/sayanghosh). Mathematical framework and APGD theory by Sayan Ghosh. Architecture redesign, optimizer engineering (v7→v8), and all experiments by [Hardik Gogia](https://github.com/hardikgogia29).*

---

## What this is

Synthetic Aperture Radar images are **complex-valued by nature** — every pixel carries both an amplitude *and* a phase. Standard deep learning pipelines discard the phase, throwing away physically meaningful signal. This project keeps it.

The full pipeline is complex-valued end-to-end: complex weights, complex convolutions, a phase-aware activation function, and gradients computed via **Wirtinger calculus** — the mathematically correct way to differentiate a real-valued loss over complex parameters.

The novel piece is **APGD (Adaptive Phase Gradient Descent)** — an optimizer that uses a *complex learning rate* η = ρe^{iθ} and searches for the optimal phase angle θ at every step via Armijo backtracking and a foveated phase grid.

**Result: 94.4% accuracy on 10-class MSTAR SAR target recognition, 6,549 training samples.**

---

## The Math (in brief)

### Why Wirtinger?

A complex-valued loss L : ℂᴾ → ℝ is not holomorphic — the classical complex derivative is undefined. Wirtinger calculus treats w and w̄ as formally independent:

$$\nabla_{\bar{w}} \mathcal{L} = \frac{\partial \mathcal{L}}{\partial \bar{w}} \triangleq \frac{1}{2}\left(\frac{\partial \mathcal{L}}{\partial u} + i\frac{\partial \mathcal{L}}{\partial v}\right)$$

PyTorch computes exactly this when `.backward()` is called on `complex64` parameters.

### The complex descent lemma

For η = ρe^{iθ}, the update w ← w − η·∇_w̄L produces:

$$\mathcal{L}(w_{t+1}) \approx \mathcal{L}(w_t) - 2\rho\cos(\theta)\,\|\nabla_{\bar{w}}\mathcal{L}\|^2$$

Descent requires **cos(θ) > 0**, i.e. θ ∈ (−π/2, π/2). Under L-smoothness the tight bound is ρ < 4cos(θ)/L. The imaginary part of η rotates the update in complex parameter space — enabling exploration of directions unreachable with a real learning rate.

### APGD per-step algorithm

1. Forward + backward (train mode). Record f₀, raw ‖g‖
2. Clip gradients (norm ≤ 2.0), precondition → p
3. **Armijo backtracking** at θ=0 (eval mode, dropout off):
   `f_new ≤ f₀ − α·2ρ·‖p‖²` — shrink ρ ← βρ until accepted
4. **Saddle detection:** if bt_iters ≥ `saddle_bt_thresh`:
   - Probe K phase candidates (foveated grid, dense in ±45°) at scale ρ₀
   - Re-run Armijo with best θ*
5. Apply w ← w − ρ·e^{iθ*}·p
6. **ρ recovery:** after 20 consecutive clean steps, grow base_lr by 5% toward original

The saddle signal is backtracking exhaustion — not ‖g‖ < τ, which fails here because Wirtinger gradient norms stay at 3–15 throughout training regardless of convergence.

---

## Architecture

```
Input: (B, 1, 128, 128) complex64      z = A·e^{iφ}
│
├── ComplexConv2d(1→32)   + Cardioid
├── ComplexConv2d(32→64)  + Cardioid
├── ComplexAvgPool2d ──────────────── 64×64
├── ComplexConv2d(64→64)  + Cardioid
├── ComplexAvgPool2d ──────────────── 32×32
├── ComplexConv2d(64→128) + Cardioid
├── ComplexConv2d(128→128)+ Cardioid
├── ComplexAvgPool2d ──────────────── 16×16
│
├── ComplexGlobalAvgPool2d ──────────  (B, 128)
├── ComplexDropout(p=0.3)
├── ComplexLinear(128→256) + Cardioid
├── ComplexLinear(256→10)
└── |·| → (B, 10) float32 → CrossEntropyLoss
```

**1,244,042 complex parameters (≡ 2,488,084 real scalars)**

**Cardioid activation:** `f(z) = ½(1 + cos(arg z))·z` — passes signals aligned with the positive real axis, suppresses anti-aligned ones. Respects complex structure; not a split activation.

**2-call optimized convolution:** stacking real/imaginary channels reduces 4 conv calls to 2 per layer (~2× faster), mathematically identical.

---

## Results

### Overall

| Run | Optimizer | Best Val Acc | Epoch | Stable? |
|-----|-----------|:------------:|:-----:|:-------:|
| `apgd-v2` | SGD (no schedule) | 94.03% | 297 | ✗ — final acc 85.25% |
| `apgd-v2` | APGD v7 (no schedule) | 91.49% | 300 | ✓ |
| `v3_trained` | SGD + cosine anneal | 82.36% | 183 | ✓ |
| `v3_trained` | **APGD v8** (no schedule) | **94.41%** | **329** | **✓** |

SGD without a schedule peaks higher but oscillates — val accuracy swings >10pp in late training, final checkpoint 9pp below best. APGD reaches the same ceiling and holds it.

### Per-class: SGD vs APGD v8

| Class | SGD | APGD v8 | Δ |
|-------|:---:|:-------:|:---:|
| bmp2_tank | 79.6% | 92.3% | **▲ +12.7pp** |
| btr70_transport | 46.4% | 92.9% | **▲ +46.5pp** |
| t72_tank | 88.0% | 97.9% | **▲ +9.9pp** |
| d7_bulldozer | 86.0% | 99.2% | **▲ +13.2pp** |
| t62_tank | 94.4% | 98.4% | **▲ +4.0pp** |
| zil131_truck | 87.0% | 93.9% | **▲ +6.9pp** |
| btr60_transport | 44.0% | 59.3% | **▲ +15.3pp** |
| zsu23-4_gun | 97.8% | 98.2% | ▲ +0.4pp |
| brdm2_truck | 88.0% | 97.1% | **▲ +9.1pp** |
| 2s1_gun | 84.9% | 93.8% | **▲ +8.9pp** |
| **Overall** | **82.4%** | **94.4%** | **▲ +12.0pp** |

APGD wins every single class. The most dramatic gain is `btr70_transport` (+46.5pp) — the transport vehicle SGD consistently conflates with `btr60`. The persistent weak spot is `btr60_transport` (91 test samples) — the hardest class for both optimizers.

---

## Repository Structure

```
Complex-LR/
├── README.md
│
├── cvnn_model.py          # CV_CNN — complex conv, Cardioid activation, GAP head
├── optimizers_v7.py       # APGD v7 — uniform phase grid, fixed base_lr
├── optimizers_v8.py       # APGD v8 — foveated grid, ρ recovery, looser Armijo
│
├── apgd-v2.ipynb          # SGD vs APGD v7, no LR schedule, 300 epochs
└── v3_trained.ipynb       # SGD (cosine) vs APGD v8, 350 epochs — best results
```

---

## v7 → v8

| | v7 | v8 |
|--|----|----|
| Phase grid | Uniform, K=17 | Foveated — dense ±45°, sparse ±84° |
| base_lr | Fixed throughout | Self-healing: +5% every 20 clean steps |
| Armijo α | 0.3 (strict) | 0.05 (loose) |
| saddle_bt_thresh | 5 | 2 |
| **APGD best acc** | **91.49%** | **94.41%** |

---

## Quick Start

```python
from cvnn_model import CV_CNN
from optimizers_v8 import APGD_SGD
import torch.nn as nn

model     = CV_CNN(num_classes=10).cuda()
optimizer = APGD_SGD(model, base_lr=0.03, saddle_bt_thresh=2,
                     alpha=0.05, beta=0.5, K=17, max_backtrack=15, clip=2.0)
criterion = nn.CrossEntropyLoss()

# x: (B, 1, 128, 128) complex64  |  y: (B,) long
loss, theta, g_norm, regime = optimizer.step(x, y, criterion)
```

```python
from optimizers_v8 import ComplexSGD

# theta=0.0 → standard SGD on Wirtinger gradients
# theta≠0.0 → complex LR, rotates update in parameter space
optimizer = ComplexSGD(model, lr=0.03, theta=0.0)
```

**Requirements:** `torch`, `numpy`, `scikit-learn`, `matplotlib` · PyTorch 2.x · CUDA 11.8 · Kaggle T4 (~36s/epoch)

---

## Dataset

[MSTAR](https://www.sdms.afrl.af.mil/index.php?collection=mstar) — 10-class merged subset. X-band SAR, ground military vehicles, multiple depression angles. 128×128 complex64 images constructed as z = A·e^{iφ}.

`6,549 train · 2,596 val · 10 classes`

---

## License

MIT
