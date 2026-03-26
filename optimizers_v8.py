"""
optimizers.py — Complex-Valued Optimizers for CVNNs
=====================================================
Static LR  (θ fixed):   ComplexSGD | ComplexAdaGrad | ComplexAdam
Dynamic LR (θ adapted): APGD_SGD   | APGD_AdaGrad   | APGD_Adam

APGD Design (v8) — Pure Armijo Backtracking
─────────────────────────────────────────────
Saddle signal: bt_iters_with_θ=0 >= saddle_bt_thresh
If saddle fires: run phase search over Θ_K, re-run backtracking with θ*

Changes in this version (v8)
──────────────────────────────
1. dp uses p_grads (clipped+preconditioned gradient, not raw).
   Armijo: f_new ≤ f0 − α·2ρ cos(θ)·‖p‖²  calibrated to actual step applied.

2. _phase_search probes at rho0 (Armijo-consistent scale), not base_lr.

3. Best-seen rho returned on backtrack exhaustion (not last-tried rho).

4. _apply casts clr to complex64 (no silent float64 upcast).

5. eval() mode in _backtrack and _phase_search — dropout off,
   loss evaluations are deterministic per batch.

6. Pure Armijo backtracking (Goldstein removed).
   Goldstein bisection caused bt0=15 on every step because stochastic
   gradient noise (even with eval() mode) makes the bisection oscillate
   indefinitely. Pure Armijo shrinks rho monotonically and converges
   in 1-5 iterations under normal conditions. rho_min floor ensures
   the worst-case step is not numerically zero.
"""

import torch
import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
# Shared utilities
# ──────────────────────────────────────────────────────────────────────────────

def _grad_norm(grads: dict) -> float:
    sq = sum(g.norm() ** 2 for g in grads.values())
    return sq.sqrt().item()


def _snapshot(model) -> tuple[dict, dict]:
    grads, params = {}, {}
    for n, p in model.named_parameters():
        if p.grad is not None:
            grads[n]  = p.grad.clone()
            params[n] = p.data.clone()
    return grads, params


def _restore(model, params: dict):
    with torch.no_grad():
        for n, p in model.named_parameters():
            if n in params:
                p.data.copy_(params[n])


def _apply(model, params: dict, directions: dict, clr: complex):
    # Cast clr to complex64 — np.exp(1j*θ) is complex128, which would
    # silently upcast to float64 and then truncate back to float32.
    with torch.no_grad():
        for n, p in model.named_parameters():
            if n in directions:
                clr_t = torch.tensor(clr, dtype=directions[n].dtype,
                                     device=directions[n].device)
                p.data.copy_(params[n] - clr_t * directions[n])


# ──────────────────────────────────────────────────────────────────────────────
# Static LR Optimizers
# ──────────────────────────────────────────────────────────────────────────────

class ComplexSGD:
    """η = ρ·e^{iθ},  descent condition: ρ < 4cos(θ)/L  (Remark 7)."""
    def __init__(self, model, lr: float = 0.01, theta: float = 0.0):
        self.model      = model
        self.complex_lr = lr * np.exp(1j * theta)

    def zero_grad(self):  self.model.zero_grad()

    def step(self):
        grads, params = _snapshot(self.model)
        _apply(self.model, params, grads, self.complex_lr)


class ComplexAdaGrad:
    """AdaGrad + complex LR."""
    def __init__(self, model, lr: float = 0.01, theta: float = 0.0,
                 delta: float = 1e-8):
        self.model      = model
        self.delta      = delta
        self.G          = {}
        self.complex_lr = lr * np.exp(1j * theta)

    def zero_grad(self):  self.model.zero_grad()

    def step(self):
        grads, params = _snapshot(self.model)
        p = {}
        for n, g in grads.items():
            if n not in self.G:
                self.G[n] = torch.zeros_like(g.real)
            self.G[n] += g.abs() ** 2
            p[n] = g / (self.G[n] + self.delta).sqrt()
        _apply(self.model, params, p, self.complex_lr)


class ComplexAdam:
    """Adam + complex LR."""
    def __init__(self, model, lr: float = 0.001, theta: float = 0.0,
                 beta1: float = 0.9, beta2: float = 0.999, delta: float = 1e-8):
        self.model      = model
        self.beta1      = beta1
        self.beta2      = beta2
        self.delta      = delta
        self.m, self.v  = {}, {}
        self.t          = 0
        self.complex_lr = lr * np.exp(1j * theta)

    def zero_grad(self):  self.model.zero_grad()

    def step(self):
        self.t += 1
        grads, params = _snapshot(self.model)
        p = {}
        for n, g in grads.items():
            if n not in self.m:
                self.m[n] = torch.zeros_like(g)
                self.v[n] = torch.zeros_like(g.real)
            self.m[n] = self.beta1 * self.m[n] + (1 - self.beta1) * g
            self.v[n] = self.beta2 * self.v[n] + (1 - self.beta2) * g.abs() ** 2
            m_hat = self.m[n] / (1 - self.beta1 ** self.t)
            v_hat = self.v[n] / (1 - self.beta2 ** self.t)
            p[n]  = m_hat / (v_hat.sqrt() + self.delta)
        _apply(self.model, params, p, self.complex_lr)


# ──────────────────────────────────────────────────────────────────────────────
# APGD Base — v8
# ──────────────────────────────────────────────────────────────────────────────

class _APGDBase:
    """
    Per-step logic:
    1. Forward + backward in train() mode. Record raw ‖g‖.
    2. Clip grads, snapshot, precondition.
    3. Pure Armijo backtracking in eval() mode with θ=0. Count bt_iters.
       dp = ||p_grads||^2 (clipped gradient norm) — calibrated to actual step.
    4. If bt_iters >= saddle_bt_thresh: phase search (eval() mode),
       re-run Armijo with θ*.
    5. Apply update.
    """

    def __init__(self, model, base_lr: float = 0.01,
                 saddle_bt_thresh: int = 10,
                 alpha: float = 0.3, beta: float = 0.5,
                 K: int = 9, max_backtrack: int = 15, clip: float = 2.0,
                 verbose=False):
        self.model            = model
        self.base_lr          = base_lr
        self.saddle_bt_thresh = saddle_bt_thresh
        self.alpha            = alpha
        self.beta             = beta
        self.max_backtrack    = max_backtrack
        self.clip             = clip
        self._step_count      = 0
        self._orig_base_lr    = base_lr
        self._consec_bt1      = 0          # consecutive steps where bt0 == 1
        self._rho_grow_every  = 20         # recover after 20 consecutive full steps
        self._rho_grow_factor = 1.05       # grow by 5% per recovery event

        if verbose is True:
            self._print_every = 1
        elif not verbose:
            self._print_every = 0
        else:
            self._print_every = int(verbose)

        # Foveated grid: dense in ±45° where descent force > 70%, sparse outside
        _half = K // 2
        _inner = torch.linspace(-np.pi / 4, np.pi / 4, _half + 1).tolist()   # dense inner
        _outer_neg = torch.linspace(-np.pi / 2 + 0.1, -np.pi / 4, K - _half - 1, ).tolist()              # sparse outer-
        _outer_pos = torch.linspace( np.pi / 4,  np.pi / 2 - 0.1,
                                    K - _half - 1).tolist()                # sparse outer+
        raw = sorted(set(_outer_neg + _inner + _outer_pos))[:K]
        raw[raw.index(min(raw, key=abs))] = 0.0   # force exact 0 at centre
        self.theta_candidates = raw

        # Hard floor: APGD never takes a step smaller than this
        self._rho_min = base_lr * (beta ** (max_backtrack - 1))

    # ── Preconditioning ───────────────────────────────────────────────────────

    def _precondition(self, grads: dict) -> dict:
        return grads   # identity for SGD

    # ── Pure Armijo backtracking ──────────────────────────────────────────────

    def _backtrack(self, inputs, labels, criterion, params, p_grads,
                   theta: float, f0: float) -> tuple[float, int]:
        """
        Pure Armijo: f_new <= f0 - alpha * 2*rho*cos(theta) * Re<p, p>
        Shrinks rho by beta each iteration. Returns (best_seen_rho, n_iters).
        eval() mode disables dropout so the loss is deterministic per batch.

        dp uses p_grads (clipped+preconditioned) not raw_grads.
        Rationale: the actual step applied is rho*p_grads, so the correct
        directional derivative is Re<g_clip, p_grads>. For APGD_SGD,
        p_grads = g_clip, giving dp = ||g_clip||^2. Using raw_grads caused
        dp to scale as ||g_raw|| * ||g_clip||, making Armijo impossible to
        satisfy when g_raw >> clip threshold, forcing rho to near-zero and
        trapping the model in a high-gradient regime throughout training.
        """
        cos_t = np.cos(theta)
        if cos_t <= 1e-6:
            return 1e-8, 0

        # dp = Re<p_grads, p_grads> = ||p_grads||^2 — calibrated to actual step
        dp = sum(
            (p_grads[n].conj() * p_grads[n]).real.sum().item()
            for n in p_grads
        )
        if dp <= 0:
            return 1e-8, 0

        rho = self.base_lr
        best_rho, best_loss = rho, float('inf')

        self.model.eval()
        try:
            for i in range(self.max_backtrack):
                clr = rho * np.exp(1j * theta)
                _apply(self.model, params, p_grads, clr)
                with torch.no_grad():
                    new_loss = criterion(self.model(inputs), labels).item()
                _restore(self.model, params)

                if new_loss < best_loss:
                    best_loss, best_rho = new_loss, rho

                if new_loss <= f0 - self.alpha * 2.0 * rho * cos_t * dp:
                    return rho, i + 1

                rho *= self.beta   # pure shrink

        except Exception:
            _restore(self.model, params)  # restore params on exception
            raise
        finally:
            self.model.train()

        return max(best_rho, self._rho_min), self.max_backtrack

    # ── Phase search ──────────────────────────────────────────────────────────

    def _phase_search(self, inputs, labels, criterion, params, p_grads,
                      probe_rho: float) -> tuple[float, dict]:
        """
        Probe each θ ∈ Θ_K at probe_rho (= rho0 from θ=0 backtrack).
        eval() mode for deterministic probes.
        """
        best_loss, best_theta = float('inf'), 0.0
        phase_losses = {}

        self.model.eval()
        try:
            with torch.no_grad():
                for theta in self.theta_candidates:
                    clr  = probe_rho * np.exp(1j * theta)
                    _apply(self.model, params, p_grads, clr)
                    loss = criterion(self.model(inputs), labels).item()
                    phase_losses[theta] = loss
                    if loss < best_loss:
                        best_loss, best_theta = loss, theta
            _restore(self.model, params)  # always restore before leaving try
        except Exception:
            _restore(self.model, params)  # restore on exception too
            raise
        finally:
            self.model.train()

        return best_theta, phase_losses

    # ── Debug printer ─────────────────────────────────────────────────────────

    def _debug_print(self, step, f0, g_raw, g_clipped,
                     regime, theta, rho, bt0_iters, bt1_iters,
                     phase_losses=None):
        print(
            f"  [APGD step {step:5d}]  "
            f"loss={f0:.4f}  g_raw={g_raw:.3f}  g_clip={g_clipped:.3f}  "
            f"bt0={bt0_iters:2d}  "
            f"regime={'SADDLE' if regime == 'saddle' else 'normal':6s}  "
            f"θ*={theta:+.3f}rad  rho={rho:.5f}  bt1={bt1_iters:2d}"
        )
        if phase_losses:
            entries = "  ".join(
                f"{np.degrees(th):+.0f}°→{l:.4f}{'★' if th == theta else ' '}"
                for th, l in sorted(phase_losses.items())
            )
            print(f"             phase scan: {entries}")

    # ── Public step ───────────────────────────────────────────────────────────

    def step(self, inputs, labels, criterion) -> tuple:
        """
        Returns (loss, theta, gamma, regime).
        loss:   float — batch loss before update
        theta:  float — phase used
        gamma:  float — raw ‖g‖
        regime: str   — 'normal' | 'saddle' | 'converged'
        """
        self._step_count += 1

        # Forward + backward in train() mode
        self.model.zero_grad()
        out  = self.model(inputs)
        loss = criterion(out, labels)
        f0   = loss.item()
        loss.backward()

        # Raw gradients BEFORE clipping (for g_raw metric only, not used in Armijo)
        raw_grads = {n: p.grad.clone()
                     for n, p in self.model.named_parameters()
                     if p.grad is not None}
        g_raw = _grad_norm(raw_grads)

        if g_raw < 1e-10:
            return f0, 0.0, g_raw, 'converged'

        if self.clip > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)

        grads, params = _snapshot(self.model)
        g_clipped     = _grad_norm(grads)
        p_grads       = self._precondition(grads)

        # Step 1: Armijo with θ=0
        rho0, bt0_iters = self._backtrack(
            inputs, labels, criterion, params, p_grads, 0.0, f0)

        # Step 2: saddle check
        phase_losses = None
        # ── Rho recovery on consecutive clean steps ───────────────────────────────
        if bt0_iters == 1:
            self._consec_bt1 += 1
        elif self._consec_bt1 >= self._rho_grow_every:
            self.base_lr = min(self.base_lr * self._rho_grow_factor, self._orig_base_lr)
            self._consec_bt1 = 0
        else:
            self._consec_bt1 = 0
        
        if bt0_iters >= self.saddle_bt_thresh:
            theta, phase_losses = self._phase_search(
                inputs, labels, criterion, params, p_grads, rho0)
            if theta != 0.0:
                rho, bt1_iters = self._backtrack(inputs, labels, criterion, params, p_grads, theta, f0)
                regime = 'saddle'
            else:
                rho, bt1_iters = rho0, bt0_iters
                regime = 'saddle'    # ← phase search ran; theta=0 was confirmed optimal
        else:
            theta, rho, bt1_iters = 0.0, rho0, bt0_iters
            regime = 'normal'

        clr = rho * np.exp(1j * theta)
        _apply(self.model, params, p_grads, clr)
        del raw_grads

        if self._print_every > 0 and self._step_count % self._print_every == 0:
            self._debug_print(self._step_count, f0, g_raw, g_clipped,
                              regime, theta, rho, bt0_iters, bt1_iters,
                              phase_losses)

        return f0, theta, g_raw, regime


# ── Concrete variants ─────────────────────────────────────────────────────────

class APGD_SGD(_APGDBase):
    """APGD with vanilla SGD (identity preconditioning)."""
    pass


class APGD_AdaGrad(_APGDBase):
    """APGD with AdaGrad preconditioning."""
    def __init__(self, *args, delta: float = 1e-8, **kwargs):
        super().__init__(*args, **kwargs)
        self.G, self.delta = {}, delta

    def _precondition(self, grads: dict) -> dict:
        p = {}
        for n, g in grads.items():
            if n not in self.G:
                self.G[n] = torch.zeros_like(g.real)
            self.G[n] += g.abs() ** 2
            p[n] = g / (self.G[n] + self.delta).sqrt()
        return p


class APGD_Adam(_APGDBase):
    """APGD with Adam preconditioning."""
    def __init__(self, *args, beta1: float = 0.9, beta2: float = 0.999,
                 delta: float = 1e-8, **kwargs):
        super().__init__(*args, **kwargs)
        self.beta1, self.beta2, self.delta = beta1, beta2, delta
        self.m, self.v, self.t = {}, {}, 0

    def _precondition(self, grads: dict) -> dict:
        self.t += 1
        p = {}
        for n, g in grads.items():
            if n not in self.m:
                self.m[n] = torch.zeros_like(g)
                self.v[n] = torch.zeros_like(g.real)
            self.m[n] = self.beta1 * self.m[n] + (1 - self.beta1) * g
            self.v[n] = self.beta2 * self.v[n] + (1 - self.beta2) * g.abs() ** 2
            m_hat = self.m[n] / (1 - self.beta1 ** self.t)
            v_hat = self.v[n] / (1 - self.beta2 ** self.t)
            p[n]  = m_hat / (v_hat.sqrt() + self.delta)
        return p
